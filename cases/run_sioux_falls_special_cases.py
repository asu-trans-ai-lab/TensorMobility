from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensormobility.dta.algorithms import fw_gap
from tensormobility.core.flow_through import build_flow_through_tensors
from tensormobility.dynamics.fluid_queue import build_departure_tensor, run_fluid_point_queue, smooth_departure_profile
from tensormobility.core.objective import STBObjective
from tensormobility.dta.sioux_falls import build_sioux_falls_path_set
from tensormobility.dta.special_cases import solve_case_1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/sioux_falls_special_cases")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    path_set = build_sioux_falls_path_set(k_paths=4)
    inst = path_set.instance
    path_set.nodes.to_csv(output / "nodes.csv", index=False)
    path_set.links.to_csv(output / "links.csv", index=False)
    inst.groups.to_csv(output / "od_groups.csv", index=False)
    inst.columns.to_csv(output / "paths.csv", index=False)

    ue = None
    temperature_rows = []
    logit_results = {}
    # Solve UE once, then a temperature ladder showing the logit->UE limit.
    for temperature in (5.0, 2.0, 1.0, 0.5, 0.2):
        logit, ue_candidate = solve_case_1(path_set, temperature=temperature)
        if ue is None:
            ue = ue_candidate
        logit_results[temperature] = logit
        link_distance = np.linalg.norm(inst.A @ logit.flow - inst.A @ ue.flow, ord=1) / max(np.linalg.norm(inst.A @ ue.flow, ord=1), 1.0)
        base_gap, _ = fw_gap(logit.flow, STBObjective(inst).gradient(logit.flow), inst.group_columns, inst.demands)
        temperature_rows.append({
            "temperature": temperature,
            "entropy_gap": logit.relative_gap,
            "deterministic_fw_gap_at_logit_flow": base_gap / max(abs(STBObjective(inst).value(logit.flow)), 1.0),
            "relative_link_flow_distance_to_ue": link_distance,
            "mass_residual": logit.feasibility_residual,
        })
    assert ue is not None
    temperature_df = pd.DataFrame(temperature_rows)
    temperature_df.to_csv(output / "case1_logit_to_ue.csv", index=False)

    chosen = logit_results[1.0]
    ftt = build_flow_through_tensors(inst, chosen.flow)
    np.savez_compressed(
        output / "case3_flow_through_tensors.npz",
        zone_generation=ftt.zone_generation,
        od_flow=ftt.od_flow,
        path_flow=ftt.path_flow,
        link_flow=ftt.link_flow,
        zone_to_od=ftt.zone_to_od,
        od_to_path=ftt.od_to_path,
        path_to_link=ftt.path_to_link,
    )

    profile = smooth_departure_profile()
    departures = build_departure_tensor(chosen.flow, profile)
    base_queue = run_fluid_point_queue(path_set, departures, horizon_steps=96)
    shock_multiplier = np.ones(inst.n_resources)
    # A central bottleneck shock on 10->16 and 16->17.
    labels = list(inst.resource_labels)
    shock_multiplier[labels.index("10->16")] = 0.45
    shock_multiplier[labels.index("16->17")] = 0.55
    shock_queue = run_fluid_point_queue(path_set, departures, horizon_steps=96, capacity_multiplier=shock_multiplier)
    base_queue.history.to_csv(output / "case2_fluid_queue_base.csv", index=False)
    shock_queue.history.to_csv(output / "case2_fluid_queue_shock.csv", index=False)

    summary = {
        "network": {"nodes": 24, "links": 76, "od_groups": inst.n_groups, "paths": inst.n_columns},
        "case_1_logit_sue": {
            "temperature_ladder": [5.0, 2.0, 1.0, 0.5, 0.2],
            "ue_relative_gap": ue.relative_gap,
            "ue_mass_residual": ue.feasibility_residual,
        },
        "case_2_fluid_queue": {
            "base_max_total_queue": base_queue.max_total_queue,
            "shock_max_total_queue": shock_queue.max_total_queue,
            "base_max_conservation_residual": float(np.max(np.abs(base_queue.conservation_residual))),
            "shock_max_conservation_residual": float(np.max(np.abs(shock_queue.conservation_residual))),
            "base_final_completed": base_queue.final_completed,
            "shock_final_completed": shock_queue.final_completed,
        },
        "case_3_flow_through_tensor": {
            "zone_od_mass_residual": ftt.mass_residual_zone_od,
            "od_path_mass_residual": ftt.mass_residual_od_path,
            "link_reconstruction_residual": ftt.link_reconstruction_residual,
            "zone_to_od_shape": list(ftt.zone_to_od.shape),
            "od_to_path_shape": list(ftt.od_to_path.shape),
            "path_to_link_shape": list(ftt.path_to_link.shape),
        },
    }
    (output / "special_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(8, 5))
    plt.loglog(temperature_df["temperature"], temperature_df["relative_link_flow_distance_to_ue"], marker="o")
    plt.xlabel("Logit temperature")
    plt.ylabel("Relative link-flow distance to deterministic UE")
    plt.title("Sioux Falls: Logit SUE approaches UE")
    plt.tight_layout()
    fig.savefig(output / "case1_logit_to_ue.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(base_queue.history["minutes"], base_queue.history["total_queue"], label="base")
    plt.plot(shock_queue.history["minutes"], shock_queue.history["total_queue"], label="capacity shock")
    plt.xlabel("Minutes")
    plt.ylabel("Total queued fluid")
    plt.title("Sioux Falls fluid point-queue evolution")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output / "case2_queue_evolution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
