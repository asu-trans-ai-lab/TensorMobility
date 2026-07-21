from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensormobility.dta.algorithms import solve_exact_fw, solve_fw_gp, solve_logit_sue_msa
from tensormobility.core.instance import STBInstance, build_toy_instance, capacity_shock_multiplier
from tensormobility.dta.latent import build_route_family_atoms, solve_latent_fw_gp
from tensormobility.behavior.neural_router import solve_neural_certified_fw_gp, train_router
from tensormobility.core.objective import STBObjective


def flow_summary(instance: STBInstance, flow: np.ndarray) -> pd.DataFrame:
    frame = instance.columns.copy()
    frame["flow"] = flow
    return (
        frame.groupby(["destination", "mode", "period"], as_index=False)["flow"]
        .sum()
        .sort_values(["period", "destination", "mode"])
    )


def result_row(result, scenario: str) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario,
        "algorithm": result.name,
        "objective": result.objective,
        "relative_gap": result.relative_gap,
        "feasibility_residual": result.feasibility_residual,
        "active_columns": result.active_columns,
        "pricing_calls": result.pricing_calls,
        "wall_time_seconds": result.wall_time_seconds,
    }
    row.update(result.metadata)
    return row


def run_suite(output_dir: str | Path, seed: int = 7) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = build_toy_instance(seed)
    shock = base.with_capacity_multiplier(capacity_shock_multiplier(base), name="targeted_capacity_shock")

    router_training = train_router(base)
    router_training.history.to_csv(output / "neural_router_training.csv", index=False)

    records: list[dict[str, object]] = []
    all_results: dict[tuple[str, str], object] = {}
    base_rep = build_route_family_atoms(base)

    for scenario in (base, shock):
        scenario_dir = output / scenario.name
        scenario_dir.mkdir(exist_ok=True)

        results = []
        results.append(solve_logit_sue_msa(scenario))
        results.append(solve_exact_fw(scenario, max_iterations=600, tolerance=1e-6))
        results.append(solve_fw_gp(scenario))
        static, _ = solve_latent_fw_gp(
            scenario, representation=base_rep, adaptive=False, max_iterations_per_cycle=250
        )
        adaptive, _ = solve_latent_fw_gp(
            scenario, representation=base_rep, adaptive=True, max_iterations_per_cycle=250
        )
        results.extend([static, adaptive])
        results.append(
            solve_neural_certified_fw_gp(
                scenario, router_training.model, max_iterations=300, tolerance=1e-7
            )
        )

        for result in results:
            records.append(result_row(result, scenario.name))
            all_results[(scenario.name, result.name)] = result
            result.history.to_csv(scenario_dir / f"{result.name}_history.csv", index=False)
            flow_summary(scenario, result.flow).to_csv(
                scenario_dir / f"{result.name}_flow_summary.csv", index=False
            )
            detail = scenario.columns.copy()
            detail["flow"] = result.flow
            detail.to_csv(scenario_dir / f"{result.name}_column_flows.csv", index=False)

    summary = pd.DataFrame(records)
    summary.to_csv(output / "algorithm_comparison.csv", index=False)

    certified = (
        summary[summary["algorithm"].isin(["exact_fw", "fw_gp"])]
        .groupby("scenario", as_index=False)["objective"]
        .min()
        .rename(columns={"objective": "certified_full_space_objective"})
    )
    summary = summary.merge(certified, on="scenario", how="left")
    summary["objective_gap_vs_certified_full_space"] = (
        summary["objective"] - summary["certified_full_space_objective"]
    ) / summary["certified_full_space_objective"].abs().clip(lower=1.0)
    summary.to_csv(output / "algorithm_comparison_with_exact_gap.csv", index=False)

    # One plot per figure, with default matplotlib colors/styles.
    fig = plt.figure(figsize=(9, 5))
    for scenario_name, group in summary.groupby("scenario"):
        x = np.arange(len(group))
        plt.plot(x, group["objective_gap_vs_certified_full_space"], marker="o", label=scenario_name)
    algorithms = list(summary[summary["scenario"] == base.name]["algorithm"])
    plt.xticks(np.arange(len(algorithms)), algorithms, rotation=25, ha="right")
    plt.ylabel("Objective gap vs certified full-space solution")
    plt.title("STB-FTT algorithm objective comparison")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output / "objective_gap_comparison.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    for scenario_name, group in summary.groupby("scenario"):
        x = np.arange(len(group))
        plt.semilogy(x, group["relative_gap"].clip(lower=1e-12), marker="o", label=scenario_name)
    plt.xticks(np.arange(len(algorithms)), algorithms, rotation=25, ha="right")
    plt.ylabel("Reported relative optimality residual")
    plt.title("STB-FTT certification residuals")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output / "certificate_comparison.png", dpi=180)
    plt.close(fig)

    failure = summary[
        (summary["scenario"] == shock.name)
        & summary["algorithm"].isin(["static_latent_fw_gp", "adaptive_latent_fw_gp", "neural_certified_fw_gp"])
    ][
        [
            "algorithm",
            "objective_gap_vs_certified_full_space",
            "relative_gap",
            "active_columns",
            "pricing_calls",
            "promotions",
        ]
    ]
    failure.to_csv(output / "capacity_shock_failure_recovery.csv", index=False)

    metadata = {
        "name": "STB-FTT Algorithm Lab v0.1",
        "axes": ["Space", "Time", "Behavior"],
        "base_columns": base.n_columns,
        "groups": base.n_groups,
        "link_time_resources": base.n_resources,
        "initial_atoms": base_rep.n_atoms,
        "algorithms": algorithms,
        "important_scope": "Departure-period time expansion with convex BPR loading; dynamic queue DNL is an extension hook, not claimed in v0.1.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary
