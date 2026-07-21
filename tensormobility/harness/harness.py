from __future__ import annotations

"""Common simulation harness for STB-FTT special cases and expansion tests."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from tensormobility.behavior.activity import build_activity_stb_model, hierarchical_behavior_choice
from tensormobility.behavior.activity_dta import compare_behavior_shares, multi_start_stability, solve_activity_dta
from tensormobility.dta.algorithms import solve_exact_fw, solve_logit_sue_msa
from tensormobility.dynamics.cohort_queue import run_cohort_point_queue
from tensormobility.core.flow_through import build_flow_through_tensors
from tensormobility.dta.sioux_falls import build_sioux_falls_path_set


@dataclass(frozen=True)
class HarnessSummary:
    case0_ftt: dict[str, float | int | list[int]]
    case1_static: dict[str, float | int]
    case2_queue: dict[str, float | int]
    case3_activity: dict[str, float | int]
    case4_dynamic: dict[str, float | int | bool]
    scalability: dict[str, float | int]


def _gaussian_profile(n: int, center_fraction: float = 0.42, spread_fraction: float = 0.18) -> np.ndarray:
    x = np.arange(n, dtype=float)
    center = center_fraction * max(n - 1, 1)
    spread = max(spread_fraction * n, 1.0)
    p = np.exp(-0.5 * ((x - center) / spread) ** 2)
    return p / p.sum()


def run_scalability_profile(output: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k_paths, n_dep in ((2, 3), (2, 6), (3, 6), (4, 12)):
        start = time.perf_counter()
        sf = build_sioux_falls_path_set(k_paths=k_paths)
        model = build_activity_stb_model(sf, n_departures=n_dep, demand_scale=0.08)
        build_seconds = time.perf_counter() - start
        start = time.perf_counter()
        behavior = hierarchical_behavior_choice(model)
        forward_seconds = time.perf_counter() - start
        semantic_entries = (
            model.n_groups * len(set(model.columns["activity"]))
            * len(set(model.columns["destination"]))
            * len(set(model.columns["mode"]))
            * n_dep * len(sf.paths)
        )
        # A fairer dense comparison is link-time x behavioral-column.
        dense_link_time_entries = len(sf.links) * n_dep * model.n_columns
        dense_bytes = dense_link_time_entries * 8
        sparse_bytes = model.column_to_path_departure_person.estimated_bytes
        rows.append({
            "k_paths": k_paths,
            "departure_intervals": n_dep,
            "traveler_groups": model.n_groups,
            "behavioral_columns": model.n_columns,
            "path_departure_states": model.path_departure_axis.size,
            "operator_nnz": model.column_to_path_departure_person.matrix.nnz,
            "operator_density": model.column_to_path_departure_person.density,
            "sparse_operator_bytes": sparse_bytes,
            "dense_link_time_tensor_bytes": dense_bytes,
            "dense_to_sparse_byte_ratio": dense_bytes / max(sparse_bytes, 1),
            "build_seconds": build_seconds,
            "behavior_forward_seconds": forward_seconds,
            "mass_residual": behavior.mass_residual,
            "semantic_entries_upper_bound": semantic_entries,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output / "scalability_profile.csv", index=False)
    return df


def run_harness(output_dir: str | Path) -> HarnessSummary:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Case 0/1 use the standard 33-OD, four-path Sioux Falls set.
    sf = build_sioux_falls_path_set(k_paths=4, demand_scale=0.50)
    logit = solve_logit_sue_msa(sf.instance, temperature=1.0, max_iterations=500, tolerance=1e-6)
    ue = solve_exact_fw(sf.instance, max_iterations=500, tolerance=1e-6)
    ftt = build_flow_through_tensors(sf.instance, logit.flow)

    case0 = {
        "zone_od_residual": ftt.mass_residual_zone_od,
        "od_path_residual": ftt.mass_residual_od_path,
        "link_reconstruction_residual": ftt.link_reconstruction_residual,
        "zone_to_od_shape": list(ftt.zone_to_od.shape),
        "od_to_path_shape": list(ftt.od_to_path.shape),
        "path_to_link_shape": list(ftt.path_to_link.shape),
    }
    case1 = {
        "logit_entropy_gap": logit.relative_gap,
        "logit_mass_residual": logit.feasibility_residual,
        "ue_relative_gap": ue.relative_gap,
        "ue_mass_residual": ue.feasibility_residual,
    }

    # Case 2: fixed path flow with three time resolutions over the same one-hour
    # departure window.  This is a numerical consistency test, not an external
    # DTA gold-standard validation.
    queue_rows: list[dict[str, float]] = []
    aggregate_path_flow = logit.flow
    for dt in (10.0, 5.0, 2.5):
        n_dep = int(round(60.0 / dt))
        profile = _gaussian_profile(n_dep)
        path_departures = aggregate_path_flow[:, None] * profile[None, :]
        horizon = int(round(420.0 / dt))
        queue_multiplier = np.ones(len(sf.links), dtype=float)
        queue_labels = list(sf.instance.resource_labels)
        queue_multiplier[queue_labels.index("10->16")] = 0.35
        queue_multiplier[queue_labels.index("16->17")] = 0.40
        queue_multiplier[queue_labels.index("10->17")] = 0.50
        result = run_cohort_point_queue(
            sf,
            path_departures,
            dt_minutes=dt,
            horizon_steps=int(round(600.0 / dt)),
            capacity_multiplier=queue_multiplier,
        )
        queue_rows.append({
            "dt_minutes": dt,
            "departure_intervals": n_dep,
            "max_total_queue": result.max_total_queue,
            "max_link_queue": result.max_link_queue,
            "final_completed_fraction": result.final_completed / max(result.total_demand, 1.0),
            "max_conservation_residual": float(np.max(np.abs(result.conservation_residual))),
        })
    queue_df = pd.DataFrame(queue_rows)
    queue_df.to_csv(output / "case2_time_step_refinement.csv", index=False)
    finest = queue_rows[-1]
    case2 = {
        "time_resolutions": len(queue_rows),
        "finest_dt_minutes": finest["dt_minutes"],
        "finest_max_total_queue": finest["max_total_queue"],
        "finest_conservation_residual": finest["max_conservation_residual"],
    }

    # Case 3: full activity behavior expansion on 28,512 columns.
    sf_full = build_sioux_falls_path_set(k_paths=4)
    activity_full = build_activity_stb_model(sf_full, n_departures=12, demand_scale=0.08)
    behavior_full = hierarchical_behavior_choice(activity_full)
    behavior_full.shares.to_csv(output / "case3_activity_shares.csv", index=False)
    case3 = {
        "traveler_groups": activity_full.n_groups,
        "behavioral_columns": activity_full.n_columns,
        "path_departure_states": activity_full.path_departure_axis.size,
        "behavior_mass_residual": behavior_full.mass_residual,
        "sparse_operator_nnz": activity_full.column_to_path_departure_person.matrix.nnz,
        "sparse_operator_bytes": activity_full.column_to_path_departure_person.estimated_bytes,
    }

    # Case 4: coupled activity-DTA on a medium configuration.  The report uses
    # only fixed-point and conservation diagnostics; no global DUE certificate
    # is claimed.
    sf_dyn = build_sioux_falls_path_set(k_paths=3)
    activity_dyn = build_activity_stb_model(sf_dyn, n_departures=6, demand_scale=0.30)
    base = solve_activity_dta(
        activity_dyn,
        max_iterations=14,
        tolerance=8e-3,
        horizon_steps=130,
        safe_residual_strength=0.25,
    )
    multiplier = np.ones(len(sf_dyn.links), dtype=float)
    labels = list(sf_dyn.instance.resource_labels)
    multiplier[labels.index("10->16")] = 0.25
    multiplier[labels.index("16->17")] = 0.30
    multiplier[labels.index("10->17")] = 0.45
    shock = solve_activity_dta(
        activity_dyn,
        max_iterations=14,
        tolerance=8e-3,
        horizon_steps=130,
        capacity_multiplier=multiplier,
        safe_residual_strength=0.25,
    )
    base.history.to_csv(output / "case4_dynamic_base_history.csv", index=False)
    shock.history.to_csv(output / "case4_dynamic_shock_history.csv", index=False)
    share_change = compare_behavior_shares(base, shock)
    share_change.to_csv(output / "case4_behavior_adaptation.csv", index=False)
    base.queue.history.to_csv(output / "case4_queue_base.csv", index=False)
    shock.queue.history.to_csv(output / "case4_queue_shock.csv", index=False)

    case4 = {
        "base_converged": base.converged,
        "shock_converged": shock.converged,
        "base_fixed_point_residual": base.fixed_point_residual,
        "shock_fixed_point_residual": shock.fixed_point_residual,
        "base_max_queue": base.queue.max_total_queue,
        "shock_max_queue": shock.queue.max_total_queue,
        "base_behavior_mass_residual": base.behavior.mass_residual,
        "shock_behavior_mass_residual": shock.behavior.mass_residual,
        "base_queue_conservation_residual": float(np.max(np.abs(base.queue.conservation_residual))),
        "shock_queue_conservation_residual": float(np.max(np.abs(shock.queue.conservation_residual))),
        "largest_absolute_behavior_share_change": float(share_change["absolute_share_change"].max()),
    }

    # A small multi-start map marks the boundary between a stable numerical
    # experiment and an unsupported uniqueness claim.
    stability, _ = multi_start_stability(
        activity_dyn,
        queue_feedback_strength=1.0,
        safe_residual_strength=0.25,
        max_iterations=10,
        demand_horizon_steps=120,
    )
    stability.to_csv(output / "case4_multistart_stability.csv", index=False)

    scale_df = run_scalability_profile(output)
    scale = {
        "largest_columns": int(scale_df["behavioral_columns"].max()),
        "largest_dense_to_sparse_ratio": float(scale_df["dense_to_sparse_byte_ratio"].max()),
        "largest_forward_seconds": float(scale_df.loc[scale_df["behavioral_columns"].idxmax(), "behavior_forward_seconds"]),
    }

    summary = HarnessSummary(case0, case1, case2, case3, case4, scale)
    (output / "harness_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary
