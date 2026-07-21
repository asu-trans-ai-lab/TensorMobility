from __future__ import annotations

"""One-paper unified STB harness: analytical anchors, scalable grid column
 generation, sparse path queues, well-posedness continuation, and GMNS exchange.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytical_cases import run_parallel_link_verification, run_queue_refinement
from tensormobility.behavior.bounded_learning import train_bounded_behavioral_residual
from tensormobility.dta.column_generation import ColumnGenerationResult, solve_path_column_generation
from tensormobility.core.gmns_adapter import create_external_dnl_exchange, export_unified_assignment, write_gmns_case
from tensormobility.core.grid_network import GridCase, build_grid_case
from tensormobility.core.network_core import build_static_network
from tensormobility.dynamics.path_queue import departure_matrix_from_path_table, run_sparse_path_point_queue
from tensormobility.harness.well_posedness import build_well_posedness_map


@dataclass(frozen=True)
class UnifiedHarnessSummary:
    analytical_parallel: dict[str, float | int]
    analytical_queue: dict[str, float | int]
    grid_scalability: dict[str, float | int]
    bounded_learning: dict[str, float | int]
    dynamic_grid: dict[str, float | int]
    well_posedness: dict[str, float | int]
    gmns_exchange: dict[str, str | int]


def _gaussian_profile(n: int, center: float | None = None, width: float = 1.8) -> np.ndarray:
    x = np.arange(n, dtype=float)
    c = (n - 1) / 2 if center is None else float(center)
    profile = np.exp(-0.5 * ((x - c) / width) ** 2)
    return profile / profile.sum()


def _central_capacity_multiplier(case: GridCase, factor: float = 0.35) -> np.ndarray:
    network = case.network
    node_xy = network.nodes.set_index("node_id")[["x_coord", "y_coord"]]
    center_x = (case.columns - 1) / 2.0
    center_y = -(case.rows - 1) / 2.0
    multiplier = np.ones(network.n_links, dtype=float)
    selected: list[int] = []
    for i, row in network.links.iterrows():
        u = node_xy.loc[int(row["from_node_id"])]
        v = node_xy.loc[int(row["to_node_id"])]
        mid_x = 0.5 * (float(u["x_coord"]) + float(v["x_coord"]))
        mid_y = 0.5 * (float(u["y_coord"]) + float(v["y_coord"]))
        if abs(mid_x - center_x) <= 1.25 and abs(mid_y - center_y) <= 1.25:
            multiplier[int(i)] = factor
            selected.append(int(i))
    if not selected:
        multiplier[network.n_links // 2] = factor
    return multiplier


def _network_with_capacity_multiplier(case: GridCase, multiplier: np.ndarray, name_suffix: str) -> GridCase:
    links = case.network.links.copy()
    links["capacity"] = links["capacity"].to_numpy(float) * multiplier
    network = build_static_network(f"{case.network.name}_{name_suffix}", case.network.nodes, links)
    return GridCase(network=network, demand=case.demand.copy(), rows=case.rows, columns=case.columns)


def _write_markdown(summary: UnifiedHarnessSummary, grid_df: pd.DataFrame, queue_refine: pd.DataFrame, wp_counts: pd.Series, output: Path) -> None:
    a = summary.analytical_parallel
    d = summary.dynamic_grid
    lines = [
        "# STB-FTT Unified Harness v0.4 — Reproduced Results",
        "",
        "## 1. Analytical anchors",
        "",
        f"- Two-parallel-link BPR UE maximum flow error: `{a['max_flow_error']:.3e}` vehicles.",
        f"- Maximum link-cost error: `{a['max_cost_error']:.3e}` minutes.",
        f"- Returned static FW relative gap: `{a['relative_gap']:.3e}`.",
        "- The single-bottleneck queue is compared against its closed-form buildup and clearance curve.",
        "",
        queue_refine.to_markdown(index=False),
        "",
        "## 2. Scalable grid column generation",
        "",
        grid_df.to_markdown(index=False),
        "",
        "The solver never enumerates the complete path set. Each Frank–Wolfe linear oracle is an exact shortest-path pricing call; generated paths are retained to provide path and route flows.",
        "",
        "## 3. Bounded behavioral learning",
        "",
        f"- Test probability MAE without the residual: `{summary.bounded_learning['test_baseline_probability_mae']:.6f}`.",
        f"- Test probability MAE with the bounded learned residual: `{summary.bounded_learning['test_learned_probability_mae']:.6f}`.",
        f"- The residual is bounded by `{summary.bounded_learning['residual_bound']:.2f}` and is flow-independent during the static inner solve.",
        "",
        "## 4. Grid queue and capacity shock",
        "",
        f"- Base peak queue: `{d['base_peak_queue']:.3f}` vehicles.",
        f"- Fixed-path shock peak queue: `{d['fixed_path_shock_peak_queue']:.3f}` vehicles.",
        f"- Reoptimized-path shock peak queue: `{d['reoptimized_shock_peak_queue']:.3f}` vehicles.",
        f"- Base queue conservation residual: `{d['base_conservation_residual']:.3e}`.",
        f"- Shock queue conservation residual: `{d['shock_conservation_residual']:.3e}`.",
        f"- Static shock-assignment relative gap: `{d['shock_assignment_gap']:.3e}`.",
        "",
        "## 5. Well-posedness continuation map",
        "",
        wp_counts.rename_axis("classification").reset_index(name="cells").to_markdown(index=False),
        "",
        "The map is an exact scalar two-route continuation experiment. It identifies root multiplicity and local raw-map derivative; it is not presented as a theorem for a full DTA network.",
        "",
        "## 6. Unified exchange",
        "",
        "The generated GMNS/TAPLite-style result folder contains `od_flow.csv`, `path_flow.csv`, `route_flow.csv`, `link_flow.csv`, `departure_profile.csv`, `path_departure_flow.csv`, and optional `link_time_flow.csv`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_unified_harness(output_folder: str | Path) -> UnifiedHarnessSummary:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Analytical static equilibrium.
    parallel = run_parallel_link_verification()
    parallel_table = pd.DataFrame({
        "link_id": ["parallel_1", "parallel_2"],
        "analytical_flow": parallel.analytical_flow,
        "numerical_flow": parallel.numerical_flow,
        "analytical_cost": parallel.analytical_cost,
        "numerical_cost": parallel.numerical_cost,
    })
    parallel_table.to_csv(output / "analytical_parallel_ue.csv", index=False)
    parallel.solver.history.to_csv(output / "analytical_parallel_solver_history.csv", index=False)

    # 2. Analytical queue refinement.
    queue_refinement = run_queue_refinement()
    queue_refinement.table.to_csv(output / "analytical_queue_refinement.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(queue_refinement.table["dt_minutes"], queue_refinement.table["clearance_error_minutes"], marker="o")
    ax.set_xlabel("Time step (minutes)")
    ax.set_ylabel("Clearance-time error (minutes)")
    ax.set_title("Analytical point-queue time-step refinement")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "analytical_queue_refinement.png", dpi=180)
    plt.close(fig)

    # 3. Static grid scalability.
    configurations = (
        (8, 16, 400.0),
        (12, 32, 425.0),
        (20, 64, 450.0),
        (30, 100, 450.0),
        (40, 150, 450.0),
        (50, 200, 450.0),
    )
    grid_rows: list[dict[str, float | int]] = []
    largest_case: GridCase | None = None
    largest_result: ColumnGenerationResult | None = None
    for size, n_od, demand_per_od in configurations:
        case = build_grid_case(size, size, n_od=n_od, demand_per_od=demand_per_od)
        result = solve_path_column_generation(
            case.network,
            case.demand,
            max_iterations=100,
            tolerance=3e-4 if size >= 30 else 2e-4,
        )
        grid_rows.append({
            "grid": f"{size}x{size}",
            "nodes": case.network.n_nodes,
            "links": case.network.n_links,
            "od_pairs": len(case.demand),
            "iterations": result.iterations,
            "relative_gap": result.relative_gap,
            "generated_paths": result.generated_paths,
            "active_paths": result.active_paths,
            "shortest_path_calls": result.shortest_path_calls,
            "wall_time_seconds": result.wall_time_seconds,
            "demand_residual": result.demand_residual,
        })
        largest_case, largest_result = case, result
    assert largest_case is not None and largest_result is not None
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(output / "grid_scalability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(grid_df["nodes"], grid_df["wall_time_seconds"], marker="o")
    ax.set_xlabel("Grid nodes")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Implicit path column generation scalability")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "grid_scalability_runtime.png", dpi=180)
    plt.close(fig)

    # 4. Bounded AI component: recover a flow-independent behavioral residual.
    learning_case = build_grid_case(10, 10, n_od=20, demand_per_od=450.0, base_capacity=900.0)
    learning_assignment = solve_path_column_generation(
        learning_case.network, learning_case.demand, max_iterations=100, tolerance=3e-4
    )
    learning = train_bounded_behavioral_residual(
        learning_case.network, learning_assignment.path_table, epochs=120
    )
    learning.metrics.to_csv(output / "bounded_learning_metrics.csv", index=False)
    learning.prediction_table.to_csv(output / "bounded_learning_predictions.csv", index=False)
    learning.history.to_csv(output / "bounded_learning_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(learning.history["epoch"], learning.history["cross_entropy"], marker="o", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Grouped cross entropy")
    ax.set_title("Bounded behavioral residual recovery")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "bounded_learning_training.png", dpi=180)
    plt.close(fig)

    # 5. Medium grid static-to-dynamic path-flow experiment.
    dynamic_case = build_grid_case(12, 12, n_od=24, demand_per_od=500.0, base_capacity=900.0)
    base_assignment = solve_path_column_generation(dynamic_case.network, dynamic_case.demand, max_iterations=120, tolerance=2e-4)
    profile = _gaussian_profile(12)
    base_paths, base_departures = departure_matrix_from_path_table(base_assignment.path_table, profile)
    base_queue = run_sparse_path_point_queue(
        dynamic_case.network,
        base_paths,
        base_departures,
        dt_minutes=5.0,
        horizon_steps=150,
    )
    shock_multiplier = _central_capacity_multiplier(dynamic_case, factor=0.35)
    fixed_path_shock = run_sparse_path_point_queue(
        dynamic_case.network,
        base_paths,
        base_departures,
        dt_minutes=5.0,
        horizon_steps=180,
        capacity_multiplier=shock_multiplier,
    )
    shock_case = _network_with_capacity_multiplier(dynamic_case, shock_multiplier, "central_capacity_shock")
    shock_assignment = solve_path_column_generation(shock_case.network, shock_case.demand, max_iterations=150, tolerance=2e-4)
    shock_paths, shock_departures = departure_matrix_from_path_table(shock_assignment.path_table, profile)
    shock_queue = run_sparse_path_point_queue(
        shock_case.network,
        shock_paths,
        shock_departures,
        dt_minutes=5.0,
        horizon_steps=180,
    )
    base_queue.history.to_csv(output / "grid_queue_base.csv", index=False)
    fixed_path_shock.history.to_csv(output / "grid_queue_fixed_path_shock.csv", index=False)
    shock_queue.history.to_csv(output / "grid_queue_reoptimized_shock.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(base_queue.history["minutes"], base_queue.history["total_queue"], label="base")
    ax.plot(fixed_path_shock.history["minutes"], fixed_path_shock.history["total_queue"], label="shock: fixed paths")
    ax.plot(shock_queue.history["minutes"], shock_queue.history["total_queue"], label="shock: reoptimized paths")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Total point queue")
    ax.set_title("Grid path-flow queue response")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "grid_queue_response.png", dpi=180)
    plt.close(fig)

    # 6. Well-posedness continuation.
    wp = build_well_posedness_map()
    wp.table.to_csv(output / "well_posedness_map.csv", index=False)
    class_code = {
        "locally_contracting": 0,
        "sensitive_single_observed": 1,
        "multiple_fixed_points": 2,
        "numerically_unstable_or_start_sensitive": 3,
    }
    pivot = wp.table.assign(code=wp.table["classification"].map(class_code)).pivot(
        index="lambda_queue", columns="lambda_behavior", values="code"
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Learned behavioral feedback strength")
    ax.set_ylabel("Physical queue feedback strength")
    ax.set_xticks(np.arange(0, len(pivot.columns), max(1, len(pivot.columns) // 7)))
    ax.set_xticklabels([f"{pivot.columns[i]:.1f}" for i in ax.get_xticks().astype(int)])
    ax.set_yticks(np.arange(0, len(pivot.index), max(1, len(pivot.index) // 6)))
    ax.set_yticklabels([f"{pivot.index[i]:.1f}" for i in ax.get_yticks().astype(int)])
    ax.set_title("Canonical STB well-posedness map")
    cbar = fig.colorbar(image, ax=ax, ticks=list(class_code.values()))
    cbar.ax.set_yticklabels(["contracting", "sensitive", "multiple", "unstable"])
    fig.tight_layout()
    fig.savefig(output / "well_posedness_map.png", dpi=180)
    plt.close(fig)

    # 7. Unified GMNS/TAPLite and external DNL exchange.
    exchange_root = output / "unified_exchange"
    write_gmns_case(dynamic_case.network, dynamic_case.demand, exchange_root / "gmns_input")
    export_unified_assignment(
        dynamic_case.network,
        dynamic_case.demand,
        base_assignment,
        exchange_root / "taplite_style_results",
        departure_profile=profile,
        queue=base_queue,
    )
    create_external_dnl_exchange(
        dynamic_case.network,
        dynamic_case.demand,
        base_assignment.path_table,
        profile,
        exchange_root / "external_dnl",
    )

    summary = UnifiedHarnessSummary(
        analytical_parallel={
            "max_flow_error": parallel.max_flow_error,
            "max_cost_error": parallel.max_cost_error,
            "relative_gap": parallel.relative_gap,
            "iterations": parallel.solver.iterations,
        },
        analytical_queue={
            "finest_dt_minutes": float(queue_refinement.table["dt_minutes"].min()),
            "finest_clearance_error_minutes": float(
                queue_refinement.table.loc[queue_refinement.table["dt_minutes"].idxmin(), "clearance_error_minutes"]
            ),
            "analytical_peak_queue": queue_refinement.analytical_peak_queue,
        },
        grid_scalability={
            "largest_nodes": int(grid_df["nodes"].max()),
            "largest_links": int(grid_df["links"].max()),
            "largest_od_pairs": int(grid_df["od_pairs"].max()),
            "largest_generated_paths": int(grid_df.loc[grid_df["nodes"].idxmax(), "generated_paths"]),
            "largest_relative_gap": float(grid_df.loc[grid_df["nodes"].idxmax(), "relative_gap"]),
            "largest_wall_time_seconds": float(grid_df.loc[grid_df["nodes"].idxmax(), "wall_time_seconds"]),
        },
        bounded_learning={
            "path_alternatives": len(learning.prediction_table),
            "train_od_groups": len(learning.train_groups),
            "test_od_groups": len(learning.test_groups),
            "test_baseline_probability_mae": float(learning.metrics.loc[learning.metrics["split"] == "test", "baseline_probability_mae"].iloc[0]),
            "test_learned_probability_mae": float(learning.metrics.loc[learning.metrics["split"] == "test", "learned_probability_mae"].iloc[0]),
            "test_baseline_kl": float(learning.metrics.loc[learning.metrics["split"] == "test", "baseline_kl"].iloc[0]),
            "test_learned_kl": float(learning.metrics.loc[learning.metrics["split"] == "test", "learned_kl"].iloc[0]),
            "residual_bound": learning.bound,
        },
        dynamic_grid={
            "base_peak_queue": base_queue.max_total_queue,
            "fixed_path_shock_peak_queue": fixed_path_shock.max_total_queue,
            "reoptimized_shock_peak_queue": shock_queue.max_total_queue,
            "base_conservation_residual": float(np.max(np.abs(base_queue.conservation_residual))),
            "shock_conservation_residual": float(np.max(np.abs(shock_queue.conservation_residual))),
            "shock_assignment_gap": shock_assignment.relative_gap,
            "base_paths": len(base_assignment.path_table),
            "shock_paths": len(shock_assignment.path_table),
            "shocked_links": int(np.count_nonzero(shock_multiplier < 1.0)),
        },
        well_posedness={
            "cells": len(wp.table),
            "multiple_fixed_point_cells": int((wp.table["root_count"] > 1).sum()),
            "locally_contracting_cells": int((wp.table["classification"] == "locally_contracting").sum()),
            "unstable_or_start_sensitive_cells": int(
                (wp.table["classification"] == "numerically_unstable_or_start_sensitive").sum()
            ),
        },
        gmns_exchange={
            "network": dynamic_case.network.name,
            "od_pairs": len(dynamic_case.demand),
            "path_rows": len(base_assignment.path_table),
            "folder": str(exchange_root),
        },
    )
    (output / "unified_harness_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    _write_markdown(summary, grid_df, queue_refinement.table, wp.table["classification"].value_counts(), output)
    return summary
