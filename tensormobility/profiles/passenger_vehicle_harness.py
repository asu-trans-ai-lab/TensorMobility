from __future__ import annotations

"""One-command passenger/vehicle extension harness for the single STB paper."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensormobility.dta.column_generation import solve_path_column_generation
from tensormobility.core.grid_network import GridCase, build_grid_case
from tensormobility.core.network_core import build_static_network
from tensormobility.profiles.passenger_vehicle import (
    PassengerBehaviorParameters,
    build_passenger_vehicle_model,
    compute_mode_interaction_matrix,
    evaluate_passenger_choice,
    export_passenger_vehicle_results,
    solve_analytical_passenger_vehicle_market,
    solve_passenger_vehicle_equilibrium,
    solve_vehicle_service_column_generation,
)


@dataclass(frozen=True)
class PassengerVehicleHarnessSummary:
    analytical_market: dict[str, float | int]
    scalability: dict[str, float | int]
    base_multimodal: dict[str, float | int]
    capacity_shock: dict[str, float | int]
    output_folder: str


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


def _network_with_capacity_multiplier(case: GridCase, multiplier: np.ndarray, suffix: str) -> GridCase:
    links = case.network.links.copy()
    links["capacity"] = links["capacity"].to_numpy(float) * multiplier
    network = build_static_network(f"{case.network.name}_{suffix}", case.network.nodes, links)
    return GridCase(network=network, demand=case.demand.copy(), rows=case.rows, columns=case.columns)


def _balanced_parameters() -> PassengerBehaviorParameters:
    return PassengerBehaviorParameters(
        logit_temperature=5.0,
        auto_monetary_cost=6.0,
        shared_wait_minutes=1.5,
        shared_monetary_cost=0.2,
        bus_wait_minutes=1.5,
        bus_in_vehicle_factor=0.90,
        bus_monetary_cost=0.1,
        park_ride_monetary_cost=0.3,
        parking_penalty=0.5,
        transfer_penalty=0.5,
    )


def _vehicle_options() -> dict[str, float | int]:
    return {
        "max_pair_distance": 12.0,
        "pair_neighbours": 8,
        "pickup_overhead": 0.5,
        "pooled_overhead": 1.0,
        "bus_dispatch_cost": 2.0,
        "max_iterations": 15,
    }


def _mode_share_dict(table: pd.DataFrame) -> dict[str, float]:
    return {str(row.mode): float(row.share) for row in table.itertuples(index=False)}


def _service_metrics(result) -> dict[str, float]:
    request = result.passenger.service_request_table
    shared_passengers = float(request.loc[request["service_mode"] == "shared_ride", "demand"].sum())
    bus_passengers = float(request.loc[request["service_mode"] == "bus", "demand"].sum())
    patterns = result.vehicle.pattern_table
    shared_vehicles = float(patterns.loc[patterns["service_mode"] == "shared_ride", "vehicle_flow"].sum())
    bus_runs = float(patterns.loc[patterns["service_mode"] == "bus", "vehicle_flow"].sum())
    return {
        "shared_passengers": shared_passengers,
        "shared_vehicle_trips": shared_vehicles,
        "realized_shared_occupancy": shared_passengers / max(shared_vehicles, 1e-12),
        "shared_vehicle_savings_vs_one_per_passenger": shared_passengers - shared_vehicles,
        "bus_passengers": bus_passengers,
        "bus_runs": bus_runs,
        "bus_load_factor": bus_passengers / max(40.0 * bus_runs, 1e-12),
    }


def _write_results(
    output: Path,
    summary: PassengerVehicleHarnessSummary,
    analytical,
    scalability: pd.DataFrame,
    base,
    shock,
    mode_interaction,
) -> None:
    base_mode = base.passenger.mode_share[["mode", "share"]].copy()
    shock_mode = shock.passenger.mode_share[["mode", "share"]].copy()
    comparison = base_mode.merge(shock_mode, on="mode", suffixes=("_base", "_shock"))
    comparison["share_change"] = comparison["share_shock"] - comparison["share_base"]
    lines = [
        "# STB-FTT Passenger-Vehicle Multimodal Harness v0.5",
        "",
        "## 1. Analytical passenger-vehicle market",
        "",
        f"- Maximum analytical-vs-iterative passenger-flow error: `{analytical.max_flow_error:.3e}` persons.",
        f"- Service-price error: `{analytical.price_error:.3e}` generalized-cost units.",
        f"- Analytical shared-vehicle requirement: `{analytical.analytical_vehicle_flow:.6f}` vehicles for `{analytical.analytical_shared_flow:.6f}` shared passengers.",
        "",
        "## 2. Passenger/vehicle scalability",
        "",
        scalability.to_markdown(index=False),
        "",
        "The route universe is not enumerated. Road paths come from exact shortest-path Frank-Wolfe pricing, and compatible shared-vehicle route patterns are introduced by a separate set-covering column-generation loop.",
        "",
        "## 3. Base multimodal equilibrium",
        "",
        base.passenger.mode_share.to_markdown(index=False),
        "",
        f"- Passenger mass residual: `{base.passenger.mass_residual:.3e}`.",
        f"- Vehicle service shortfall: `{base.vehicle.demand_residual:.3e}`.",
        f"- Vehicle full candidate-pool pricing gap: `{base.vehicle.relative_pricing_gap:.3e}`.",
        f"- Coupled relative fixed-point residual: `{base.fixed_point_residual:.3e}`.",
        f"- Queue conservation residual: `{np.max(np.abs(base.queue.conservation_residual)):.3e}`.",
        "",
        "## 4. Central road-capacity shock",
        "",
        comparison.to_markdown(index=False),
        "",
        f"- Base peak point queue: `{base.queue.max_total_queue:.3f}` vehicles.",
        f"- Shock peak point queue: `{shock.queue.max_total_queue:.3f}` vehicles.",
        f"- Shock passenger mass residual: `{shock.passenger.mass_residual:.3e}`.",
        f"- Shock vehicle service shortfall: `{shock.vehicle.demand_residual:.3e}`.",
        "",
        "## 5. Connected mode-choice graph",
        "",
        mode_interaction.derivative_matrix.to_markdown(),
        "",
        f"- Cost perturbation used for each row: `{mode_interaction.delta_cost:.3f}` generalized-cost units.",
        f"- Maximum row-sum residual: `{mode_interaction.max_row_sum_residual:.3e}`.",
        "- Negative diagonal terms are own-cost effects; positive off-diagonal terms are direct substitution under fixed network and service supply.",
        "",
        "## 6. Interpretation",
        "",
        "Passenger columns carry person mass and behavioral utility. Vehicle columns carry service trips, seat capacity, occupied movement, and pooled-route structure. The explicit coupling R x <= S y separates passengers from vehicles while allowing service prices, occupancy, fleet requirements, road queues, and cross-mode substitution to interact.",
        "",
        "The current vehicle master is continuous and aggregate. It does not claim integer fleet dispatch, exact individual matching, finite-storage spillback, or a globally unique multimodal DUE.",
    ]
    (output / "RESULTS_PASSENGER_VEHICLE.md").write_text("\n".join(lines), encoding="utf-8")


def run_passenger_vehicle_harness(output_folder: str | Path) -> PassengerVehicleHarnessSummary:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Independent analytical anchor.
    analytical = solve_analytical_passenger_vehicle_market()
    analytical.history.to_csv(output / "analytical_passenger_vehicle_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(analytical.history["iteration"], analytical.history["auto_flow"], label="iterative auto flow")
    ax.axhline(analytical.analytical_auto_flow, linestyle="--", label="analytical auto flow")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Passengers")
    ax.set_title("Analytical two-sided passenger-vehicle anchor")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "analytical_passenger_vehicle_convergence.png", dpi=180)
    plt.close(fig)

    # 2. Scaling of passenger columns + vehicle route-column generation.
    scale_rows: list[dict[str, float | int]] = []
    scale_cases = (
        (6, 6, 6, 3),
        (8, 8, 12, 4),
        (10, 10, 16, 4),
        (12, 12, 20, 4),
        (20, 20, 40, 6),
    )
    for rows, cols, n_od, n_departures in scale_cases:
        start = time.perf_counter()
        case = build_grid_case(rows, cols, n_od=n_od, demand_per_od=220.0, base_capacity=700.0, seed=rows + cols)
        assignment = solve_path_column_generation(case.network, case.demand, max_iterations=90, tolerance=4e-4)
        model = build_passenger_vehicle_model(case, assignment, n_departures=n_departures, max_auto_paths=2)
        passenger = evaluate_passenger_choice(model, case.network.free_flow_time, parameters=_balanced_parameters())
        vehicle = solve_vehicle_service_column_generation(
            case.network,
            passenger.service_request_table,
            case.network.free_flow_time,
            **_vehicle_options(),
        )
        elapsed = time.perf_counter() - start
        scale_rows.append({
            "grid": f"{rows}x{cols}",
            "nodes": case.network.n_nodes,
            "links": case.network.n_links,
            "od_pairs": len(case.demand),
            "generated_road_paths": assignment.generated_paths,
            "passenger_groups": len(model.groups),
            "passenger_columns": len(model.columns),
            "service_requests": len(model.request_template),
            "vehicle_patterns": len(vehicle.pattern_table),
            "positive_vehicle_patterns": int(np.count_nonzero(vehicle.pattern_table["vehicle_flow"].to_numpy(float) > 1e-9)),
            "vehicle_cg_iterations": vehicle.iterations,
            "vehicle_pricing_calls": vehicle.pricing_calls,
            "passenger_mass_residual": passenger.mass_residual,
            "vehicle_service_shortfall": vehicle.demand_residual,
            "wall_time_seconds": elapsed,
        })
    scalability = pd.DataFrame(scale_rows)
    scalability.to_csv(output / "passenger_vehicle_scalability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(scalability["passenger_columns"], scalability["wall_time_seconds"], marker="o")
    ax.set_xlabel("Passenger behavioral columns")
    ax.set_ylabel("End-to-end setup + pricing time (s)")
    ax.set_title("Passenger and vehicle layer scalability")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "passenger_vehicle_scalability.png", dpi=180)
    plt.close(fig)

    # 3. Base multimodal coupled solve.
    base_case = build_grid_case(8, 8, n_od=12, demand_per_od=250.0, base_capacity=650.0, seed=21)
    base_assignment = solve_path_column_generation(base_case.network, base_case.demand, max_iterations=100, tolerance=3e-4)
    base_model = build_passenger_vehicle_model(base_case, base_assignment, n_departures=4, max_auto_paths=2)
    base = solve_passenger_vehicle_equilibrium(
        base_model,
        parameters=_balanced_parameters(),
        max_iterations=50,
        damping=0.15,
        tolerance=5e-3,
        dynamic_feedback=False,
        queue_horizon_steps=160,
        vehicle_solver_options=_vehicle_options(),
        passenger_price_rule="average_cost",
    )
    export_passenger_vehicle_results(base, output / "base_multimodal")

    # Local connected-choice graph: direct mode substitution under frozen supply.
    mode_interaction = compute_mode_interaction_matrix(
        base_model,
        base.link_cost,
        base.vehicle.average_service_price,
        parameters=_balanced_parameters(),
        delta_cost=0.25,
    )
    mode_interaction.long_table.to_csv(output / "mode_interaction_long.csv", index=False)
    mode_interaction.derivative_matrix.to_csv(output / "mode_interaction_matrix.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    image = ax.imshow(mode_interaction.derivative_matrix.to_numpy(float), aspect="auto")
    labels = list(mode_interaction.derivative_matrix.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(list(mode_interaction.derivative_matrix.index))
    ax.set_xlabel("Responding passenger mode")
    ax.set_ylabel("Mode receiving +0.25 cost")
    ax.set_title("Connected mode-choice substitution matrix")
    fig.colorbar(image, ax=ax, label="Share derivative per cost unit")
    fig.tight_layout()
    fig.savefig(output / "connected_mode_interaction.png", dpi=180)
    plt.close(fig)

    # 4. Road capacity shock with the same behavioral and service structure.
    multiplier = _central_capacity_multiplier(base_case, factor=0.35)
    shock_case = _network_with_capacity_multiplier(base_case, multiplier, "central_shock")
    shock_model = build_passenger_vehicle_model(shock_case, base_assignment, n_departures=4, max_auto_paths=2)
    shock = solve_passenger_vehicle_equilibrium(
        shock_model,
        parameters=_balanced_parameters(),
        max_iterations=55,
        damping=0.15,
        tolerance=6e-3,
        dynamic_feedback=False,
        queue_horizon_steps=190,
        vehicle_solver_options=_vehicle_options(),
        passenger_price_rule="average_cost",
    )
    export_passenger_vehicle_results(shock, output / "capacity_shock")

    # Figures shared by the paper.
    mode = base.passenger.mode_share[["mode", "share"]].merge(
        shock.passenger.mode_share[["mode", "share"]], on="mode", suffixes=("_base", "_shock")
    )
    x = np.arange(len(mode))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - width / 2, mode["share_base"], width, label="base")
    ax.bar(x + width / 2, mode["share_shock"], width, label="capacity shock")
    ax.set_xticks(x)
    ax.set_xticklabels(mode["mode"], rotation=15)
    ax.set_ylabel("Passenger share")
    ax.set_title("Passenger behavioral response across multimodal alternatives")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "multimodal_mode_share_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(base.queue.history["minutes"], base.queue.history["total_queue"], label="base")
    ax.plot(shock.queue.history["minutes"], shock.queue.history["total_queue"], label="capacity shock")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Total vehicle queue")
    ax.set_title("Vehicle-generated queue experienced by passengers")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "multimodal_queue_response.png", dpi=180)
    plt.close(fig)

    pattern = base.vehicle.pattern_table.groupby("pattern_type", as_index=False)["vehicle_flow"].sum()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.bar(pattern["pattern_type"], pattern["vehicle_flow"])
    ax.set_ylabel("Vehicle-route flow")
    ax.set_title("Generated vehicle service patterns")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "vehicle_pattern_flow.png", dpi=180)
    plt.close(fig)

    base_metrics = _service_metrics(base)
    shock_metrics = _service_metrics(shock)
    summary = PassengerVehicleHarnessSummary(
        analytical_market={
            "max_flow_error": analytical.max_flow_error,
            "price_error": analytical.price_error,
            "iterations": analytical.iterations,
            "analytical_shared_flow": analytical.analytical_shared_flow,
            "analytical_vehicle_flow": analytical.analytical_vehicle_flow,
        },
        scalability={
            "largest_nodes": int(scalability["nodes"].max()),
            "largest_links": int(scalability["links"].max()),
            "largest_passenger_columns": int(scalability["passenger_columns"].max()),
            "largest_vehicle_patterns": int(scalability.loc[scalability["nodes"].idxmax(), "vehicle_patterns"]),
            "largest_wall_time_seconds": float(scalability.loc[scalability["nodes"].idxmax(), "wall_time_seconds"]),
        },
        base_multimodal={
            "fixed_point_residual": base.fixed_point_residual,
            "passenger_mass_residual": base.passenger.mass_residual,
            "vehicle_service_shortfall": base.vehicle.demand_residual,
            "vehicle_pricing_gap": base.vehicle.relative_pricing_gap,
            "peak_queue": base.queue.max_total_queue,
            "queue_conservation_residual": float(np.max(np.abs(base.queue.conservation_residual))),
            "passenger_columns": len(base.model.columns),
            "vehicle_patterns": len(base.vehicle.pattern_table),
            **base_metrics,
            **{f"mode_share_{k}": v for k, v in _mode_share_dict(base.passenger.mode_share).items()},
        },
        capacity_shock={
            "fixed_point_residual": shock.fixed_point_residual,
            "passenger_mass_residual": shock.passenger.mass_residual,
            "vehicle_service_shortfall": shock.vehicle.demand_residual,
            "vehicle_pricing_gap": shock.vehicle.relative_pricing_gap,
            "peak_queue": shock.queue.max_total_queue,
            "queue_conservation_residual": float(np.max(np.abs(shock.queue.conservation_residual))),
            "shocked_links": int(np.count_nonzero(multiplier < 1.0)),
            **shock_metrics,
            **{f"mode_share_{k}": v for k, v in _mode_share_dict(shock.passenger.mode_share).items()},
        },
        output_folder=str(output),
    )
    (output / "passenger_vehicle_harness_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    _write_results(output, summary, analytical, scalability, base, shock, mode_interaction)
    return summary
