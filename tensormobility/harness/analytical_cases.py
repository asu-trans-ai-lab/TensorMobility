from __future__ import annotations

"""Analytical anchor cases for solver and temporal-discretization validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tensormobility.dta.column_generation import ColumnGenerationResult, bpr_cost, solve_path_column_generation
from tensormobility.core.network_core import StaticNetwork, build_static_network


@dataclass(frozen=True)
class ParallelLinkAnalyticalResult:
    analytical_flow: np.ndarray
    numerical_flow: np.ndarray
    analytical_cost: np.ndarray
    numerical_cost: np.ndarray
    max_flow_error: float
    max_cost_error: float
    relative_gap: float
    solver: ColumnGenerationResult


@dataclass(frozen=True)
class QueueRefinementResult:
    table: pd.DataFrame
    pulse_arrival_rate: float
    service_rate: float
    pulse_duration_minutes: float
    analytical_peak_queue: float
    analytical_clearance_minutes: float


def build_parallel_bpr_case(
    *,
    demand: float = 4000.0,
    free_flow_time: tuple[float, float] = (8.0, 10.0),
    capacity: tuple[float, float] = (2200.0, 3400.0),
) -> tuple[StaticNetwork, pd.DataFrame]:
    nodes = pd.DataFrame({"node_id": [1, 2], "x_coord": [0.0, 1.0], "y_coord": [0.0, 0.0]})
    links = pd.DataFrame({
        "link_id": ["parallel_1", "parallel_2"],
        "from_node_id": [1, 1],
        "to_node_id": [2, 2],
        "free_flow_time": list(free_flow_time),
        "capacity": list(capacity),
        "length": [1.0, 1.0],
    })
    network = build_static_network("parallel_bpr", nodes, links)
    od = pd.DataFrame({
        "od_id": ["od_parallel"],
        "origin_node_id": [1],
        "destination_node_id": [2],
        "volume": [float(demand)],
    })
    return network, od


def analytical_parallel_bpr_equilibrium(
    demand: float,
    free_flow_time: np.ndarray,
    capacity: np.ndarray,
    *,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> np.ndarray:
    t0 = np.asarray(free_flow_time, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    if t0.shape != (2,) or cap.shape != (2,):
        raise ValueError("parallel analytical case requires two links")

    def difference(x1: float) -> float:
        x = np.asarray([x1, demand - x1], dtype=float)
        c = bpr_cost(x, t0, cap, alpha=alpha, beta=beta)
        return float(c[0] - c[1])

    left = difference(0.0)
    right = difference(float(demand))
    if left >= 0.0:
        return np.asarray([0.0, float(demand)])
    if right <= 0.0:
        return np.asarray([float(demand), 0.0])
    x1 = float(brentq(difference, 0.0, float(demand), xtol=1e-12, rtol=1e-12))
    return np.asarray([x1, float(demand) - x1])


def run_parallel_link_verification(
    *,
    demand: float = 4000.0,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> ParallelLinkAnalyticalResult:
    network, od = build_parallel_bpr_case(demand=demand)
    analytical_flow = analytical_parallel_bpr_equilibrium(demand, network.free_flow_time, network.capacity)
    solver = solve_path_column_generation(
        network,
        od,
        max_iterations=max_iterations,
        tolerance=tolerance,
        minimum_iterations=2,
        prune_relative_flow=1e-14,
    )
    analytical_cost = bpr_cost(analytical_flow, network.free_flow_time, network.capacity)
    return ParallelLinkAnalyticalResult(
        analytical_flow=analytical_flow,
        numerical_flow=solver.link_flow.copy(),
        analytical_cost=analytical_cost,
        numerical_cost=solver.link_cost.copy(),
        max_flow_error=float(np.max(np.abs(analytical_flow - solver.link_flow))),
        max_cost_error=float(np.max(np.abs(analytical_cost - solver.link_cost))),
        relative_gap=float(solver.relative_gap),
        solver=solver,
    )


def analytical_point_queue(
    times_minutes: np.ndarray,
    *,
    arrival_rate_per_minute: float,
    service_rate_per_minute: float,
    pulse_duration_minutes: float,
) -> np.ndarray:
    t = np.asarray(times_minutes, dtype=float)
    if arrival_rate_per_minute <= service_rate_per_minute:
        return np.zeros_like(t)
    buildup = arrival_rate_per_minute - service_rate_per_minute
    peak = buildup * pulse_duration_minutes
    queue = np.where(
        t <= pulse_duration_minutes,
        buildup * t,
        np.maximum(peak - service_rate_per_minute * (t - pulse_duration_minutes), 0.0),
    )
    return np.maximum(queue, 0.0)


def simulate_single_bottleneck_queue(
    *,
    dt_minutes: float,
    arrival_rate_per_minute: float,
    service_rate_per_minute: float,
    pulse_duration_minutes: float,
    horizon_minutes: float,
) -> pd.DataFrame:
    n_steps = int(np.ceil(horizon_minutes / dt_minutes))
    q = 0.0
    rows: list[dict[str, float]] = [{"time_minutes": 0.0, "queue": 0.0}]
    for step in range(n_steps):
        start = step * dt_minutes
        end = min((step + 1) * dt_minutes, horizon_minutes)
        width = end - start
        overlap = max(min(end, pulse_duration_minutes) - min(start, pulse_duration_minutes), 0.0)
        arrivals = arrival_rate_per_minute * overlap
        service = service_rate_per_minute * width
        q = max(q + arrivals - service, 0.0)
        rows.append({"time_minutes": end, "queue": q})
    return pd.DataFrame(rows)


def run_queue_refinement(
    *,
    dt_values: tuple[float, ...] = (10.0, 5.0, 2.5, 1.0, 0.5),
    arrival_rate_per_minute: float = 30.0,
    service_rate_per_minute: float = 18.0,
    pulse_duration_minutes: float = 40.0,
) -> QueueRefinementResult:
    peak = (arrival_rate_per_minute - service_rate_per_minute) * pulse_duration_minutes
    clearance = pulse_duration_minutes + peak / service_rate_per_minute
    horizon = clearance + 20.0
    rows: list[dict[str, float]] = []
    for dt in dt_values:
        sim = simulate_single_bottleneck_queue(
            dt_minutes=dt,
            arrival_rate_per_minute=arrival_rate_per_minute,
            service_rate_per_minute=service_rate_per_minute,
            pulse_duration_minutes=pulse_duration_minutes,
            horizon_minutes=horizon,
        )
        analytical = analytical_point_queue(
            sim["time_minutes"].to_numpy(float),
            arrival_rate_per_minute=arrival_rate_per_minute,
            service_rate_per_minute=service_rate_per_minute,
            pulse_duration_minutes=pulse_duration_minutes,
        )
        error = np.abs(sim["queue"].to_numpy(float) - analytical)
        simulated_clear = sim.loc[sim["time_minutes"] >= pulse_duration_minutes]
        cleared = simulated_clear[simulated_clear["queue"] <= 1e-10]
        clearance_est = float(cleared["time_minutes"].iloc[0]) if not cleared.empty else np.nan
        rows.append({
            "dt_minutes": float(dt),
            "max_queue_error": float(error.max(initial=0.0)),
            "l1_queue_error": float(np.sum(error) * dt),
            "simulated_peak_queue": float(sim["queue"].max()),
            "analytical_peak_queue": float(peak),
            "simulated_clearance_minutes": clearance_est,
            "analytical_clearance_minutes": float(clearance),
            "clearance_error_minutes": abs(clearance_est - clearance) if np.isfinite(clearance_est) else np.nan,
        })
    return QueueRefinementResult(
        table=pd.DataFrame(rows),
        pulse_arrival_rate=float(arrival_rate_per_minute),
        service_rate=float(service_rate_per_minute),
        pulse_duration_minutes=float(pulse_duration_minutes),
        analytical_peak_queue=float(peak),
        analytical_clearance_minutes=float(clearance),
    )
