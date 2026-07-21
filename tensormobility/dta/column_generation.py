from __future__ import annotations

"""Scalable path column generation using network Frank--Wolfe.

The linear minimization oracle is an exact shortest-path calculation for each
OD group.  Complete path pools are never enumerated.  Because every all-or-
nothing direction is stored as an ordered path, the link-based Frank--Wolfe
iterate also retains a path-flow decomposition suitable for GMNS export and
subsequent path-resolved dynamic loading.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from tensormobility.core.network_core import StaticNetwork


@dataclass(frozen=True)
class ColumnGenerationResult:
    network_name: str
    link_flow: np.ndarray
    link_cost: np.ndarray
    path_table: pd.DataFrame
    history: pd.DataFrame
    objective: float
    relative_gap: float
    absolute_gap: float
    generated_paths: int
    active_paths: int
    shortest_path_calls: int
    iterations: int
    wall_time_seconds: float
    demand_residual: float
    metadata: dict[str, float | int | str]


def bpr_cost(
    flow: np.ndarray,
    free_flow_time: np.ndarray,
    capacity: np.ndarray,
    *,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> np.ndarray:
    v = np.asarray(flow, dtype=float)
    t0 = np.asarray(free_flow_time, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    ratio = np.maximum(v, 0.0) / cap
    return t0 * (1.0 + alpha * ratio**beta)


def beckmann_objective(
    flow: np.ndarray,
    free_flow_time: np.ndarray,
    capacity: np.ndarray,
    *,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float:
    v = np.maximum(np.asarray(flow, dtype=float), 0.0)
    t0 = np.asarray(free_flow_time, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    value = t0 * (v + alpha * v ** (beta + 1.0) / ((beta + 1.0) * cap**beta))
    return float(np.sum(value))


def _path_cost(path: tuple[int, ...], link_cost: np.ndarray) -> float:
    return float(np.sum(link_cost[np.asarray(path, dtype=int)])) if path else 0.0


def _aggregate_link_flow(path_flows: list[dict[tuple[int, ...], float]], n_links: int) -> np.ndarray:
    link_flow = np.zeros(n_links, dtype=float)
    for group_paths in path_flows:
        for path, flow in group_paths.items():
            if flow <= 0:
                continue
            link_flow[np.asarray(path, dtype=int)] += float(flow)
    return link_flow


def _line_search(
    current_link_flow: np.ndarray,
    direction_link_flow: np.ndarray,
    network: StaticNetwork,
    *,
    alpha: float,
    beta: float,
) -> float:
    if np.max(np.abs(direction_link_flow)) < 1e-14:
        return 0.0
    result = minimize_scalar(
        lambda gamma: beckmann_objective(
            current_link_flow + float(gamma) * direction_link_flow,
            network.free_flow_time,
            network.capacity,
            alpha=alpha,
            beta=beta,
        ),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 80},
    )
    if not result.success:
        return 2.0 / 3.0
    return float(np.clip(result.x, 0.0, 1.0))


def solve_path_column_generation(
    network: StaticNetwork,
    demand: pd.DataFrame,
    *,
    max_iterations: int = 250,
    tolerance: float = 1e-5,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    prune_relative_flow: float = 1e-11,
    minimum_iterations: int = 2,
) -> ColumnGenerationResult:
    required = {"od_id", "origin_node_id", "destination_node_id", "volume"}
    if not required.issubset(demand.columns):
        raise ValueError(f"demand missing fields {required - set(demand.columns)}")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid solver controls")
    od = demand.reset_index(drop=True).copy()
    volumes = od["volume"].to_numpy(float)
    if np.any(volumes < 0) or not np.all(np.isfinite(volumes)):
        raise ValueError("demand volume must be finite and nonnegative")

    path_flows: list[dict[tuple[int, ...], float]] = []
    generated: list[set[tuple[int, ...]]] = []
    od_pairs = [(int(r.origin_node_id), int(r.destination_node_id)) for r in od.itertuples(index=False)]
    initial_paths, _ = network.batch_shortest_paths(od_pairs, network.free_flow_time)
    shortest_path_calls = len(od_pairs)
    for path, volume in zip(initial_paths, volumes, strict=True):
        path_flows.append({path: float(volume)})
        generated.append({path})

    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    absolute_gap = np.inf
    rel_gap = np.inf

    for iteration in range(1, max_iterations + 1):
        current_link_flow = _aggregate_link_flow(path_flows, network.n_links)
        costs = bpr_cost(current_link_flow, network.free_flow_time, network.capacity, alpha=bpr_alpha, beta=bpr_beta)
        aon_link_flow = np.zeros(network.n_links, dtype=float)
        shortest_paths: list[tuple[int, ...]] = []
        total_system_cost = 0.0
        shortest_path_total = 0.0

        shortest_paths, shortest_costs = network.batch_shortest_paths(od_pairs, costs)
        shortest_path_calls += len(od_pairs)
        for g, (shortest_path, shortest_cost, volume) in enumerate(zip(shortest_paths, shortest_costs, volumes, strict=True)):
            generated[g].add(shortest_path)
            if volume > 0:
                aon_link_flow[np.asarray(shortest_path, dtype=int)] += float(volume)
            shortest_path_total += float(volume) * float(shortest_cost)
            for path, flow in path_flows[g].items():
                total_system_cost += float(flow) * _path_cost(path, costs)

        absolute_gap = max(total_system_cost - shortest_path_total, 0.0)
        rel_gap = absolute_gap / max(total_system_cost, 1.0)
        objective = beckmann_objective(
            current_link_flow, network.free_flow_time, network.capacity, alpha=bpr_alpha, beta=bpr_beta
        )
        active_count = sum(sum(flow > 1e-9 for flow in group.values()) for group in path_flows)
        rows.append({
            "iteration": iteration,
            "beckmann_objective": objective,
            "total_system_travel_time": total_system_cost,
            "shortest_path_total": shortest_path_total,
            "absolute_gap": absolute_gap,
            "relative_gap": rel_gap,
            "active_paths": active_count,
            "generated_paths": sum(len(v) for v in generated),
        })
        if iteration >= minimum_iterations and rel_gap <= tolerance:
            break

        gamma = _line_search(
            current_link_flow,
            aon_link_flow - current_link_flow,
            network,
            alpha=bpr_alpha,
            beta=bpr_beta,
        )
        total_demand = max(float(volumes.sum()), 1.0)
        threshold = prune_relative_flow * total_demand
        for g, (shortest_path, volume) in enumerate(zip(shortest_paths, volumes, strict=True)):
            updated: dict[tuple[int, ...], float] = {}
            for path, flow in path_flows[g].items():
                new_flow = (1.0 - gamma) * float(flow)
                if new_flow > threshold:
                    updated[path] = new_flow
            updated[shortest_path] = updated.get(shortest_path, 0.0) + gamma * float(volume)
            path_flows[g] = updated

    elapsed = time.perf_counter() - start
    final_link_flow = _aggregate_link_flow(path_flows, network.n_links)
    final_cost = bpr_cost(final_link_flow, network.free_flow_time, network.capacity, alpha=bpr_alpha, beta=bpr_beta)
    final_objective = beckmann_objective(
        final_link_flow, network.free_flow_time, network.capacity, alpha=bpr_alpha, beta=bpr_beta
    )

    path_rows: list[dict[str, object]] = []
    demand_residual = 0.0
    link_ids = network.links["link_id"].astype(str).to_numpy()
    for g, row in enumerate(od.itertuples(index=False)):
        group_total = float(sum(path_flows[g].values()))
        demand_residual = max(demand_residual, abs(group_total - float(row.volume)))
        ordered = sorted(path_flows[g].items(), key=lambda item: (-item[1], item[0]))
        for rank, (path, flow) in enumerate(ordered):
            nodes = network.node_sequence(path)
            path_rows.append({
                "path_id": f"{row.od_id}:p{rank}",
                "route_id": f"{row.od_id}:r{rank}",
                "od_id": str(row.od_id),
                "origin_node_id": int(row.origin_node_id),
                "destination_node_id": int(row.destination_node_id),
                "path_flow": float(flow),
                "path_cost": _path_cost(path, final_cost),
                "free_flow_path_time": _path_cost(path, network.free_flow_time),
                "node_sequence": ";".join(map(str, nodes)),
                "link_sequence": ";".join(str(link_ids[i]) for i in path),
                "link_index_sequence": ";".join(map(str, path)),
                "n_links": len(path),
            })
    path_table = pd.DataFrame(path_rows)

    # Recompute the final gap from the returned path decomposition.
    total_system_cost = float(final_link_flow @ final_cost)
    shortest_path_total = 0.0
    _, final_shortest_costs = network.batch_shortest_paths(od_pairs, final_cost)
    shortest_path_calls += len(od_pairs)
    for row, shortest_cost in zip(od.itertuples(index=False), final_shortest_costs, strict=True):
        shortest_path_total += float(row.volume) * float(shortest_cost)
    absolute_gap = max(total_system_cost - shortest_path_total, 0.0)
    rel_gap = absolute_gap / max(total_system_cost, 1.0)

    return ColumnGenerationResult(
        network_name=network.name,
        link_flow=final_link_flow,
        link_cost=final_cost,
        path_table=path_table,
        history=pd.DataFrame(rows),
        objective=final_objective,
        relative_gap=rel_gap,
        absolute_gap=absolute_gap,
        generated_paths=sum(len(v) for v in generated),
        active_paths=int(np.count_nonzero(path_table["path_flow"].to_numpy(float) > 1e-9)) if not path_table.empty else 0,
        shortest_path_calls=shortest_path_calls,
        iterations=len(rows),
        wall_time_seconds=float(elapsed),
        demand_residual=float(demand_residual),
        metadata={
            "algorithm": "network_frank_wolfe_with_implicit_shortest_path_column_generation",
            "bpr_alpha": float(bpr_alpha),
            "bpr_beta": float(bpr_beta),
            "certificate": "static_beckmann_fw_relative_gap",
        },
    )
