from __future__ import annotations

"""Deterministic scalable grid instances for column-generation experiments."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from tensormobility.core.network_core import StaticNetwork, build_static_network


@dataclass(frozen=True)
class GridCase:
    network: StaticNetwork
    demand: pd.DataFrame
    rows: int
    columns: int


def build_grid_case(
    rows: int,
    columns: int,
    *,
    n_od: int,
    demand_per_od: float = 500.0,
    base_capacity: float = 1800.0,
    seed: int = 7,
    bidirectional: bool = True,
) -> GridCase:
    if rows < 2 or columns < 2:
        raise ValueError("grid requires at least 2 x 2 nodes")
    if n_od < 1:
        raise ValueError("n_od must be positive")
    rng = np.random.default_rng(seed)

    node_rows: list[dict[str, float | int]] = []
    node_id = lambda r, c: r * columns + c + 1
    for r in range(rows):
        for c in range(columns):
            node_rows.append({"node_id": node_id(r, c), "x_coord": float(c), "y_coord": float(-r), "zone_id": node_id(r, c)})

    link_rows: list[dict[str, float | int | str]] = []
    link_counter = 0
    for r in range(rows):
        for c in range(columns):
            u = node_id(r, c)
            neighbours: list[tuple[int, int, str]] = []
            if c + 1 < columns:
                neighbours.append((u, node_id(r, c + 1), "E"))
            if r + 1 < rows:
                neighbours.append((u, node_id(r + 1, c), "S"))
            if bidirectional:
                if c - 1 >= 0:
                    neighbours.append((u, node_id(r, c - 1), "W"))
                if r - 1 >= 0:
                    neighbours.append((u, node_id(r - 1, c), "N"))
            for from_node, to_node, direction in neighbours:
                link_counter += 1
                # Small deterministic heterogeneity prevents an unrealistically
                # large number of exactly tied paths while preserving grid structure.
                t0 = 1.0 + 0.08 * ((from_node * 17 + to_node * 13) % 5)
                cap_factor = 0.80 + 0.40 * (((from_node + 3 * to_node) % 7) / 6.0)
                link_rows.append({
                    "link_id": f"g{link_counter}",
                    "from_node_id": from_node,
                    "to_node_id": to_node,
                    "free_flow_time": t0,
                    "capacity": base_capacity * cap_factor,
                    "length": 1.0,
                    "direction": direction,
                })

    nodes = pd.DataFrame(node_rows)
    links = pd.DataFrame(link_rows)
    network = build_static_network(f"grid_{rows}x{columns}", nodes, links)

    left = [node_id(r, 0) for r in range(rows)]
    right = [node_id(r, columns - 1) for r in range(rows)]
    top = [node_id(0, c) for c in range(columns)]
    bottom = [node_id(rows - 1, c) for c in range(columns)]
    candidates: list[tuple[int, int]] = []
    for r in range(rows):
        candidates.append((left[r], right[(rows - 1 - r) % rows]))
        candidates.append((right[r], left[(r + rows // 2) % rows]))
    for c in range(columns):
        candidates.append((top[c], bottom[(columns - 1 - c) % columns]))
        candidates.append((bottom[c], top[(c + columns // 2) % columns]))

    # Add reproducible long-distance interior pairs when the requested OD set
    # is larger than the boundary construction.
    all_nodes = nodes["node_id"].to_numpy(int)
    while len(candidates) < n_od * 2:
        o, d = rng.choice(all_nodes, size=2, replace=False)
        o_r, o_c = divmod(int(o) - 1, columns)
        d_r, d_c = divmod(int(d) - 1, columns)
        if abs(o_r - d_r) + abs(o_c - d_c) >= max(rows, columns) // 2:
            candidates.append((int(o), int(d)))

    seen: set[tuple[int, int]] = set()
    selected: list[tuple[int, int]] = []
    for pair in candidates:
        if pair[0] != pair[1] and pair not in seen:
            selected.append(pair)
            seen.add(pair)
        if len(selected) == n_od:
            break
    if len(selected) < n_od:
        raise RuntimeError("could not construct enough unique OD pairs")

    demand_rows = []
    for od_index, (origin, destination) in enumerate(selected):
        variation = 0.75 + 0.50 * ((od_index % 9) / 8.0)
        demand_rows.append({
            "od_id": f"od_{od_index}",
            "origin_node_id": origin,
            "destination_node_id": destination,
            "volume": float(demand_per_od * variation),
        })
    return GridCase(network=network, demand=pd.DataFrame(demand_rows), rows=rows, columns=columns)
