from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from tensormobility.core.instance import STBInstance

# Canonical Sioux Falls link attributes from the TransportationNetworks TNTP
# benchmark.  Each tuple is (tail, head, capacity, free_flow_time).
SIOUX_FALLS_LINKS: tuple[tuple[int, int, float, float], ...] = (
    (1,2,25900.20064,6),(1,3,23403.47319,4),(2,1,25900.20064,6),(2,6,4958.180928,5),
    (3,1,23403.47319,4),(3,4,17110.52372,4),(3,12,23403.47319,4),(4,3,17110.52372,4),
    (4,5,17782.7941,2),(4,11,4908.82673,6),(5,4,17782.7941,2),(5,6,4947.995469,4),
    (5,9,10000,5),(6,2,4958.180928,5),(6,5,4947.995469,4),(6,8,4898.587646,2),
    (7,8,7841.81131,3),(7,18,23403.47319,2),(8,6,4898.587646,2),(8,7,7841.81131,3),
    (8,9,5050.193156,10),(8,16,5045.822583,5),(9,5,10000,5),(9,8,5050.193156,10),
    (9,10,13915.78842,3),(10,9,13915.78842,3),(10,11,10000,5),(10,15,13512.00155,6),
    (10,16,4854.917717,4),(10,17,4993.510694,8),(11,4,4908.82673,6),(11,10,10000,5),
    (11,12,4908.82673,6),(11,14,4876.508287,4),(12,3,23403.47319,4),(12,11,4908.82673,6),
    (12,13,25900.20064,3),(13,12,25900.20064,3),(13,24,5091.256152,4),(14,11,4876.508287,4),
    (14,15,5127.526119,5),(14,23,4924.790605,4),(15,10,13512.00155,6),(15,14,5127.526119,5),
    (15,19,14564.75315,3),(15,22,9599.180565,3),(16,8,5045.822583,5),(16,10,4854.917717,4),
    (16,17,5229.910063,2),(16,18,19679.89671,3),(17,10,4993.510694,8),(17,16,5229.910063,2),
    (17,19,4823.950831,2),(18,7,23403.47319,2),(18,16,19679.89671,3),(18,20,23403.47319,4),
    (19,15,14564.75315,3),(19,17,4823.950831,2),(19,20,5002.607563,4),(20,18,23403.47319,4),
    (20,19,5002.607563,4),(20,21,5059.91234,6),(20,22,5075.697193,5),(21,20,5059.91234,6),
    (21,22,5229.910063,2),(21,24,4885.357564,3),(22,15,9599.180565,3),(22,20,5075.697193,5),
    (22,21,5229.910063,2),(22,23,5000,4),(23,14,4924.790605,4),(23,22,5000,4),
    (23,24,5078.508436,2),(24,13,5091.256152,4),(24,21,4885.357564,3),(24,23,5078.508436,2),
)

SIOUX_FALLS_NODES: tuple[tuple[int, float, float], ...] = (
    (1,-96.77041974,43.61282792),(2,-96.71125063,43.60581298),(3,-96.77430341,43.5729616),
    (4,-96.74716843,43.56365362),(5,-96.73156909,43.56403357),(6,-96.71164389,43.58758553),
    (7,-96.69342281,43.5638436),(8,-96.71138171,43.56232379),(9,-96.73124137,43.54859634),
    (10,-96.73143801,43.54527088),(11,-96.74684071,43.54413068),(12,-96.78013678,43.54394065),
    (13,-96.79337655,43.49070718),(14,-96.75103549,43.52930613),(15,-96.73150355,43.52940117),
    (16,-96.71138171,43.54674361),(17,-96.71138171,43.54128009),(18,-96.69407825,43.54674361),
    (19,-96.71131617,43.52959125),(20,-96.71118508,43.5153335),(21,-96.73097920,43.51048509),
    (22,-96.73124137,43.51485818),(23,-96.75090441,43.51485818),(24,-96.74920028,43.50316422),
)

# A reproducible 33-OD subset, matching the scale used in the uploaded BTCG
# Sioux Falls guide. Demands are drawn from the canonical TNTP table.
SIOUX_FALLS_33_OD: tuple[tuple[int, int, float], ...] = (
    (1,10,1300),(1,8,800),(1,4,500),(2,10,600),(3,11,300),(4,10,1200),
    (4,11,1400),(4,8,700),(5,10,1000),(5,9,800),(6,16,900),(6,8,800),
    (7,10,1900),(7,16,1400),(7,17,1000),(8,10,1600),(8,16,2200),(8,17,1400),
    (9,10,2800),(9,11,1400),(10,11,4000),(10,15,4000),(10,16,4400),(10,17,3900),
    (11,10,3900),(11,14,1600),(12,10,2000),(13,22,1300),(14,10,2100),(15,22,2600),
    (16,17,2800),(17,19,1700),(22,23,2100),
)


@dataclass(frozen=True)
class SiouxFallsPathSet:
    instance: STBInstance
    paths: tuple[tuple[int, ...], ...]
    link_pairs: tuple[tuple[int, int], ...]
    path_od: np.ndarray
    nodes: pd.DataFrame
    links: pd.DataFrame


def build_sioux_falls_path_set(k_paths: int = 4, demand_scale: float = 1.0) -> SiouxFallsPathSet:
    if k_paths < 1:
        raise ValueError("k_paths must be positive")
    graph = nx.DiGraph()
    link_pairs: list[tuple[int, int]] = []
    capacities: list[float] = []
    free_flow: list[float] = []
    for tail, head, capacity, t0 in SIOUX_FALLS_LINKS:
        graph.add_edge(tail, head, weight=t0)
        link_pairs.append((tail, head))
        capacities.append(capacity)
        free_flow.append(t0)
    link_index = {pair: i for i, pair in enumerate(link_pairs)}

    groups_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    incidence: list[np.ndarray] = []
    paths: list[tuple[int, ...]] = []
    path_od: list[int] = []

    column_index = 0
    for g, (origin, destination, demand) in enumerate(SIOUX_FALLS_33_OD):
        demand = float(demand) * demand_scale
        groups_rows.append({
            "group_index": g,
            "group_id": f"od_{origin}_{destination}",
            "origin": origin,
            "destination": destination,
            "demand": demand,
            "network_n_zones": 24,
        })
        generator = nx.shortest_simple_paths(graph, origin, destination, weight="weight")
        selected: list[tuple[int, ...]] = []
        for _ in range(k_paths):
            try:
                selected.append(tuple(next(generator)))
            except StopIteration:
                break
        if not selected:
            raise RuntimeError(f"No path for OD {(origin, destination)}")
        # If fewer than k distinct paths exist, retain the actual feasible set.
        for rank, node_path in enumerate(selected):
            a = np.zeros(len(link_pairs), dtype=float)
            pairs = tuple(zip(node_path[:-1], node_path[1:]))
            for pair in pairs:
                a[link_index[pair]] = 1.0
            ff_time = float(a @ np.asarray(free_flow))
            column_rows.append({
                "column_index": column_index,
                "column_id": f"od{g}:p{rank}",
                "group_index": g,
                "group_id": f"od_{origin}_{destination}",
                "origin": origin,
                "destination": destination,
                "route_rank": rank,
                "period": "static",
                "mode": "car",
                "route": f"ksp_{rank}",
                "node_sequence": ";".join(map(str, node_path)),
                "path_links": "|".join(f"{u}->{v}" for u, v in pairs),
                "free_flow_path_time": ff_time,
            })
            incidence.append(a)
            paths.append(node_path)
            path_od.append(g)
            column_index += 1

    columns = pd.DataFrame(column_rows)
    groups = pd.DataFrame(groups_rows)
    A = np.column_stack(incidence)
    links = pd.DataFrame(SIOUX_FALLS_LINKS, columns=["from_node", "to_node", "capacity", "free_flow_time"])
    links.insert(0, "link_index", np.arange(len(links), dtype=int))
    nodes = pd.DataFrame(SIOUX_FALLS_NODES, columns=["node_id", "x", "y"])
    instance = STBInstance(
        name="sioux_falls_33od_4path",
        columns=columns,
        groups=groups,
        A=A,
        resource_labels=tuple(f"{u}->{v}" for u, v in link_pairs),
        free_flow_time=np.asarray(free_flow, dtype=float),
        capacity=np.asarray(capacities, dtype=float),
        behavior_cost=np.zeros(len(columns), dtype=float),
        bpr_alpha=0.15,
        bpr_beta=4.0,
    )
    return SiouxFallsPathSet(
        instance=instance,
        paths=tuple(paths),
        link_pairs=tuple(link_pairs),
        path_od=np.asarray(path_od, dtype=int),
        nodes=nodes,
        links=links,
    )
