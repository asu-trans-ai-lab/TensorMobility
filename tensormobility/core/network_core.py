from __future__ import annotations

"""Generic directed-network primitives used by the scalable STB harness.

The module deliberately avoids a framework-specific graph tensor.  Links have
stable integer positions, while public identifiers remain in DataFrames.  A
path is represented by an ordered tuple of link positions; this is sufficient
for static incidence, path flow export, and path-resolved queue loading.
"""

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class StaticNetwork:
    name: str
    nodes: pd.DataFrame
    links: pd.DataFrame
    adjacency: tuple[tuple[int, ...], ...]
    node_id_to_pos: dict[int, int]
    node_pos_to_id: tuple[int, ...]

    def __post_init__(self) -> None:
        required_node = {"node_id"}
        required_link = {"link_id", "from_node_id", "to_node_id", "free_flow_time", "capacity"}
        if not required_node.issubset(self.nodes.columns):
            raise ValueError(f"nodes missing fields {required_node - set(self.nodes.columns)}")
        if not required_link.issubset(self.links.columns):
            raise ValueError(f"links missing fields {required_link - set(self.links.columns)}")
        if len(self.adjacency) != len(self.nodes):
            raise ValueError("adjacency length does not equal node count")
        if np.any(self.links["capacity"].to_numpy(float) <= 0):
            raise ValueError("link capacity must be positive")
        if np.any(self.links["free_flow_time"].to_numpy(float) <= 0):
            raise ValueError("free-flow time must be positive")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def free_flow_time(self) -> np.ndarray:
        return self.links["free_flow_time"].to_numpy(float)

    @property
    def capacity(self) -> np.ndarray:
        return self.links["capacity"].to_numpy(float)

    @property
    def from_pos(self) -> np.ndarray:
        return np.asarray([self.node_id_to_pos[int(v)] for v in self.links["from_node_id"]], dtype=int)

    @property
    def to_pos(self) -> np.ndarray:
        return np.asarray([self.node_id_to_pos[int(v)] for v in self.links["to_node_id"]], dtype=int)

    def shortest_path(self, origin_node_id: int, destination_node_id: int, link_cost: np.ndarray) -> tuple[tuple[int, ...], float]:
        costs = np.asarray(link_cost, dtype=float)
        if costs.shape != (self.n_links,):
            raise ValueError("link_cost has wrong shape")
        if np.any(~np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("link_cost must be finite and nonnegative")
        try:
            source = self.node_id_to_pos[int(origin_node_id)]
            target = self.node_id_to_pos[int(destination_node_id)]
        except KeyError as exc:
            raise ValueError(f"unknown OD node {exc.args[0]}") from exc
        if source == target:
            return tuple(), 0.0

        dist = np.full(self.n_nodes, np.inf, dtype=float)
        predecessor_link = np.full(self.n_nodes, -1, dtype=int)
        dist[source] = 0.0
        heap: list[tuple[float, int]] = [(0.0, source)]
        to_pos = self.to_pos
        while heap:
            current_distance, u = heapq.heappop(heap)
            if current_distance != dist[u]:
                continue
            if u == target:
                break
            for link_idx in self.adjacency[u]:
                v = int(to_pos[link_idx])
                candidate = current_distance + float(costs[link_idx])
                if candidate + 1e-14 < dist[v]:
                    dist[v] = candidate
                    predecessor_link[v] = link_idx
                    heapq.heappush(heap, (candidate, v))
        if not np.isfinite(dist[target]):
            raise RuntimeError(f"no path from {origin_node_id} to {destination_node_id}")

        path_rev: list[int] = []
        node = target
        from_pos = self.from_pos
        while node != source:
            link_idx = int(predecessor_link[node])
            if link_idx < 0:
                raise RuntimeError("shortest-path predecessor chain is incomplete")
            path_rev.append(link_idx)
            node = int(from_pos[link_idx])
        return tuple(reversed(path_rev)), float(dist[target])


    def batch_shortest_paths(
        self,
        od_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        link_cost: np.ndarray,
    ) -> tuple[list[tuple[int, ...]], np.ndarray]:
        """Compute shortest paths for many ODs, reusing one tree per origin.

        Parallel links are reduced to the minimum-cost edge for the current
        call; the selected physical link index is retained for reconstruction.
        """
        costs = np.asarray(link_cost, dtype=float)
        if costs.shape != (self.n_links,):
            raise ValueError("link_cost has wrong shape")
        if np.any(~np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("link_cost must be finite and nonnegative")
        pairs = [(int(o), int(d)) for o, d in od_pairs]
        if not pairs:
            return [], np.zeros(0, dtype=float)
        pos_pairs = []
        for o, d in pairs:
            if o not in self.node_id_to_pos or d not in self.node_id_to_pos:
                raise ValueError(f"unknown OD node {(o, d)}")
            pos_pairs.append((self.node_id_to_pos[o], self.node_id_to_pos[d]))

        from_pos = self.from_pos
        to_pos = self.to_pos
        best: dict[tuple[int, int], tuple[float, int]] = {}
        for link_idx, (u, v, c) in enumerate(zip(from_pos, to_pos, costs, strict=True)):
            key = (int(u), int(v))
            old = best.get(key)
            if old is None or float(c) < old[0]:
                best[key] = (float(c), int(link_idx))
        rows = np.fromiter((k[0] for k in best), dtype=int)
        cols = np.fromiter((k[1] for k in best), dtype=int)
        data = np.fromiter((v[0] for v in best.values()), dtype=float)
        graph = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        unique_sources = sorted(set(o for o, _ in pos_pairs))
        source_index = {source: i for i, source in enumerate(unique_sources)}
        distances, predecessors = dijkstra(
            graph, directed=True, indices=np.asarray(unique_sources, dtype=int), return_predecessors=True
        )
        if distances.ndim == 1:
            distances = distances[None, :]
            predecessors = predecessors[None, :]
        paths: list[tuple[int, ...]] = []
        path_costs = np.zeros(len(pos_pairs), dtype=float)
        for i, (source, target) in enumerate(pos_pairs):
            tree = source_index[source]
            distance = float(distances[tree, target])
            if not np.isfinite(distance):
                raise RuntimeError(f"no path from {pairs[i][0]} to {pairs[i][1]}")
            path_rev: list[int] = []
            node = target
            while node != source:
                predecessor = int(predecessors[tree, node])
                if predecessor < 0:
                    raise RuntimeError("shortest-path predecessor chain is incomplete")
                selected = best.get((predecessor, node))
                if selected is None:
                    raise RuntimeError("could not map predecessor arc to physical link")
                path_rev.append(int(selected[1]))
                node = predecessor
            paths.append(tuple(reversed(path_rev)))
            path_costs[i] = distance
        return paths, path_costs

    def node_sequence(self, path_links: Iterable[int]) -> tuple[int, ...]:
        links = tuple(int(i) for i in path_links)
        if not links:
            return tuple()
        from_ids = self.links["from_node_id"].to_numpy(int)
        to_ids = self.links["to_node_id"].to_numpy(int)
        nodes = [int(from_ids[links[0]])]
        for link_idx in links:
            if nodes[-1] != int(from_ids[link_idx]):
                raise ValueError("path link sequence is not contiguous")
            nodes.append(int(to_ids[link_idx]))
        return tuple(nodes)

    def path_incidence(self, paths: Iterable[tuple[int, ...]]) -> sparse.csr_matrix:
        path_list = list(paths)
        rows: list[int] = []
        cols: list[int] = []
        for p, seq in enumerate(path_list):
            for link_idx in seq:
                rows.append(int(link_idx))
                cols.append(p)
        data = np.ones(len(rows), dtype=float)
        return sparse.coo_matrix((data, (rows, cols)), shape=(self.n_links, len(path_list))).tocsr()


def build_static_network(name: str, nodes: pd.DataFrame, links: pd.DataFrame) -> StaticNetwork:
    node_table = nodes.copy().reset_index(drop=True)
    link_table = links.copy().reset_index(drop=True)
    node_table["node_id"] = node_table["node_id"].astype(int)
    link_table["link_id"] = link_table["link_id"].astype(str)
    link_table["from_node_id"] = link_table["from_node_id"].astype(int)
    link_table["to_node_id"] = link_table["to_node_id"].astype(int)
    link_table["free_flow_time"] = link_table["free_flow_time"].astype(float)
    link_table["capacity"] = link_table["capacity"].astype(float)

    node_ids = tuple(int(v) for v in node_table["node_id"])
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node_id")
    node_id_to_pos = {node_id: i for i, node_id in enumerate(node_ids)}
    adjacency_lists: list[list[int]] = [[] for _ in node_ids]
    for i, row in link_table.iterrows():
        u = int(row["from_node_id"])
        v = int(row["to_node_id"])
        if u not in node_id_to_pos or v not in node_id_to_pos:
            raise ValueError(f"link {row['link_id']} references unknown node")
        adjacency_lists[node_id_to_pos[u]].append(int(i))
    return StaticNetwork(
        name=name,
        nodes=node_table,
        links=link_table,
        adjacency=tuple(tuple(v) for v in adjacency_lists),
        node_id_to_pos=node_id_to_pos,
        node_pos_to_id=node_ids,
    )
