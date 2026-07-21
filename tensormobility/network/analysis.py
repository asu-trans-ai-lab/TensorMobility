"""Centrality and communities — networkx-backed by design decision:
these are analysis tools, not differentiable operators, so the mature
library wins over a native rebuild.
"""
from __future__ import annotations

import networkx as nx


def _digraph(n_nodes, from_nodes, to_nodes, weights=None):
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    if weights is None:
        weights = [1.0] * len(from_nodes)
    for f, t, w in zip(from_nodes, to_nodes, weights):
        g.add_edge(int(f), int(t), weight=float(w))
    return g


def betweenness_centrality(n_nodes, from_nodes, to_nodes,
                           weights=None):
    """Weighted betweenness — which nodes carry the shortest paths.
    On a grid with center-cross arterials this ranks the CBD cross
    highest, matching where the assignment loads flow."""
    g = _digraph(n_nodes, from_nodes, to_nodes, weights)
    return nx.betweenness_centrality(g, weight='weight')


def communities(n_nodes, from_nodes, to_nodes, weights=None):
    """Greedy-modularity communities on the undirected projection."""
    g = _digraph(n_nodes, from_nodes, to_nodes,
                 weights).to_undirected()
    return [set(c) for c in
            nx.algorithms.community.greedy_modularity_communities(
                g, weight='weight')]
