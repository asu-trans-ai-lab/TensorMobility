"""Network science as an executable layer (design review §2.2).

Division of labor per the author's decision (2026-07-22): incidence
and Laplacian operators are NATIVE sparse constructions — the adjoint
path needs them differentiable and matrix-free — while centrality and
community detection wrap networkx, which is already a dependency.
"""
from tensormobility.network.incidence import (incidence_matrix,
                                              node_balance)
from tensormobility.network.laplacian import (fiedler_value,
                                              laplacian)
from tensormobility.network.analysis import (betweenness_centrality,
                                             communities)

__all__ = ['incidence_matrix', 'node_balance', 'laplacian',
           'fiedler_value', 'betweenness_centrality', 'communities']
