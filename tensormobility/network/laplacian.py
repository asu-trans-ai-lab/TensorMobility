"""Native sparse weighted graph Laplacian.

L = A W A^T built from the incidence operator (symmetrized weights
for directed links). Facts the tests certify: row sums are zero, L is
positive semidefinite, and the Fiedler value (second-smallest
eigenvalue) is positive iff the graph is connected — the spectral
face of "every OD pair is routable".
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from tensormobility.network.incidence import incidence_matrix


def laplacian(n_nodes, from_nodes, to_nodes, weights=None):
    A = incidence_matrix(n_nodes, from_nodes, to_nodes)
    m = A.shape[1]
    w = np.ones(m) if weights is None else np.asarray(weights, float)
    if (w < 0).any():
        raise ValueError('weights must be nonnegative')
    W = sparse.diags(w)
    return (A @ W @ A.T).tocsr()


def fiedler_value(L):
    """Second-smallest eigenvalue of L (algebraic connectivity)."""
    n = L.shape[0]
    if n <= 2:
        vals = np.linalg.eigvalsh(L.toarray())
        return float(sorted(vals)[1]) if n == 2 else 0.0
    vals = eigsh(L.asfptype(), k=2, sigma=-1e-8,
                 return_eigenvectors=False)
    return float(sorted(vals)[1])
