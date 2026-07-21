"""Native sparse node-link incidence.

The incidence matrix IS the conservation law: for link flows x and
node net-supply b, feasibility is A x = b. This is the same
node-balance audit the assignment certificates compute, now exposed
as an operator so adjoints can use its transpose (A^T maps node
potentials to link potential differences).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def incidence_matrix(n_nodes, from_nodes, to_nodes):
    """Sparse node-link incidence A (n_nodes x n_links).

    Column j has +1 at from_nodes[j] (flow leaves) and -1 at
    to_nodes[j] (flow arrives), so (A x)[i] = outflow_i - inflow_i.
    """
    f = np.asarray(from_nodes, dtype=int)
    t = np.asarray(to_nodes, dtype=int)
    if f.shape != t.shape:
        raise ValueError('from/to length mismatch')
    m = len(f)
    rows = np.concatenate([f, t])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    return sparse.csr_matrix((vals, (rows, cols)),
                             shape=(n_nodes, m))


def node_balance(A, link_flow, supply):
    """Conservation residual r = A x - b with a certificate."""
    r = A @ np.asarray(link_flow, float) - np.asarray(supply, float)
    res = float(np.abs(r).max())
    return r, {'name': 'node_conservation', 'residual': res,
               'passed': res < 1e-6}
