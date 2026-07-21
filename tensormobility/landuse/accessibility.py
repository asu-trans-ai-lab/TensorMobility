"""Logsum accessibility — the quantity that closes the land-use loop.

    A[z] = (1/beta) * ln sum_j exp(beta * (ln E[j] - c[z,j]))

This is the same kernel as the explainer's land-use card (gravity ==
logit destination choice) read as a welfare measure: accessibility is
the expected maximum utility of the destination choice, and it is what
travel conditions feed BACK to location choice (S -> accessibility ->
L in the flow-through chain).
"""
from __future__ import annotations

import numpy as np


def logsum_accessibility(employment, cost, beta=1.0):
    """Accessibility per origin zone.

    employment: (Z,) opportunities per zone (must be > 0 somewhere)
    cost: (Z, Z) travel cost matrix c[origin, destination]
    beta: cost sensitivity (> 0)

    Returns (A, certificate) with A shape (Z,). Monotonicity is part
    of the contract: adding opportunities or lowering any cost can
    never reduce any zone's accessibility.
    """
    E = np.asarray(employment, dtype=float)
    c = np.asarray(cost, dtype=float)
    if beta <= 0:
        raise ValueError('beta must be positive')
    if (E < 0).any():
        raise ValueError('employment must be nonnegative')
    with np.errstate(divide='ignore'):
        v = np.where(E > 0, np.log(np.maximum(E, 1e-300)), -np.inf)
    util = beta * (v[None, :] - c)              # (Z origins, Z dests)
    m = util.max(axis=1, keepdims=True)
    A = (m[:, 0] + np.log(np.exp(util - m).sum(axis=1))) / beta
    cert = {'name': 'accessibility_logsum',
            'finite': bool(np.isfinite(A).all()),
            'passed': bool(np.isfinite(A).all()),
            'min': float(A.min()), 'max': float(A.max())}
    return A, cert
