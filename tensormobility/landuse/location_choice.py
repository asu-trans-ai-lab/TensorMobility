"""Capacity-aware logit location choice.

    P[n,z] = exp(V[n,z]) / sum_{j in C_n} exp(V[n,j])

with a feasibility mask (zoning / choice set) and optional proportional
rationing against zone capacities. Certificates: each agent's
probabilities sum to 1 over its feasible set; masked zones receive
exactly 0; rationed allocations never exceed capacity.
"""
from __future__ import annotations

import numpy as np


def location_choice(V, feasible=None, demand=None, capacity=None):
    """Logit shares with feasibility mask and optional rationing.

    V: (N, Z) systematic utilities per agent-class x zone
    feasible: optional (N, Z) boolean mask (True = in choice set)
    demand: optional (N,) agents per class -> returns allocation too
    capacity: optional (Z,) max agents per zone (used with demand)

    Returns (P, info). info['certificates'] carries row-sum, mask,
    and capacity checks; info['allocation'] is present when demand is
    given (after proportional rationing if capacity binds; the
    unplaced remainder is reported, never silently dropped).
    """
    V = np.asarray(V, dtype=float)
    if feasible is None:
        feasible = np.ones_like(V, dtype=bool)
    feasible = np.asarray(feasible, dtype=bool)
    if not feasible.any(axis=1).all():
        raise ValueError('some agent class has an empty choice set')
    u = np.where(feasible, V, -np.inf)
    m = u.max(axis=1, keepdims=True)
    e = np.exp(u - m)
    e[~feasible] = 0.0
    P = e / e.sum(axis=1, keepdims=True)

    certs = [{'name': 'row_sums_to_one',
              'residual': float(np.abs(P.sum(axis=1) - 1).max()),
              'passed': bool(np.abs(P.sum(axis=1) - 1).max() < 1e-12)},
             {'name': 'masked_zones_zero',
              'passed': bool((P[~feasible] == 0).all())}]
    info = {'certificates': certs}

    if demand is not None:
        d = np.asarray(demand, dtype=float)
        alloc = P * d[:, None]
        unplaced = np.zeros(len(d))
        if capacity is not None:
            cap = np.asarray(capacity, dtype=float)
            load = alloc.sum(axis=0)
            over = load > cap + 1e-12
            if over.any():
                scale = np.ones_like(load)
                scale[over] = cap[over] / load[over]
                placed = alloc * scale[None, :]
                unplaced = (alloc - placed).sum(axis=1)
                alloc = placed
            certs.append({'name': 'capacity_respected',
                          'passed': bool((alloc.sum(axis=0) <=
                                          cap + 1e-9).all())})
        certs.append({'name': 'agents_accounted',
                      'residual': float(np.abs(
                          alloc.sum(axis=1) + unplaced - d).max()),
                      'passed': bool(np.abs(
                          alloc.sum(axis=1) + unplaced - d).max()
                          < 1e-9)})
        info['allocation'] = alloc
        info['unplaced'] = unplaced
    return P, info
