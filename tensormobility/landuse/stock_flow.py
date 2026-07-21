"""Stock-flow transitions on the land-use clock (annual).

    L[z,k,y+1] = L[z,k,y] + D[z,k,y] - R[z,k,y]
    H[z,g,y+1] = H[z,g,y] + M_in - M_out + B - E

Both return the new state plus a machine-checkable certificate:
conservation is exact by construction (the residual is computed, not
assumed), and nonnegativity is enforced as a hard failure rather than
a silent clip.
"""
from __future__ import annotations

import numpy as np


def step_stock(L, development, removal):
    """Advance floor-space stocks one year.

    Returns (L_next, certificate). Raises ValueError if the update
    would drive any stock negative — removing more than exists is a
    modeling error, not a state.
    """
    L = np.asarray(L, dtype=float)
    D = np.asarray(development, dtype=float)
    R = np.asarray(removal, dtype=float)
    L_next = L + D - R
    if (L_next < -1e-12).any():
        bad = int((L_next < -1e-12).sum())
        raise ValueError(
            f'stock update drives {bad} cell(s) negative: removal '
            f'exceeds existing stock')
    residual = float(np.abs((L_next - L) - (D - R)).max())
    cert = {'name': 'stock_conservation',
            'residual': residual,
            'passed': residual < 1e-9,
            'total_before': float(L.sum()),
            'total_after': float(L_next.sum()),
            'net_flow': float((D - R).sum())}
    return L_next, cert


def step_household_cohorts(H, move_in, move_out, births, exits):
    """Advance household cohorts one year (same contract as stocks)."""
    H = np.asarray(H, dtype=float)
    flows = (np.asarray(move_in, float) - np.asarray(move_out, float)
             + np.asarray(births, float) - np.asarray(exits, float))
    H_next = H + flows
    if (H_next < -1e-12).any():
        raise ValueError('cohort update drives households negative')
    residual = float(np.abs((H_next - H) - flows).max())
    cert = {'name': 'cohort_conservation',
            'residual': residual,
            'passed': residual < 1e-9,
            'total_before': float(H.sum()),
            'total_after': float(H_next.sum())}
    return H_next, cert
