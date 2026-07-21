"""Sensor observation operators on a corridor space-time grid.

Every sensor becomes a sparse linear functional W on the typed (x, t)
grid, so each data-fitting term is just ``W @ field`` -- a tensor
contraction against the observation axis. The pattern follows the
computational-graph traffic-state-estimation codebase of Jiawei Lu
(ASU, 2021, MIT license; the Lu-Li-Wu-Zhou TR-C line), where loop,
probe, and travel-time sensors are compiled offline into weight
matrices over grid collocation points: bilinear interpolation for
point observations, trapezoidal integration for segment aggregates.

Grid convention: the field is flattened row-major over
(len(grid_x), len(grid_t)); operators return scipy.sparse.csr_matrix
of shape (n_obs, nx * nt).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _bracket(values: np.ndarray, grid: np.ndarray):
    """Index i and fraction a such that v = (1-a)*grid[i] + a*grid[i+1].

    Observations outside the grid are clamped to the boundary cell.
    """
    idx = np.searchsorted(grid, values, side='right') - 1
    idx = np.clip(idx, 0, len(grid) - 2)
    span = grid[idx + 1] - grid[idx]
    frac = (values - grid[idx]) / span
    return idx, np.clip(frac, 0.0, 1.0)


def point_operator(obs_x, obs_t, grid_x, grid_t) -> sparse.csr_matrix:
    """Bilinear-interpolation operator for point observations.

    Each row has (at most) four weights over the corners of the grid
    cell containing the observation; rows sum to one, and the operator
    reproduces any field that is bilinear in (x, t) exactly.
    """
    obs_x = np.asarray(obs_x, dtype=float)
    obs_t = np.asarray(obs_t, dtype=float)
    grid_x = np.asarray(grid_x, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)
    n = len(obs_x)
    nt = len(grid_t)
    ix, ax = _bracket(obs_x, grid_x)
    it, at = _bracket(obs_t, grid_t)
    rows = np.repeat(np.arange(n), 4)
    cols = np.stack([ix * nt + it,
                     ix * nt + it + 1,
                     (ix + 1) * nt + it,
                     (ix + 1) * nt + it + 1], axis=1).ravel()
    w = np.stack([(1 - ax) * (1 - at),
                  (1 - ax) * at,
                  ax * (1 - at),
                  ax * at], axis=1).ravel()
    return sparse.csr_matrix((w, (rows, cols)),
                             shape=(n, len(grid_x) * nt))


def segment_integral_operator(x0, x1, obs_t, grid_x,
                              grid_t) -> sparse.csr_matrix:
    """Trapezoidal spatial-integral operator at fixed times.

    Row j approximates the integral of the field over [x0[j], x1[j]]
    at time obs_t[j] (e.g. a vehicle count over a section from a
    density field). Exact for fields linear in x on each grid cell at
    grid times. Implemented as trapezoid weights over the grid nodes
    covered by the segment, composed with bilinear time interpolation.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    x1 = np.atleast_1d(np.asarray(x1, dtype=float))
    obs_t = np.atleast_1d(np.asarray(obs_t, dtype=float))
    grid_x = np.asarray(grid_x, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)
    nx, nt = len(grid_x), len(grid_t)
    it, at = _bracket(obs_t, grid_t)
    rows, cols, vals = [], [], []
    for j in range(len(x0)):
        a, b = sorted((x0[j], x1[j]))
        # nodes of the trapezoid partition: segment ends + interior
        # grid nodes strictly inside (a, b)
        inner = grid_x[(grid_x > a) & (grid_x < b)]
        pts = np.concatenate(([a], inner, [b]))
        # trapezoid weight of each partition node
        seg_w = np.zeros(len(pts))
        d = np.diff(pts)
        seg_w[:-1] += d / 2
        seg_w[1:] += d / 2
        # spread each partition node onto bracketing grid-x nodes
        ixp, axp = _bracket(pts, grid_x)
        for p in range(len(pts)):
            for (gx, wx) in ((ixp[p], (1 - axp[p]) * seg_w[p]),
                             (ixp[p] + 1, axp[p] * seg_w[p])):
                if wx == 0.0:
                    continue
                for (gt, wt) in ((it[j], 1 - at[j]),
                                 (it[j] + 1, at[j])):
                    if wt == 0.0:
                        continue
                    rows.append(j)
                    cols.append(gx * nt + gt)
                    vals.append(wx * wt)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(x0),
                                                          nx * nt))


def apply_operator(W: sparse.csr_matrix, field: np.ndarray) -> np.ndarray:
    """Contract an observation operator with an (nx, nt) field."""
    return W @ np.asarray(field, dtype=float).ravel()
