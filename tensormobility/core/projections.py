from __future__ import annotations

import numpy as np


def project_simplex(v: np.ndarray, mass: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum x = mass}."""

    v = np.asarray(v, dtype=float)
    if mass < 0:
        raise ValueError("mass must be nonnegative")
    if v.size == 0:
        if mass == 0:
            return v.copy()
        raise ValueError("cannot project positive mass onto an empty simplex")
    if mass == 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - mass
    rho_candidates = np.flatnonzero(u - cssv / (np.arange(v.size) + 1) > 0)
    if rho_candidates.size == 0:
        return np.full_like(v, mass / v.size)
    rho = int(rho_candidates[-1])
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    correction = mass - float(w.sum())
    if abs(correction) > 1e-10:
        w[np.argmax(w)] += correction
    return w


def feasible_residual(f: np.ndarray, group_columns: list[np.ndarray], demands: np.ndarray) -> float:
    residual = max(0.0, -float(np.min(f))) if len(f) else 0.0
    for demand, idx in zip(demands, group_columns, strict=True):
        residual = max(residual, abs(float(np.sum(f[idx])) - float(demand)))
    return residual
