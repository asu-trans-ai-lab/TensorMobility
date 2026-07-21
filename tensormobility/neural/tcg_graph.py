"""The transportation computational graph AS a neural network.

The zone -> OD -> path -> link chain is a stack of masked row-softmax
layers acting on conserved mass:

    q = alpha @ softmax_rows(G_logits | M1)      # OD split layer
    f = q @ softmax_rows(R_logits | M2)          # path proportion layer
    v = f @ Delta                                # linear incidence layer
    L = survey + split + count losses            # observation heads

Forward is flow propagation; backward is the SAME graph contracted in
reverse with each operator transposed (the FTT Jacobian identity).
The analytic gradient below is hand-derived backprop through the
masked-softmax stack and is finite-difference-certified in the tests
-- with the recorded lesson that FD checks must sample UNMASKED
coordinates (masked entries have identically zero gradient and zero
difference, making a random check vacuous).
"""
from __future__ import annotations

import numpy as np


def masked_softmax_rows(logits: np.ndarray, mask: np.ndarray
                        ) -> np.ndarray:
    z = np.where(mask > 0, logits, -np.inf)
    zmax = np.nanmax(np.where(mask > 0, z, np.nan), axis=1,
                     keepdims=True)
    e = np.where(mask > 0, np.exp(z - zmax), 0.0)
    s = e.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return e / s


def forward(alpha, G, R, Delta):
    """alpha (zones,), G row-stochastic zones x od, R row-stochastic
    od x path, Delta path x link. Returns (q, f, v)."""
    q = alpha @ G
    f = q @ R
    v = f @ Delta
    return q, f, v


def loss_and_grad(theta, shapes, M1, M2, Delta, survey, split_target,
                  count, observed):
    """Three-term TCG loss (survey / split / count) with analytic
    backprop through the masked-softmax chain.

    theta packs [alpha | G_logits.ravel | R_logits.ravel]."""
    n_z, n_od, n_p = shapes
    a_end = n_z
    g_end = a_end + n_z * n_od
    alpha = theta[:a_end]
    gl = theta[a_end:g_end].reshape(n_z, n_od)
    rl = theta[g_end:].reshape(n_od, n_p)

    G = masked_softmax_rows(gl, M1)
    R = masked_softmax_rows(rl, M2)
    q, f, v = forward(alpha, G, R, Delta)

    n_obs = int(observed.sum())
    n_m1 = int((M1 > 0).sum())
    resid = np.zeros_like(v)
    resid[observed] = v[observed] / count[observed] - 1.0
    L = (np.mean((alpha - survey) ** 2)
         + np.sum(((G - split_target) * M1) ** 2) / n_m1
         + np.sum(resid[observed] ** 2) / n_obs)

    # backward: same chain, transposed operators
    dv = np.zeros_like(v)
    dv[observed] = 2.0 * resid[observed] / (count[observed] * n_obs)
    df = Delta @ dv
    dq = R @ df
    dR = np.outer(q, df)
    dG = np.outer(alpha, dq) + 2.0 * (G - split_target) * M1 / n_m1
    dalpha = G @ dq + 2.0 * (alpha - survey) / n_z

    # softmax rows: dz = s * (ds - sum(ds * s))
    dgl = np.where(M1 > 0,
                   G * (dG - (dG * G).sum(axis=1, keepdims=True)), 0.0)
    drl = np.where(M2 > 0,
                   R * (dR - (dR * R).sum(axis=1, keepdims=True)), 0.0)
    grad = np.concatenate([dalpha, dgl.ravel(), drl.ravel()])
    return float(L), grad


def fd_check(theta, shapes, M1, M2, Delta, survey, split_target, count,
             observed, n_checks=10, seed=0) -> float:
    """FD-certify the analytic gradient on MEANINGFUL coordinates only
    (alpha + unmasked logits). Returns max relative error."""
    n_z, n_od, n_p = shapes
    meaningful = np.concatenate([
        np.arange(n_z),
        n_z + np.flatnonzero(M1.ravel() > 0),
        n_z + n_z * n_od + np.flatnonzero(M2.ravel() > 0)])
    rng = np.random.default_rng(seed)
    idx = rng.choice(meaningful, size=min(n_checks, len(meaningful)),
                     replace=False)
    _, g = loss_and_grad(theta, shapes, M1, M2, Delta, survey,
                         split_target, count, observed)
    worst = 0.0
    for i in idx:
        h = 1e-6 * max(1.0, abs(theta[i]))
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        Lp, _ = loss_and_grad(tp, shapes, M1, M2, Delta, survey,
                              split_target, count, observed)
        Lm, _ = loss_and_grad(tm, shapes, M1, M2, Delta, survey,
                              split_target, count, observed)
        fd = (Lp - Lm) / (2 * h)
        worst = max(worst, abs(fd - g[i]) / max(abs(fd), abs(g[i]),
                                                1e-8))
    return worst
