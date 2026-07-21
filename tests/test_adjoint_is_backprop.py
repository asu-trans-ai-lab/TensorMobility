"""Correspondence row 2, equality-tested: the hand-derived adjoint of
the TCG chain (transposed forward operators) equals torch autograd on
the same forward computation. Backprop == discrete adjoint, measured
not asserted."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip('torch')

from tensormobility.neural.tcg_graph import (loss_and_grad,
                                             masked_softmax_rows)


def _instance(seed=7):
    rng = np.random.default_rng(seed)
    n_z, n_od, n_p, n_l = 4, 6, 10, 8
    owner_od = np.concatenate([np.arange(n_z),
                               rng.integers(0, n_z, n_od - n_z)])
    M1 = np.zeros((n_z, n_od)); M1[owner_od, np.arange(n_od)] = 1.0
    owner_p = np.concatenate([np.arange(n_od),
                              rng.integers(0, n_od, n_p - n_od)])
    M2 = np.zeros((n_od, n_p)); M2[owner_p, np.arange(n_p)] = 1.0
    Delta = (rng.uniform(0, 1, (n_p, n_l)) > 0.5).astype(float)
    survey = rng.uniform(40, 120, n_z)
    split_t = masked_softmax_rows(rng.normal(0, 1, (n_z, n_od)), M1)
    count = rng.uniform(15, 70, n_l)
    observed = rng.uniform(0, 1, n_l) > 0.3
    theta = np.concatenate([survey + rng.normal(0, 3, n_z),
                            rng.normal(0, .3, n_z * n_od),
                            rng.normal(0, .3, n_od * n_p)])
    return (n_z, n_od, n_p), M1, M2, Delta, survey, split_t, count, \
        observed, theta


def _torch_loss(theta_t, shapes, M1, M2, Delta, survey, split_t, count,
                observed):
    n_z, n_od, n_p = shapes
    alpha = theta_t[:n_z]
    gl = theta_t[n_z:n_z + n_z * n_od].reshape(n_z, n_od)
    rl = theta_t[n_z + n_z * n_od:].reshape(n_od, n_p)

    def msm(logits, mask):
        z = torch.where(mask > 0, logits,
                        torch.tensor(-torch.inf, dtype=logits.dtype))
        z = z - z.max(dim=1, keepdim=True).values.clamp_min(-1e30)
        e = torch.where(mask > 0, torch.exp(z),
                        torch.zeros_like(logits))
        s = e.sum(dim=1, keepdim=True).clamp_min(1e-300)
        return e / s

    G = msm(gl, M1)
    R = msm(rl, M2)
    q = alpha @ G
    f = q @ R
    v = f @ Delta
    n_obs = int(observed.sum())
    n_m1 = int((M1 > 0).sum())
    resid = v[observed] / count[observed] - 1.0
    return (torch.mean((alpha - survey) ** 2)
            + torch.sum(((G - split_t) * M1) ** 2) / n_m1
            + torch.sum(resid ** 2) / n_obs)


def test_hand_adjoint_equals_autograd():
    shapes, M1, M2, Delta, survey, split_t, count, observed, theta = \
        _instance()
    L_np, g_np = loss_and_grad(theta, shapes, M1, M2, Delta, survey,
                               split_t, count, observed)
    tt = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    L_t = _torch_loss(tt, shapes,
                      torch.tensor(M1), torch.tensor(M2),
                      torch.tensor(Delta), torch.tensor(survey),
                      torch.tensor(split_t), torch.tensor(count),
                      torch.tensor(observed))
    L_t.backward()
    g_t = tt.grad.numpy()
    assert abs(float(L_t) - L_np) < 1e-10 * max(1.0, abs(L_np))
    denom = np.maximum(np.abs(g_t), 1e-12)
    # compare on meaningful coordinates (masked logits are zero both ways)
    meaningful = np.abs(g_t) > 1e-14
    rel = np.abs(g_np - g_t)[meaningful] / denom[meaningful]
    assert rel.max() < 1e-9
    # masked coordinates: both identically zero
    assert np.abs(g_np[~meaningful]).max() < 1e-14
