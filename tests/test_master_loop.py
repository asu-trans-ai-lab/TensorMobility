"""Panel 1 of the graphical abstract, executable: x* = B_theta(L(x*)),
and the special cases as OPERATOR SWITCHES on one loop."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.engines.master_loop import MasterFixedPoint
from tensormobility.neural import logit_choice


def _two_route_setting():
    """Two parallel routes, BPR loading, demand 1000."""
    t0 = np.array([10.0, 12.0])
    cap = np.array([600.0, 800.0])
    demand = 1000.0

    def loading(shares):
        v = shares * demand
        return t0 * (1.0 + 0.15 * (v / cap) ** 4)
    return loading, t0, cap, demand


def test_logit_sue_is_an_operator_switch():
    loading, *_ = _two_route_setting()
    theta = 0.8
    mfp = MasterFixedPoint(
        behavior=lambda c: logit_choice(c, theta),
        loading=loading, engine='auto')
    res = mfp.solve(np.array([0.5, 0.5]), tol=1e-12)
    assert res.converged
    assert mfp.self_consistency(res.x) < 1e-10
    # at the fixed point, shares ARE the logit of experienced costs
    assert np.allclose(res.x, logit_choice(loading(res.x), theta))


def test_ue_limit_by_switching_the_behavior_operator():
    """Raising theta (colder router) drives the SAME loop toward
    deterministic UE: experienced costs equalize across used routes."""
    loading, *_ = _two_route_setting()
    gaps = []
    for theta in (0.5, 2.0, 8.0):
        mfp = MasterFixedPoint(
            behavior=lambda c, th=theta: logit_choice(c, th),
            loading=loading, engine='auto')
        res = mfp.solve(np.array([0.5, 0.5]), tol=1e-12)
        c = loading(res.x)
        gaps.append(float(c.max() - c.min()) / float(c.min()))
    # Wardrop cost-equalization gap shrinks monotonically with theta
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-2   # colder router -> tighter Wardrop equalization


def test_frozen_behavior_reduces_to_pure_loading():
    """Special case 2's collapse: freeze B => the loop returns the
    loading of the frozen shares in one evaluation (no equilibrium)."""
    loading, *_ = _two_route_setting()
    frozen = np.array([0.3, 0.7])
    mfp = MasterFixedPoint(behavior=lambda c: frozen, loading=loading,
                           engine='picard')
    res = mfp.solve(np.array([0.5, 0.5]))
    assert res.converged
    assert np.allclose(res.x, frozen)
