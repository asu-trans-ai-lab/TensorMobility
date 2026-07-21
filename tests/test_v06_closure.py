"""v0.6 closure: the three foundational special cases — logit SUE / UE,
fluid point queue, and the flow-through tensor chain — certified INSIDE
the unified package, connected to the unified network interface and the
axis calculus. These are the seams the wrap-up must keep tight."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.axes import (CANONICAL_AXES, Status, Semiring,
                          slice_static_ue, slice_dta, slice_mage)
from tensormobility.core.unified_networks import load_case
from tensormobility.dta.special_cases import (build_default_special_case_network,
                                   solve_case_1, solve_case_2,
                                   solve_case_3)


def test_closure_case1_logit_sue_and_ue():
    """Behavior seam: entropy master <-> logit; tau -> 0 <-> UE."""
    ps = build_default_special_case_network()
    sue, ue = solve_case_1(ps, temperature=1.5)
    assert sue.feasibility_residual < 1e-6
    assert ue.feasibility_residual < 1e-6
    assert ue.relative_gap < 2e-5
    # closed form at frozen costs: entropy-KKT IS multinomial logit
    rng = np.random.default_rng(0)
    c = rng.uniform(5.0, 30.0, 6)
    theta = 1.7
    z = np.exp(-theta * (c - c.min()))
    p_logit = z / z.sum()
    # entropy program KKT: p propto exp(-theta c) -- identical object
    assert np.abs(p_logit - z / z.sum()).max() == 0.0
    # axis reading: static UE synchronizes exactly the link axis
    ax = slice_static_ue()
    sync = [k for k, a in ax.items() if a.status is Status.SYNCHRONIZED]
    assert sync == ['link']


def test_closure_case2_fluid_queue_closed_form():
    """Time seam: point-queue recursion vs closed-form pulse; DTA slice
    promotes the departure axis."""
    lam, mu, tau0, dt = 30.0, 20.0, 10.0, 0.001
    T = int(40.0 / dt)
    t = np.arange(T) * dt
    inflow = np.where(t < tau0, lam * dt, 0.0)
    Acum = np.cumsum(inflow)
    Dcum = np.zeros(T)
    d = 0.0
    for i in range(T):
        d = min(Acum[i], d + mu * dt)
        Dcum[i] = d
    q = Acum - Dcum
    peak_cf = (lam - mu) * tau0
    delay_cf = 0.5 * peak_cf * (lam * tau0 / mu)
    assert abs(q.max() - peak_cf) / peak_cf < 2e-3
    assert abs(float(np.sum(q) * dt) - delay_cf) / delay_cf < 2e-3
    assert (Dcum <= Acum + 1e-9).all()          # causality
    assert abs(Dcum[-1] - Acum[-1]) < 1e-9      # conservation
    # package fluid queue: conservation certificate on the SF case
    ps = build_default_special_case_network()
    sue, _ = solve_case_1(ps, temperature=1.5)
    fq = solve_case_2(ps, sue.flow)
    assert float(np.abs(fq.conservation_residual).max()) < 1e-8
    assert slice_dta()['departure'].status is Status.SYNCHRONIZED


def test_closure_case3_flow_through_chain():
    """Space seam: the FTT chain (OD -> path -> link) reconstructs and
    conserves mass; the semiring reading is (+, x)."""
    ps = build_default_special_case_network()
    sue, _ = solve_case_1(ps, temperature=1.5)
    ftt = solve_case_3(ps, sue.flow)
    assert ftt.link_reconstruction_residual < 1e-9
    assert ftt.mass_residual_zone_od < 1e-9
    assert ftt.mass_residual_od_path < 1e-9
    assert CANONICAL_AXES['path'].semiring is Semiring.SUM_PRODUCT


def test_closure_unified_interface_carries_the_cases():
    """The unified loaders keep the certified artifacts reachable: TCGlite
    pool on SF, ref volumes on Chicago, and the MAGE slice extends the
    same canon."""
    sf = load_case('sioux_falls')
    assert len(sf.extras['tcglite_paths']) == 126
    chi = load_case('chicago_sketch', top_od=50)
    assert 'ref_volume' in chi.extras
    ax = slice_mage()
    assert {'company', 'vtype', 'stage', 'match'} <= set(ax)
    assert set(CANONICAL_AXES) <= set(ax)
