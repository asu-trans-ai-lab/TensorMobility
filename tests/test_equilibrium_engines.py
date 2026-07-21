"""The engine escalation ladder on (a) a synthetic stiff map that
reproduces the MAGE 'stuck at 0.12' cycle, and (b) the MAGE profile
itself with different inner engines."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.engines.equilibrium_engines import (picard, anderson, newton_block,
                                         solve_fixed_point, detect_cycle)
from tensormobility.core.unified_networks import load_case
from tensormobility.profiles.mage_profile import solve_mage, default_two_company_config


def stiff_wait_map(w, patience=12.0, w0=60.0, z=1500.0,
                   demand=4800.0, theta=0.15, base_u=-20.0):
    """Scalar analogue of the MAGE choice<->wait subsystem: share via
    logit against a fixed outside option, wait via smoothed M/M/1.
    dw/dshare ~ 1e3-1e4 near saturation -> damped Picard limit-cycles,
    exactly the observed 'stuck at 0.12' pathology."""
    u = -(15.0 + w)
    s = 1.0 / (1.0 + np.exp(-theta * (base_u - u)))
    tot = s * demand
    slack = np.maximum(z - tot, 0.0)
    return w0 / (slack + w0 / patience)


def test_picard_cycles_and_is_detected():
    g = lambda w: np.atleast_1d(stiff_wait_map(w[0]))
    r = picard(g, np.array([6.0]), damp=0.3, max_iter=400, tol=1e-10)
    assert not r.converged            # honest negative: E0 fails here
    # either an explicit cycle was detected or the residual stalled
    assert r.cycle is not None or r.residual > 1e-6


def test_newton_closes_the_stiff_map():
    g = lambda w: np.atleast_1d(stiff_wait_map(w[0]))
    r = newton_block(g, np.array([6.0]), bounds=(0.0, 12.0))
    assert r.converged and r.residual < 1e-9


def test_auto_escalation_records_and_converges():
    g = lambda w: np.atleast_1d(stiff_wait_map(w[0]))
    r = solve_fixed_point(g, np.array([6.0]), engine='auto',
                          tol=1e-9, bounds=(0.0, 12.0))
    assert r.converged
    assert len(r.escalations) >= 1    # at least one promotion happened
    assert r.escalations[0]['engine'] == 'picard'


def test_anderson_on_smooth_contraction():
    A = np.array([[0.5, 0.2], [0.1, 0.6]])
    b = np.array([1.0, -0.5])
    g = lambda x: A @ x + b
    r = anderson(g, np.zeros(2), tol=1e-12)
    assert r.converged
    x_star = np.linalg.solve(np.eye(2) - A, b)
    assert np.abs(r.x - x_star).max() < 1e-8


def test_mage_engines_agree(  ):
    case = load_case('grid', rows=8, columns=8, n_od=12,
                     demand_per_od=400.0, base_capacity=900.0)
    cfg_n = default_two_company_config()
    r_newton = solve_mage(case, cfg_n)
    cfg_a = default_two_company_config()
    cfg_a.inner_engine = 'auto'
    r_auto = solve_mage(case, cfg_a)
    # both close the equilibrium and agree on mode shares
    assert r_auto['fixed_point_residual'] < 1e-4
    for k, v in r_newton['mode_share'].items():
        assert abs(r_auto['mode_share'][k] - v) < 5e-3
    # engine bookkeeping is part of the certificate
    assert r_newton['inner_engine'] == 'newton'
    assert r_auto['inner_engine'] in ('picard', 'anderson', 'newton')
