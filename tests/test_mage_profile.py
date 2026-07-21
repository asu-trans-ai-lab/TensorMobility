import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.axes import slice_mage, Status, CANONICAL_AXES
from tensormobility.core.unified_networks import load_case
from tensormobility.profiles.mage_profile import (solve_mage, default_two_company_config,
                                  multi_start_spread, Company, MAGEConfig)


@pytest.fixture(scope='module')
def grid_case():
    return load_case('grid', rows=8, columns=8, n_od=12,
                     demand_per_od=400.0, base_capacity=900.0)


@pytest.fixture(scope='module')
def base(grid_case):
    return solve_mage(grid_case, default_two_company_config())


def test_mage_axes_extension():
    axes = slice_mage()
    assert set(axes) == set(CANONICAL_AXES) | {'company', 'vtype',
                                               'stage', 'match'}
    assert axes['match'].status is Status.SYNCHRONIZED
    assert axes['company'].status is Status.SPECTATOR
    # canonical registry untouched
    assert 'company' not in CANONICAL_AXES


def test_equilibrium_certificates(base):
    assert base['fixed_point_residual'] < 1e-4
    assert base['assignment_gap'] < 2e-4          # full-space certified
    assert base['conservation_residual'] < 1e-9
    assert base['patience_violation'] < 1e-9
    assert base['rationing_complementarity_violation'] < 1e-6
    assert base['fleet_feasibility_violation'] < 1e-9
    shares = sum(base['mode_share'].values())
    assert abs(shares - 1.0) < 1e-9


def test_av_cap_shapes_equilibrium(grid_case):
    """Tightening a company's AV cap must not increase its AV share,
    and a binding cap pins that option's wait at the patience limit."""
    lo = default_two_company_config()
    lo.companies[1] = Company('beta', fleet=250.0, av_cap=0.05,
                              fare_equiv_min={'AV': 5.0, 'HV': 8.5},
                              op_cost_equiv_min={'AV': 3.5, 'HV': 7.5})
    r_lo = solve_mage(grid_case, lo)
    r_hi = solve_mage(grid_case, default_two_company_config())
    assert r_lo['mode_share']['beta:AV'] \
        <= r_hi['mode_share']['beta:AV'] + 1e-9
    assert r_lo['w_match']['beta:AV'] >= lo.patience_min - 1e-6


def test_fleet_size_reduces_waiting(grid_case):
    small = default_two_company_config()
    for c in small.companies:
        c.fleet *= 0.4
    r_small = solve_mage(grid_case, small)
    r_big = solve_mage(grid_case, default_two_company_config())
    assert max(r_small['w_match'].values()) \
        >= max(r_big['w_match'].values()) - 1e-9
    assert r_small['served_fraction'] <= r_big['served_fraction'] + 1e-9


def test_multi_start_stability(grid_case):
    ms = multi_start_spread(grid_case, default_two_company_config(),
                            seeds=(0, 1))
    # GNE uniqueness is not guaranteed in general; on this smooth logit
    # instance the spread should be tiny -- and it is REPORTED either way
    assert ms['spread'] < 1e-3


def test_interface_generality_sioux_falls():
    case = load_case('sioux_falls', demand_scale=2.0)
    cfg = default_two_company_config()
    r = solve_mage(case, cfg)
    assert r['fixed_point_residual'] < 1e-3
    assert r['conservation_residual'] < 1e-9
    assert r['assignment_gap'] < 2e-4
