"""The same sparse full-space-certified solver on all three canonical
networks through one interface."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.unified_networks import load_case
from tensormobility.dta.sparse_assignment import (network_from_case, solve_fw,
                                       solve_fw_gp, solve_atom_gp)


@pytest.mark.parametrize('case_kw', [
    dict(name='grid', rows=8, columns=8, n_od=12, demand_per_od=450.0,
         base_capacity=900.0),
    dict(name='sioux_falls', demand_scale=8.0),
    dict(name='chicago_sketch', top_od=300),
])
def test_fw_full_space_certificate_all_networks(case_kw):
    kw = dict(case_kw)
    net = network_from_case(load_case(kw.pop('name'), **kw))
    r = solve_fw(net, max_rounds=25, tolerance=1e-4)
    assert r.relative_gap < 1e-4          # TRUE full-space gap
    assert r.feasibility_residual < 1e-8


def test_fw_gp_and_atoms_agree_on_sf():
    net = network_from_case(load_case('sioux_falls', demand_scale=8.0))
    r1 = solve_fw(net, max_rounds=30, tolerance=1e-5)
    r2 = solve_fw_gp(net, max_rounds=30, tolerance=1e-5)
    assert r2.relative_gap < 1e-4
    assert abs(r1.objective - r2.objective) / r1.objective < 1e-3
    ra = solve_atom_gp(net, adaptive=True, warm_rounds=4, max_cycles=15,
                       gp_sweeps=10, tolerance=1e-4)
    rs = solve_atom_gp(net, adaptive=False, warm_rounds=4, max_cycles=15,
                       gp_sweeps=10, tolerance=1e-4)
    assert ra.feasibility_residual < 1e-8
    # promotion may only help under equal budgets
    assert ra.relative_gap <= rs.relative_gap * 1.05 + 1e-12


def test_chicago_full_scale_one_round():
    """Full 93,135-OD Chicago loads and completes a certified pricing
    round through the unified interface (seed AON gap ~0.84)."""
    net = network_from_case(load_case('chicago_sketch'))
    r = solve_fw(net, max_rounds=2, tolerance=1e-4)
    assert r.n_columns >= 93135
    assert np.isfinite(r.relative_gap)
    assert r.feasibility_residual < 1e-6
