import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.axes import (CANONICAL_AXES, Status, Semiring, promote,
                  slice_dta, slice_multilayer, slice_state_estimation,
                  clean_vi_ok, AxisSpec)
from tensormobility.core.unified_networks import load_case
from tensormobility.dta.column_generation import solve_path_column_generation


def test_axis_registry_contracts():
    assert set(CANONICAL_AXES) == {'od', 'path', 'link', 'group', 'mode',
                                   'layer', 'departure', 'state'}
    # static slice: exactly one synchronized axis (link congestion)
    sync = [k for k, a in CANONICAL_AXES.items()
            if a.status is Status.SYNCHRONIZED]
    assert sync == ['link']
    # promotion chains
    assert slice_dta()['departure'].status is Status.SYNCHRONIZED
    assert slice_multilayer()['layer'].status is Status.SYNCHRONIZED
    assert slice_state_estimation()['state'].status is Status.SYNCHRONIZED
    assert slice_state_estimation()['state'].semiring \
        is Semiring.MAX_PRODUCT
    # SYNCHRONIZED requires pinned_by
    with pytest.raises(ValueError):
        AxisSpec('x', 'bad', Status.SYNCHRONIZED,
                 Semiring.SUM_PRODUCT, 'S_mech')
    # clean-VI admissibility
    axes = slice_multilayer()
    assert clean_vi_ok(axes, {'group', 'mode'})          # values, priors
    assert not clean_vi_ok(axes, {'link'})               # flow-dependent
    assert not clean_vi_ok(axes, {'layer'})              # price-dependent


def test_all_three_networks_load_canonically():
    grid = load_case('grid', rows=6, columns=6, n_od=8)
    sf = load_case('sioux_falls')
    chi = load_case('chicago_sketch', top_od=300)
    for c in (grid, sf, chi):
        assert (c.network.free_flow_time > 0).all()
        assert (c.network.capacity > 0).all()
        assert {'od_id', 'origin_node_id', 'destination_node_id',
                'volume'}.issubset(c.demand.columns)
    assert sf.network.n_nodes == 24 and sf.network.n_links == 76
    assert len(sf.demand) == 33
    assert len(sf.extras['tcglite_paths']) == 126
    assert chi.network.n_nodes == 933 and chi.network.n_links == 2950
    assert len(chi.demand) == 300


def test_v04_solver_runs_on_sf_and_chicago():
    sf = load_case('sioux_falls', demand_scale=8.0)
    r = solve_path_column_generation(sf.network, sf.demand,
                                     max_iterations=120, tolerance=1e-4)
    assert r.relative_gap < 1e-4
    assert r.demand_residual < 1e-6

    chi = load_case('chicago_sketch', top_od=200)
    r2 = solve_path_column_generation(chi.network, chi.demand,
                                      max_iterations=60, tolerance=5e-4)
    assert r2.relative_gap < 5e-4
    assert r2.demand_residual < 1e-6


def test_chicago_full_demand_loads():
    chi = load_case('chicago_sketch')
    assert len(chi.demand) == 93135
    assert abs(chi.demand.volume.sum() - 1137492.6) < 1.0
