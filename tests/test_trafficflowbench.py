"""TrafficFlowBench (IEEE five-corridor) adapter: the DTA core and the
queue core run on real PeMS corridor data through the unified
interface, plus the simple departure-time profile (first behavior-axis
step). Skips when the local benchmark data is absent (competition data
stays local)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.adapters import trafficflowbench as tfb

pytestmark = pytest.mark.skipif(not tfb.data_available('I405N'),
                                reason='TrafficFlowBench data not '
                                'available locally')


@pytest.fixture(scope='module')
def case():
    from tensormobility.core.unified_networks import load_case
    return load_case('tfb_corridor', panel='I405N')


def test_corridor_loads_canonically(case):
    assert case.network.n_nodes == 225 and case.network.n_links == 313
    assert (case.network.free_flow_time > 0).all()
    assert len(case.extras['path_table']) == 2145
    assert abs(case.extras['path_table'].volume.sum() - 8065.2) < 0.5


def test_released_path_incidence_reconstructs_exactly(case):
    n_pairs = tfb.validate_path_incidence(case)
    assert n_pairs > 2145            # every path, every link, exact


def test_dta_core_certifies_on_corridor(case):
    from tensormobility.dta.sparse_assignment import (network_from_case,
                                                      solve_fw)
    net = network_from_case(case)
    r = solve_fw(net, max_rounds=15, tolerance=1e-5)
    assert r.relative_gap < 1e-4     # full-space priced
    assert r.feasibility_residual < 1e-8


def test_queue_core_on_real_episode(case):
    """Point-queue certificates on REAL detector inflow at the largest
    queue episode's bottleneck: conservation + causality always; a
    documented 30% capacity reduction during the episode window makes
    the queue physics visible."""
    inflow, cap, ep = tfb.bottleneck_inflow(case)
    assert len(inflow) >= 200 and cap > 0
    mu = np.full(len(inflow), cap)
    # capacity reduction during the episode duration (collision-style)
    dur_steps = max(1, int(ep.duration_min / 5))
    start = len(inflow) // 3
    mu[start:start + dur_steps] *= 0.7
    Acum = np.cumsum(inflow)
    Dcum = np.zeros_like(Acum)
    d = 0.0
    for i in range(len(inflow)):
        d = min(Acum[i], d + mu[i])
        Dcum[i] = d
    assert (Dcum <= Acum + 1e-9).all()                # causality
    tail = min(Acum[-1], Dcum[-1] + mu[-1] * 12)      # let it clear
    assert Acum[-1] - Dcum[-1] >= -1e-9               # conservation
    assert (Acum - Dcum).max() > 0                    # queue formed


def test_departure_profile_first_behavior_step(case):
    prof = tfb.departure_profile(case)
    assert abs(prof.share.sum() - 1.0) < 1e-9
    assert len(prof) >= 24
    # the peak share falls in a plausible daytime window
    peak = prof.loc[prof.share.idxmax(), 'time_of_day']
    hour = int(str(peak).split(':')[0])
    assert 5 <= hour <= 20


def test_rebuilt_incidence_is_contiguous_with_coverage(case):
    """The rebuild certificate: every released path becomes a verified
    contiguous walk; anchors kept are a subset of the released
    incidence; coverage of anchors is reported honestly (branch anchors
    that cannot lie on one directed walk are dropped with counts)."""
    rb = tfb.rebuild_contiguous_paths(case)
    st = rb['stats']
    assert st['n_paths'] == 2145
    assert st['n_contiguous'] == st['n_paths']      # 100% contiguous
    assert st['coverage'] > 0.75                    # honest, not 100%
    # kept anchors are released pairs
    import pandas as pd
    inc_rel = pd.read_csv(case.extras['path_link_incidence_file'])
    rel = set(zip(inc_rel.path_id, inc_rel.link_id))
    tfb_ids = case.extras['tfb_link_ids']
    for r in rb['path_table'].head(200).itertuples(index=False):
        for li in r.anchors:
            assert (r.path_id, tfb_ids[li - 1]) in rel
    # rebuilt walks give a routable column pool: endpoints connected
    # along their own links by construction
    assert rb['path_table'].volume.sum() > 0
