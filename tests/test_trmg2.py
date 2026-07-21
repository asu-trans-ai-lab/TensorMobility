"""TRMG2 regional adapter (E4/F1): full-bundle validation report and a
certified assignment slice. Skips when local data absent."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.adapters import trmg2

pytestmark = pytest.mark.skipif(not trmg2.data_available('AM'),
                                reason='TRMG2 data not available')


def test_f1_validation_report():
    case = trmg2.load_trmg2('AM')
    v = case.extras['validation']
    assert v['nodes'] == 33963 and v['links'] == 75939
    assert v['zones'] == 3247
    assert v['od_pairs'] > 1_000_000
    assert abs(v['demand_total'] - 458_797.127) < 1.0
    assert v['self_od_dropped'] == 0
    assert v['unmapped_zone_od_dropped'] == 0


def test_certified_assignment_slice():
    from tensormobility.dta.sparse_assignment import (network_from_case,
                                                      solve_fw)
    case = trmg2.load_trmg2('AM', top_od=500)
    net = network_from_case(case)
    r = solve_fw(net, max_rounds=12, tolerance=1e-4)
    assert r.relative_gap < 1e-3
    assert r.feasibility_residual < 1e-8
