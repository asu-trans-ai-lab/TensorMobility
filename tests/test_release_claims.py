"""Every headline claim gets a command (review P0.7): the Chicago
reference correlation, the MAGE final-state consistency, and the CLI."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.unified_networks import load_case
from tensormobility.dta.sparse_assignment import (network_from_case,
                                                  solve_fw)
from tensormobility.profiles.mage_profile import (solve_mage,
                                                  default_two_company_config)


def test_chicago_reference_correlation_claim():
    """The '1.0000 correlation with ref_volume' claim, as a command:
    full-network certified solve, then corr vs link.csv reference."""
    case = load_case('chicago_sketch')
    net = network_from_case(case)
    r = solve_fw(net, max_rounds=12, tolerance=1e-4)
    ref = case.extras['ref_volume']
    mask = ref > 0
    corr = float(np.corrcoef(r.link_flow[mask], ref[mask])[0, 1])
    assert r.relative_gap < 5e-4
    assert corr > 0.999


def test_mage_final_state_consistency():
    case = load_case('grid', rows=8, columns=8, n_od=12,
                     demand_per_od=400.0, base_capacity=900.0)
    r = solve_mage(case, default_two_company_config())
    fc = r['final_consistency']
    assert fc['r_x'] < 5e-3      # shares closed at the final state
    assert fc['r_T'] < 1e-3      # experienced times closed
    assert fc['r_z'] < 5e-2      # fleet-damping gap, reported honestly


def test_cli_smoke(capsys):
    from tensormobility.cli import main
    assert main(['version']) == 0
    assert main(['demo', 'list']) == 0
    out = capsys.readouterr().out
    assert 'W1' in out and 'pipelines' in out
