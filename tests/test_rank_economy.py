"""Rank-economy runner integrity: certified reference converges, bias
grows with support richness, adaptive promotion engages on rich
supports (the D3 thesis experiment, small slice)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cases'))
from run_rank_economy import run_one


def test_rank_economy_slice():
    r2 = run_one(2)
    r4 = run_one(4)
    assert r2['certified_gap'] < 1e-6
    assert r4['certified_gap'] < 1e-6
    # richer support helps the certified optimum (weakly)
    assert r4['certified_objective'] <= r2['certified_objective'] + 1e-6
    # thesis slice: bias grows, promotions engage at K=4
    assert r4['static_bias_rel'] > r2['static_bias_rel']
    assert r4['adaptive_promotions'] > 0
