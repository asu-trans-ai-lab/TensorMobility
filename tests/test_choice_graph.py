"""Choice Graph core: chains = behavioral columns; the sequential
(recursive-logit / soft-Bellman) face equals the flat (column) face;
equilibrium closes through the master loop; co-opetition is an
equilibrium cross-response SIGN."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.behavior.choice_graph import (ChoiceGraph,
                                                  park_and_ride_example)
from tensormobility.engines.master_loop import MasterFixedPoint


def test_chains_are_behavioral_columns():
    g = park_and_ride_example()
    chains = g.chains()
    assert len(chains) == 4
    assert ('A', 'B') in chains                       # drive-only
    assert ('A', 'ParkRide', 'BusStation', 'B') in chains


def test_sequential_face_equals_flat_face():
    """Founding identity: soft-Bellman product probabilities == flat
    logit over enumerated chains (arc-additive utilities)."""
    g = park_and_ride_example()
    for theta in (0.05, 0.2, 1.0):
        flat = g.logit_chain_shares(theta)
        seq = g.sequential_chain_shares(theta)
        assert np.allclose(flat, seq, atol=1e-12)
        assert abs(seq.sum() - 1.0) < 1e-12


def test_choice_graph_closes_the_master_loop():
    """B = choice graph, L = congestion on the drive-only chain."""
    g = park_and_ride_example()
    chains = g.chains()
    drive = chains.index(('A', 'B'))
    demand = 2000.0

    def loading(shares):
        """Experienced drive time grows with drive-chain flow (BPR)."""
        v = shares[drive] * demand
        return 38.0 * (1.0 + 0.15 * (v / 800.0) ** 4)

    def cost_to_utils(drive_time):
        u = dict(g.arcs)
        u[('A', 'B')] = -float(drive_time)
        return u

    mfp = MasterFixedPoint(
        behavior=g.behavior_operator(0.2, cost_to_utils),
        loading=loading, engine='auto')
    res = mfp.solve(np.full(len(chains), 0.25), tol=1e-11)
    assert res.converged
    assert mfp.self_consistency(res.x) < 1e-9
    # congestion pushed drive below its uncongested share
    free = g.logit_chain_shares(0.2)
    assert res.x[drive] < free[drive]


def test_coopetition_is_an_equilibrium_sign():
    """Improving Park&Ride transfer COOPERATES with bus (P&R chain up,
    total transit boardings up) while COMPETING with drive-only --
    signs read at equilibrium, not from a fixed-supply derivative."""
    theta, demand = 0.2, 2000.0

    def solve(pr_transfer_util):
        g = park_and_ride_example()
        g.arcs[('ParkRide', 'BusStation')] = pr_transfer_util
        chains = g.chains()
        drive = chains.index(('A', 'B'))
        pr = chains.index(('A', 'ParkRide', 'BusStation', 'B'))

        def loading(shares):
            v = shares[drive] * demand
            return 38.0 * (1.0 + 0.15 * (v / 800.0) ** 4)

        def cost_to_utils(drive_time):
            u = dict(g.arcs)
            u[('A', 'B')] = -float(drive_time)
            return u
        mfp = MasterFixedPoint(g.behavior_operator(theta, cost_to_utils),
                               loading, engine='auto')
        r = mfp.solve(np.full(len(chains), 0.25), tol=1e-11)
        assert r.converged
        return r.x, drive, pr

    x_bad, drive, pr = solve(-15.0)     # slow transfer
    x_good, _, _ = solve(-5.0)          # improved feeder transfer
    assert x_good[pr] > x_bad[pr]           # cooperation with bus
    assert x_good[drive] < x_bad[drive]     # competition with drive
