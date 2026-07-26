"""Integrated land-use -> activity-tour -> assignment chain on the
paper case: measured opportunities U feed accessibility and location
choice (existing landuse modules), one common column flow generates
every marginal, and the solved state carries conservation and
stationarity certificates.

The stationarity certificate is a KKT equal-gradient spread, not a
Wardrop / Frank--Wolfe gap: the paper preserves the distinction between
certified static gaps and behavior--network fixed-point residuals.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tensormobility.landuse.accessibility import logsum_accessibility
from tensormobility.landuse.location_choice import location_choice
from tensormobility.optimization.mobility_tensor_case import (
    MobilityTensorCase,
    Scenario,
)


@pytest.fixture(scope="module")
def case() -> MobilityTensorCase:
    return MobilityTensorCase()


@pytest.fixture(scope="module")
def base() -> Scenario:
    return Scenario(name="base")


@pytest.fixture(scope="module")
def solved(case: MobilityTensorCase, base: Scenario):
    return case.solve(base)


def _work_travel_costs(case: MobilityTensorCase) -> np.ndarray:
    """Free-flow shortest-path times from each origin to each work
    destination on the case grid."""
    origins = [case.origins[h] for h in sorted(case.origins)]
    works = [case.work_destinations[w] for w in sorted(case.work_destinations)]
    cost = np.zeros((len(origins), len(works)))
    for i, origin in enumerate(origins):
        lengths = nx.single_source_dijkstra_path_length(
            case.graph, origin, weight="t0"
        )
        for j, work in enumerate(works):
            cost[i, j] = lengths[work]
    return cost


def test_accessibility_is_monotone_in_opportunity_supply(case):
    cost = _work_travel_costs(case)
    supply = np.array([1000.0, 800.0])
    richer = np.array([1000.0, 1400.0])
    access_base, cert_base = logsum_accessibility(supply, cost, beta=0.5)
    access_rich, cert_rich = logsum_accessibility(richer, cost, beta=0.5)
    assert cert_base["passed"] and cert_rich["passed"]
    assert np.all(access_rich > access_base)


def test_location_choice_follows_supply_and_respects_capacity(case):
    cost = _work_travel_costs(case)
    demand = np.array([case.demands[h] for h in sorted(case.origins)])

    def shares(w2_supply: float) -> np.ndarray:
        V = np.log(np.array([1000.0, w2_supply]))[None, :] - 0.5 * cost
        P, info = location_choice(
            V, demand=demand, capacity=np.array([1500.0, 1500.0])
        )
        for certificate in info["certificates"]:
            assert certificate["passed"], certificate
        return info["allocation"].sum(axis=0) / demand.sum()

    base_shares = shares(800.0)
    shifted_shares = shares(1400.0)
    assert shifted_shares[1] > base_shares[1]
    assert base_shares.sum() == pytest.approx(1.0, abs=1e-9)


def test_projection_conservation_identities(case, solved):
    """H, W, and S layers of X = Gamma f, the leg-OD totals of R f, and
    the arc totals of Delta f are all exact bookkeeping identities of
    one common column flow."""
    f = solved.x
    total = sum(case.demands.values())
    projections = case.projections(f)
    X = projections["mobility_tensor"].data

    # Every traveler is home once and works once; shopping equals the
    # HWSH tour flow.
    assert float(X[:, :, 0].sum()) == pytest.approx(total, rel=1e-9)
    assert float(X[:, :, 1].sum()) == pytest.approx(total, rel=1e-9)
    shopping_flow = sum(
        float(flow)
        for col, flow in zip(case.columns, f)
        if col.visits_shopping
    )
    assert float(X[:, :, 2].sum()) == pytest.approx(shopping_flow, rel=1e-9)

    # q = R f counts one unit per tour leg: 2 legs for HWH, 3 for HWSH.
    q_total = projections["leg_od_demand"].total()
    legs_total = sum(
        len(col.legs) * float(flow) for col, flow in zip(case.columns, f)
    )
    assert q_total == pytest.approx(legs_total, rel=1e-9)

    # x = Delta f accumulates path length arc by arc.
    x_total = projections["arc_flow"].total()
    length_total = sum(
        col.path_length * float(flow) for col, flow in zip(case.columns, f)
    )
    assert x_total == pytest.approx(length_total, rel=1e-9)

    # The mobility tensor is a typed named-axis object with marginals.
    tensor = projections["mobility_tensor"]
    assert tensor.axes == ("cell_row", "cell_col", "activity")
    activity_marginal = tensor.marginal("cell_row").marginal("cell_col")
    assert activity_marginal.data.shape == (3,)
    assert float(activity_marginal.data.sum()) == pytest.approx(
        tensor.total(), rel=1e-12
    )


def test_solved_state_certificates_pass(case, base, solved):
    for certificate in case.certificates(solved.x, base):
        assert certificate["passed"], certificate


def test_congestion_feedback_reduces_discretionary_tours(case, base, solved):
    """The 40% central-corridor capacity cut must raise peak v/c and
    lower shopping-tour participation -- the behavior--network coupling
    the integrated model exists to represent."""
    bottleneck = Scenario(
        name="central capacity -40%", central_capacity_multiplier=0.60
    )
    bottleneck_result = case.solve(bottleneck, initial_flow=solved.x)
    base_metrics = case.scenario_metrics(base, solved)
    bottleneck_metrics = case.scenario_metrics(bottleneck, bottleneck_result)
    assert bottleneck_metrics["max_vc"] > base_metrics["max_vc"] + 0.1
    assert bottleneck_metrics["shopping_share"] < base_metrics["shopping_share"]
    for certificate in case.certificates(bottleneck_result.x, bottleneck):
        assert certificate["passed"], certificate


def test_opportunity_supply_enters_linear_cost(case, base):
    """a(U) must respond to the measured supply field: richer W2 supply
    lowers the linear cost of every W2 column and no W1 column's cost
    changes sign of response."""
    shifted = Scenario(
        name="employment shifted to W2",
        work_w1_supply=1000.0,
        work_w2_supply=1400.0,
    )
    base_cost = case.linear_column_cost(base)
    shifted_cost = case.linear_column_cost(shifted)
    for col in case.columns:
        change = shifted_cost[col.column_id] - base_cost[col.column_id]
        if col.work_name == "W2":
            assert change < 0.0
        else:
            assert change == pytest.approx(0.0, abs=1e-12)
