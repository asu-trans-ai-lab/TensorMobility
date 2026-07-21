from __future__ import annotations

import numpy as np

from tensormobility.harness.analytical_cases import run_parallel_link_verification, run_queue_refinement
from tensormobility.dta.column_generation import solve_path_column_generation
from tensormobility.core.gmns_adapter import read_gmns_bundle, write_gmns_case
from tensormobility.core.grid_network import build_grid_case
from tensormobility.dynamics.path_queue import departure_matrix_from_path_table, run_sparse_path_point_queue
from tensormobility.harness.well_posedness import build_well_posedness_map


def test_parallel_bpr_matches_analytical_equilibrium() -> None:
    result = run_parallel_link_verification()
    assert result.max_flow_error < 1e-4
    assert result.max_cost_error < 1e-6
    assert result.relative_gap < 1e-8


def test_analytical_queue_refinement_improves() -> None:
    result = run_queue_refinement()
    table = result.table.sort_values("dt_minutes", ascending=False)
    assert table.iloc[-1]["clearance_error_minutes"] <= table.iloc[0]["clearance_error_minutes"]
    assert abs(table.iloc[-1]["simulated_peak_queue"] - result.analytical_peak_queue) < 1e-8


def test_grid_column_generation_and_path_decomposition() -> None:
    case = build_grid_case(10, 10, n_od=20, demand_per_od=350.0)
    result = solve_path_column_generation(case.network, case.demand, max_iterations=80, tolerance=4e-4)
    assert result.relative_gap < 5e-4
    assert result.demand_residual < 1e-8
    assert result.generated_paths >= len(case.demand)
    assert np.isclose(result.path_table["path_flow"].sum(), case.demand["volume"].sum())


def test_sparse_grid_queue_conservation_and_capacity_response() -> None:
    case = build_grid_case(8, 8, n_od=12, demand_per_od=450.0, base_capacity=750.0)
    result = solve_path_column_generation(case.network, case.demand, max_iterations=70, tolerance=5e-4)
    profile = np.asarray([0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    paths, departures = departure_matrix_from_path_table(result.path_table, profile)
    base = run_sparse_path_point_queue(case.network, paths, departures, horizon_steps=100)
    shock = run_sparse_path_point_queue(case.network, paths, departures, horizon_steps=130, capacity_multiplier=0.5)
    assert np.max(np.abs(base.conservation_residual)) < 1e-7
    assert np.max(np.abs(shock.conservation_residual)) < 1e-7
    assert shock.max_total_queue >= base.max_total_queue


def test_well_posedness_map_contains_stable_and_multiple_regions() -> None:
    result = build_well_posedness_map(
        lambda_behavior_values=np.linspace(0.0, 1.5, 7),
        lambda_queue_values=np.linspace(0.0, 1.0, 5),
    )
    assert (result.table["classification"] == "locally_contracting").any()
    assert (result.table["root_count"] > 1).any()


def test_gmns_round_trip(tmp_path) -> None:
    case = build_grid_case(6, 6, n_od=8, demand_per_od=100.0)
    write_gmns_case(case.network, case.demand, tmp_path)
    loaded = read_gmns_bundle(tmp_path)
    assert loaded.network.n_nodes == case.network.n_nodes
    assert loaded.network.n_links == case.network.n_links
    assert len(loaded.demand) == len(case.demand)
    assert np.isclose(loaded.demand["volume"].sum(), case.demand["volume"].sum())


def test_bounded_learning_improves_held_out_probability_recovery() -> None:
    from tensormobility.behavior.bounded_learning import train_bounded_behavioral_residual

    case = build_grid_case(8, 8, n_od=12, demand_per_od=400.0, base_capacity=850.0)
    assignment = solve_path_column_generation(case.network, case.demand, max_iterations=70, tolerance=5e-4)
    learned = train_bounded_behavioral_residual(case.network, assignment.path_table, epochs=60)
    test = learned.metrics[learned.metrics["split"] == "test"].iloc[0]
    assert test["learned_probability_mae"] < test["baseline_probability_mae"]
    assert test["max_abs_learned_residual"] <= learned.bound + 1e-6
