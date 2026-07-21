from __future__ import annotations

import numpy as np

from tensormobility.behavior.activity import build_activity_stb_model, hierarchical_behavior_choice
from tensormobility.behavior.activity_dta import solve_activity_dta
from tensormobility.dynamics.cohort_queue import run_cohort_point_queue
from tensormobility.dta.sioux_falls import build_sioux_falls_path_set


def test_activity_expansion_mass_and_sparse_contracts() -> None:
    sf = build_sioux_falls_path_set(k_paths=2)
    model = build_activity_stb_model(sf, n_departures=3, demand_scale=0.01)
    result = hierarchical_behavior_choice(model)
    assert model.n_groups == 54
    assert model.n_columns == 3564
    assert result.mass_residual < 1e-9
    assert np.isclose(result.person_column_flow.sum(), model.demands.sum())
    assert model.column_to_path_departure_person.matrix.nnz == model.n_columns
    assert model.column_to_path_departure_person.density < 0.01


def test_cohort_queue_conservation() -> None:
    sf = build_sioux_falls_path_set(k_paths=2)
    model = build_activity_stb_model(sf, n_departures=4, demand_scale=0.03)
    behavior = hierarchical_behavior_choice(model)
    queue = run_cohort_point_queue(sf, behavior.path_departure_vehicle, horizon_steps=90)
    assert np.max(np.abs(queue.conservation_residual)) < 1e-7
    assert queue.final_completed / queue.total_demand > 0.999
    assert queue.path_departure_travel_time.shape == behavior.path_departure_vehicle.shape


def test_small_activity_dta_reports_correct_certificate() -> None:
    sf = build_sioux_falls_path_set(k_paths=2)
    model = build_activity_stb_model(sf, n_departures=4, demand_scale=0.02)
    result = solve_activity_dta(model, max_iterations=4, minimum_iterations=2, horizon_steps=85, tolerance=0.2)
    assert result.behavior.mass_residual < 1e-8
    assert np.max(np.abs(result.queue.conservation_residual)) < 1e-7
    assert result.metadata["certificate"] == "fixed_point_residual_only"
    assert np.isfinite(result.fixed_point_residual)
