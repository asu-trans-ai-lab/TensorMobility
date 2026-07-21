from __future__ import annotations

import numpy as np

from tensormobility.dta.column_generation import solve_path_column_generation
from tensormobility.core.grid_network import build_grid_case
from tensormobility.profiles.passenger_vehicle import (
    PassengerBehaviorParameters,
    build_passenger_vehicle_coupling_operators,
    build_passenger_vehicle_model,
    evaluate_passenger_choice,
    solve_analytical_passenger_vehicle_market,
    solve_passenger_vehicle_equilibrium,
    solve_vehicle_service_column_generation,
)


def _parameters() -> PassengerBehaviorParameters:
    return PassengerBehaviorParameters(
        logit_temperature=5.0,
        shared_wait_minutes=1.5,
        shared_monetary_cost=0.2,
        bus_wait_minutes=1.5,
        bus_in_vehicle_factor=0.9,
        bus_monetary_cost=0.1,
        park_ride_monetary_cost=0.3,
        parking_penalty=0.5,
        transfer_penalty=0.5,
        auto_monetary_cost=6.0,
    )


def _vehicle_options() -> dict[str, float | int]:
    return {
        "max_pair_distance": 12.0,
        "pair_neighbours": 8,
        "pickup_overhead": 0.5,
        "pooled_overhead": 1.0,
        "bus_dispatch_cost": 2.0,
        "max_iterations": 15,
    }


def _small_model():
    case = build_grid_case(5, 5, n_od=5, demand_per_od=120.0, base_capacity=600.0, seed=31)
    assignment = solve_path_column_generation(case.network, case.demand, max_iterations=60, tolerance=7e-4)
    model = build_passenger_vehicle_model(case, assignment, n_departures=3, max_auto_paths=2)
    return case, model


def test_analytical_two_sided_anchor() -> None:
    result = solve_analytical_passenger_vehicle_market()
    assert result.max_flow_error < 1e-7
    assert result.price_error < 1e-8
    assert abs(result.analytical_shared_flow - 2.0 * result.analytical_vehicle_flow) < 1e-10


def test_passenger_mass_vehicle_service_and_typed_coupling() -> None:
    case, model = _small_model()
    passenger = evaluate_passenger_choice(model, case.network.free_flow_time, parameters=_parameters())
    vehicle = solve_vehicle_service_column_generation(
        case.network,
        passenger.service_request_table,
        case.network.free_flow_time,
        **_vehicle_options(),
    )
    assert passenger.mass_residual < 1e-10
    assert vehicle.demand_residual < 1e-8
    assert vehicle.iterations >= 1
    assert len(vehicle.pattern_table) >= len(passenger.service_request_table)

    request_op, supply_op = build_passenger_vehicle_coupling_operators(model, vehicle)
    assert request_op.source.name == "passenger_behavioral_column"
    assert request_op.target.name == "mobility_service_request"
    assert request_op.input_measure == "person_flow"
    assert request_op.output_measure == "person_flow"
    assert supply_op.source.name == "vehicle_service_pattern"
    assert supply_op.target.name == "mobility_service_request"
    assert supply_op.input_measure == "vehicle_flow"
    assert supply_op.output_measure == "person_flow"

    person_flow = passenger.column_table["person_flow"].to_numpy(float)
    request_demand = np.asarray(request_op.matrix @ person_flow).ravel()
    vehicle_flow = vehicle.pattern_table["vehicle_flow"].to_numpy(float)
    supplied = np.asarray(supply_op.matrix @ vehicle_flow).ravel()
    assert np.max(np.abs(request_demand - passenger.service_request_table["demand"].to_numpy(float))) < 1e-9
    assert np.max(np.maximum(request_demand - supplied, 0.0), initial=0.0) < 1e-8


def test_coupled_static_multimodal_solve_and_queue_conservation() -> None:
    _, model = _small_model()
    result = solve_passenger_vehicle_equilibrium(
        model,
        parameters=_parameters(),
        max_iterations=45,
        damping=0.15,
        tolerance=1.2e-2,
        dynamic_feedback=False,
        queue_horizon_steps=130,
        vehicle_solver_options=_vehicle_options(),
        passenger_price_rule="average_cost",
    )
    assert result.fixed_point_residual < 1.2e-2
    assert result.passenger.mass_residual < 1e-10
    assert result.vehicle.demand_residual < 1e-8
    assert np.max(np.abs(result.queue.conservation_residual)) < 1e-8
    assert np.isclose(result.passenger.mode_share["share"].sum(), 1.0)
    assert result.road_link_vehicle_flow.shape == (model.case.network.n_links,)


def test_connected_mode_interaction_conserves_share_and_has_own_cost_effect() -> None:
    from tensormobility.profiles.passenger_vehicle import compute_mode_interaction_matrix

    _, model = _small_model()
    interaction = compute_mode_interaction_matrix(
        model,
        model.case.network.free_flow_time,
        parameters=_parameters(),
        delta_cost=0.25,
    )
    matrix = interaction.derivative_matrix
    assert interaction.max_row_sum_residual < 1e-10
    assert np.all(np.diag(matrix.to_numpy(float)) < 0.0)
    off_diagonal = matrix.to_numpy(float).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.all(off_diagonal >= -1e-12)
