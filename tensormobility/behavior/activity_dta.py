from __future__ import annotations

"""Dynamic activity-behavior / fluid-queue fixed-point harness.

The solver reports a fixed-point residual, not a Wardrop or FW certificate.
This distinction is deliberate: the certified static slices and the dynamic
coupled experiment have different mathematical guarantees.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from tensormobility.behavior.activity import ActivitySTBModel, BehaviorParameters, BehaviorResult, hierarchical_behavior_choice
from tensormobility.dynamics.cohort_queue import CohortQueueResult, run_cohort_point_queue


@dataclass(frozen=True)
class ActivityDTAResult:
    behavior: BehaviorResult
    queue: CohortQueueResult
    path_departure_time: np.ndarray
    history: pd.DataFrame
    converged: bool
    fixed_point_residual: float
    iterations: int
    wall_time_seconds: float
    metadata: dict[str, float | int | str]


def deterministic_safe_residual(model: ActivitySTBModel, strength: float = 0.0) -> np.ndarray:
    """Bounded, flow-independent interaction residual for controlled tests.

    This represents the only learning-style term admitted to the certified
    static behavior layer.  It can express class x activity x mode interactions
    but cannot react to endogenous flow, queue, or solver residuals.
    """
    if strength < 0 or strength > 1:
        raise ValueError("strength must lie in [0, 1]")
    c = model.columns
    raw = (
        0.35 * (c["class_index"].to_numpy(float) - 1.0) * (c["mode_index"].to_numpy(float) - 0.5)
        + 0.25 * (c["activity_index"].to_numpy(float) - 1.0) * (c["departure_index"].to_numpy(float) / max(model.n_departures - 1, 1) - 0.5)
        + 0.15 * np.sin(0.7 * c["destination"].to_numpy(float))
    )
    max_abs = max(float(np.max(np.abs(raw))), 1e-12)
    return strength * 0.70 * raw / max_abs


def solve_activity_dta(
    model: ActivitySTBModel,
    *,
    parameters: BehaviorParameters = BehaviorParameters(),
    capacity_multiplier: np.ndarray | float = 1.0,
    max_iterations: int = 30,
    tolerance: float = 5e-3,
    dt_minutes: float = 5.0,
    horizon_steps: int = 180,
    initial_time_multiplier: float = 1.0,
    queue_feedback_strength: float = 1.0,
    safe_residual_strength: float = 0.0,
    minimum_iterations: int = 6,
) -> ActivityDTAResult:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if queue_feedback_strength < 0:
        raise ValueError("queue_feedback_strength must be nonnegative")
    if initial_time_multiplier <= 0:
        raise ValueError("initial_time_multiplier must be positive")

    n_paths, n_dep = model.path_departure_shape
    ff = np.repeat(model.path_ff_time[:, None], n_dep, axis=1)
    path_time = initial_time_multiplier * ff
    safe_residual = deterministic_safe_residual(model, safe_residual_strength)
    rows: list[dict[str, float]] = []
    start = time.perf_counter()
    final_behavior: BehaviorResult | None = None
    final_queue: CohortQueueResult | None = None
    residual = np.inf

    for iteration in range(1, max_iterations + 1):
        behavior = hierarchical_behavior_choice(
            model,
            path_time,
            parameters=parameters,
            safe_column_residual=safe_residual,
        )
        queue = run_cohort_point_queue(
            model.path_set,
            behavior.path_departure_vehicle,
            dt_minutes=dt_minutes,
            horizon_steps=horizon_steps,
            capacity_multiplier=capacity_multiplier,
        )
        raw_time = ff + queue_feedback_strength * (queue.path_departure_travel_time - ff)
        raw_time = np.maximum(raw_time, ff)
        step = 1.0 / max(iteration, 3)
        next_time = (1.0 - step) * path_time + step * raw_time
        residual = float(
            np.linalg.norm(next_time - path_time, ord=1)
            / max(np.linalg.norm(path_time, ord=1), 1.0)
        )
        share_change = np.nan
        if final_behavior is not None:
            share_change = float(
                np.linalg.norm(behavior.person_column_flow - final_behavior.person_column_flow, ord=1)
                / max(behavior.person_column_flow.sum(), 1.0)
            )
        rows.append({
            "iteration": iteration,
            "fixed_point_residual": residual,
            "column_flow_change": share_change,
            "max_total_queue": queue.max_total_queue,
            "queue_conservation_residual": float(np.max(np.abs(queue.conservation_residual))),
            "behavior_mass_residual": behavior.mass_residual,
            "completed_fraction": queue.final_completed / max(queue.total_demand, 1.0),
        })
        path_time = next_time
        final_behavior = behavior
        final_queue = queue
        if iteration >= minimum_iterations and residual <= tolerance:
            break

    assert final_behavior is not None and final_queue is not None
    elapsed = time.perf_counter() - start
    return ActivityDTAResult(
        behavior=final_behavior,
        queue=final_queue,
        path_departure_time=path_time,
        history=pd.DataFrame(rows),
        converged=bool(residual <= tolerance),
        fixed_point_residual=float(residual),
        iterations=len(rows),
        wall_time_seconds=float(elapsed),
        metadata={
            "queue_feedback_strength": float(queue_feedback_strength),
            "safe_residual_strength": float(safe_residual_strength),
            "dt_minutes": float(dt_minutes),
            "horizon_steps": int(horizon_steps),
            "certificate": "fixed_point_residual_only",
        },
    )


def compare_behavior_shares(base: ActivityDTAResult, scenario: ActivityDTAResult) -> pd.DataFrame:
    left = base.behavior.shares.rename(columns={"flow": "base_flow", "share": "base_share"})
    right = scenario.behavior.shares.rename(columns={"flow": "scenario_flow", "share": "scenario_share"})
    merged = left.merge(right, on=["dimension", "alternative"], how="outer").fillna(0.0)
    merged["share_change"] = merged["scenario_share"] - merged["base_share"]
    merged["absolute_share_change"] = merged["share_change"].abs()
    return merged.sort_values(["dimension", "absolute_share_change"], ascending=[True, False]).reset_index(drop=True)


def multi_start_stability(
    model: ActivitySTBModel,
    *,
    queue_feedback_strength: float,
    safe_residual_strength: float,
    initial_multipliers: tuple[float, ...] = (0.8, 1.0, 1.4),
    max_iterations: int = 20,
    demand_horizon_steps: int = 150,
) -> tuple[pd.DataFrame, list[ActivityDTAResult]]:
    results: list[ActivityDTAResult] = []
    for multiplier in initial_multipliers:
        results.append(
            solve_activity_dta(
                model,
                queue_feedback_strength=queue_feedback_strength,
                safe_residual_strength=safe_residual_strength,
                initial_time_multiplier=multiplier,
                max_iterations=max_iterations,
                horizon_steps=demand_horizon_steps,
                tolerance=7e-3,
            )
        )
    reference = results[0]
    rows: list[dict[str, float]] = []
    for multiplier, result in zip(initial_multipliers, results, strict=True):
        flow_distance = float(
            np.linalg.norm(result.behavior.person_column_flow - reference.behavior.person_column_flow, ord=1)
            / max(reference.behavior.person_column_flow.sum(), 1.0)
        )
        time_distance = float(
            np.linalg.norm(result.path_departure_time - reference.path_departure_time, ord=1)
            / max(np.linalg.norm(reference.path_departure_time, ord=1), 1.0)
        )
        rows.append({
            "initial_time_multiplier": multiplier,
            "converged": float(result.converged),
            "fixed_point_residual": result.fixed_point_residual,
            "flow_distance_to_first_start": flow_distance,
            "time_distance_to_first_start": time_distance,
            "iterations": result.iterations,
            "wall_time_seconds": result.wall_time_seconds,
        })
    return pd.DataFrame(rows), results
