from __future__ import annotations

"""Path-departure cohort fluid point-queue loading.

This module is the temporal STB operator used by the activity-DTA harness.
Each commodity is one physical-path/departure cohort.  The state is stored by
(path-departure cohort, position along path), avoiding a dense
path x departure x link x time tensor while preserving exact path identity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tensormobility.dta.sioux_falls import SiouxFallsPathSet


@dataclass(frozen=True)
class CohortQueueResult:
    path_departure_travel_time: np.ndarray
    queue_by_link_time: np.ndarray
    inflow_by_link_time: np.ndarray
    outflow_by_link_time: np.ndarray
    completed_by_cohort: np.ndarray
    conservation_residual: np.ndarray
    total_demand: float
    final_completed: float
    final_queued: float
    final_in_transit: float
    max_total_queue: float
    max_link_queue: float
    dt_minutes: float
    history: pd.DataFrame


def run_cohort_point_queue(
    path_set: SiouxFallsPathSet,
    path_departures: np.ndarray,
    *,
    dt_minutes: float = 5.0,
    horizon_steps: int = 180,
    capacity_multiplier: np.ndarray | float = 1.0,
    incomplete_penalty_minutes: float = 180.0,
) -> CohortQueueResult:
    dep = np.asarray(path_departures, dtype=float)
    n_paths = len(path_set.paths)
    n_links = len(path_set.link_pairs)
    if dep.ndim != 2 or dep.shape[0] != n_paths:
        raise ValueError("path_departures must have shape physical_path x departure_interval")
    if np.any(dep < -1e-10) or not np.all(np.isfinite(dep)):
        raise ValueError("path_departures must be finite and nonnegative")
    n_departures = dep.shape[1]
    if horizon_steps <= n_departures:
        raise ValueError("horizon_steps must extend beyond the departure window")
    if dt_minutes <= 0:
        raise ValueError("dt_minutes must be positive")

    mult = np.asarray(capacity_multiplier, dtype=float)
    if mult.ndim == 0:
        mult = np.full(n_links, float(mult))
    if mult.shape != (n_links,) or np.any(mult <= 0):
        raise ValueError("capacity_multiplier has wrong shape or nonpositive values")

    link_index = {pair: i for i, pair in enumerate(path_set.link_pairs)}
    path_link_sequence = [
        [link_index[(u, v)] for u, v in zip(nodes[:-1], nodes[1:])]
        for nodes in path_set.paths
    ]
    max_positions = max(len(seq) for seq in path_link_sequence)
    position_link = np.full((n_paths, max_positions), -1, dtype=int)
    path_length = np.zeros(n_paths, dtype=int)
    for p, seq in enumerate(path_link_sequence):
        position_link[p, : len(seq)] = seq
        path_length[p] = len(seq)

    n_cohorts = n_paths * n_departures
    cohort_path = np.repeat(np.arange(n_paths, dtype=int), n_departures)
    cohort_departure = np.tile(np.arange(n_departures, dtype=int), n_paths)
    cohort_demand = dep.reshape(-1)
    valid_position = np.zeros((n_cohorts, max_positions), dtype=bool)
    cohort_position_link = np.full((n_cohorts, max_positions), -1, dtype=int)
    for c, p in enumerate(cohort_path):
        valid_position[c, : path_length[p]] = True
        cohort_position_link[c, : path_length[p]] = position_link[p, : path_length[p]]

    # Flat state members for each physical link.  This is the key sparse index
    # replacing a link x path x departure x time tensor.
    link_members: list[np.ndarray] = []
    flat_links = cohort_position_link.reshape(-1)
    for a in range(n_links):
        link_members.append(np.flatnonzero(flat_links == a))

    links = path_set.links
    service = links["capacity"].to_numpy(float) * mult * (dt_minutes / 60.0)
    ff_steps = np.maximum(1, np.ceil(links["free_flow_time"].to_numpy(float) / dt_minutes).astype(int))
    tail = int(ff_steps.max()) + 2

    q = np.zeros((n_cohorts, max_positions), dtype=float)
    scheduled = np.zeros((n_cohorts, max_positions, horizon_steps + tail), dtype=float)
    completion_schedule = np.zeros((n_cohorts, horizon_steps + tail), dtype=float)
    for c in range(n_cohorts):
        departure = int(cohort_departure[c])
        scheduled[c, 0, departure] = cohort_demand[c]

    queue_hist = np.zeros((n_links, horizon_steps), dtype=float)
    inflow_hist = np.zeros((n_links, horizon_steps), dtype=float)
    outflow_hist = np.zeros((n_links, horizon_steps), dtype=float)
    completed_by_cohort = np.zeros(n_cohorts, dtype=float)
    completion_time_sum = np.zeros(n_cohorts, dtype=float)
    residual = np.zeros(horizon_steps, dtype=float)
    total_demand = float(cohort_demand.sum())

    for t in range(horizon_steps):
        arriving_completion = completion_schedule[:, t]
        completed_by_cohort += arriving_completion
        completion_time_sum += arriving_completion * (t * dt_minutes)

        arrivals = scheduled[:, :, t]
        # Aggregate entering flow by the physical link at each path position.
        flat_arrivals = arrivals.reshape(-1)
        for a, members in enumerate(link_members):
            if members.size:
                inflow_hist[a, t] = float(flat_arrivals[members].sum())

        available = q + arrivals
        q_new = available.copy()
        flat_available = available.reshape(-1)
        flat_q_new = q_new.reshape(-1)

        for a, members in enumerate(link_members):
            if members.size == 0:
                continue
            total_available = float(flat_available[members].sum())
            if total_available <= 0:
                continue
            scale = min(1.0, float(service[a]) / total_available)
            discharged = flat_available[members] * scale
            flat_q_new[members] -= discharged
            outflow_hist[a, t] = float(discharged.sum())

            arrival_time = t + int(ff_steps[a])
            if arrival_time >= scheduled.shape[2]:
                arrival_time = scheduled.shape[2] - 1
            cohort_idx, pos_idx = np.unravel_index(members, (n_cohorts, max_positions))
            for c, pos, amount in zip(cohort_idx, pos_idx, discharged, strict=True):
                if amount <= 0:
                    continue
                p = int(cohort_path[c])
                if pos + 1 < int(path_length[p]):
                    scheduled[c, pos + 1, arrival_time] += float(amount)
                else:
                    completion_schedule[c, arrival_time] += float(amount)

        q = np.maximum(q_new, 0.0)
        flat_q = q.reshape(-1)
        for a, members in enumerate(link_members):
            if members.size:
                queue_hist[a, t] = float(flat_q[members].sum())

        future_scheduled = float(scheduled[:, :, t + 1 :].sum())
        future_completion = float(completion_schedule[:, t + 1 :].sum())
        accounted = float(completed_by_cohort.sum()) + float(q.sum()) + future_scheduled + future_completion
        residual[t] = accounted - total_demand

    remaining = np.maximum(cohort_demand - completed_by_cohort, 0.0)
    observed_tt = np.zeros(n_cohorts, dtype=float)
    completed_mask = completed_by_cohort > 1e-12
    departure_minutes = cohort_departure * dt_minutes
    observed_tt[completed_mask] = (
        completion_time_sum[completed_mask] / completed_by_cohort[completed_mask]
        - departure_minutes[completed_mask]
    )
    # Penalize incomplete cohorts without silently treating truncation as a
    # valid travel time.  The result object still exposes remaining mass.
    ff_path = np.asarray([
        path_set.instance.columns.iloc[p]["free_flow_path_time"] for p in cohort_path
    ], dtype=float)
    observed_tt[~completed_mask] = ff_path[~completed_mask] + incomplete_penalty_minutes
    partial = completed_mask & (remaining > 1e-8)
    if np.any(partial):
        completion_fraction = completed_by_cohort[partial] / np.maximum(cohort_demand[partial], 1e-12)
        observed_tt[partial] += (1.0 - completion_fraction) * incomplete_penalty_minutes
    observed_tt = np.maximum(observed_tt, ff_path)

    future_scheduled_tail = float(scheduled[:, :, horizon_steps:].sum())
    future_completion_tail = float(completion_schedule[:, horizon_steps:].sum())
    final_in_transit = future_scheduled_tail + future_completion_tail
    history = pd.DataFrame({
        "time_step": np.arange(horizon_steps),
        "minutes": np.arange(horizon_steps) * dt_minutes,
        "total_queue": queue_hist.sum(axis=0),
        "total_inflow": inflow_hist.sum(axis=0),
        "total_outflow": outflow_hist.sum(axis=0),
        "cumulative_completed": np.cumsum(completion_schedule[:, :horizon_steps].sum(axis=0)),
        "conservation_residual": residual,
    })

    return CohortQueueResult(
        path_departure_travel_time=observed_tt.reshape(n_paths, n_departures),
        queue_by_link_time=queue_hist,
        inflow_by_link_time=inflow_hist,
        outflow_by_link_time=outflow_hist,
        completed_by_cohort=completed_by_cohort.reshape(n_paths, n_departures),
        conservation_residual=residual,
        total_demand=total_demand,
        final_completed=float(completed_by_cohort.sum()),
        final_queued=float(q.sum()),
        final_in_transit=float(final_in_transit),
        max_total_queue=float(queue_hist.sum(axis=0).max(initial=0.0)),
        max_link_queue=float(queue_hist.max(initial=0.0)),
        dt_minutes=float(dt_minutes),
        history=history,
    )
