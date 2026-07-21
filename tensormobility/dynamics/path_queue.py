from __future__ import annotations

"""Generic sparse-event path/departure fluid point-queue backend.

Unlike the earlier dense schedule cube, this implementation stores only active
arrival and completion events.  The state index is
(path-departure cohort, ordered position along its path), preserving route and
departure identity without materializing path x departure x link x time.
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from tensormobility.core.network_core import StaticNetwork


@dataclass(frozen=True)
class SparsePathQueueResult:
    path_departure_travel_time: np.ndarray
    queue_by_link_time: np.ndarray
    inflow_by_link_time: np.ndarray
    outflow_by_link_time: np.ndarray
    completed_by_path_departure: np.ndarray
    conservation_residual: np.ndarray
    total_demand: float
    final_completed: float
    final_queued: float
    final_scheduled: float
    max_total_queue: float
    max_link_queue: float
    dt_minutes: float
    history: pd.DataFrame
    metadata: dict[str, float | int | str]


def parse_link_index_sequence(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(int(v) for v in text.replace("|", ";").split(";") if str(v).strip())


def departure_matrix_from_path_table(path_table: pd.DataFrame, profile: np.ndarray) -> tuple[list[tuple[int, ...]], np.ndarray]:
    if "path_flow" not in path_table.columns or "link_index_sequence" not in path_table.columns:
        raise ValueError("path_table requires path_flow and link_index_sequence")
    shares = np.asarray(profile, dtype=float)
    if shares.ndim != 1 or shares.size < 1 or np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("departure profile must be a nonnegative vector summing to one")
    paths = [parse_link_index_sequence(v) for v in path_table["link_index_sequence"]]
    flows = path_table["path_flow"].to_numpy(float)
    if np.any(flows < -1e-10):
        raise ValueError("path flow must be nonnegative")
    return paths, flows[:, None] * shares[None, :]


def run_sparse_path_point_queue(
    network: StaticNetwork,
    paths: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    path_departure_flow: np.ndarray,
    *,
    dt_minutes: float = 5.0,
    horizon_steps: int = 180,
    capacity_multiplier: np.ndarray | float = 1.0,
    incomplete_penalty_minutes: float = 180.0,
) -> SparsePathQueueResult:
    path_list = [tuple(int(v) for v in p) for p in paths]
    dep = np.asarray(path_departure_flow, dtype=float)
    n_paths = len(path_list)
    if dep.ndim != 2 or dep.shape[0] != n_paths:
        raise ValueError("path_departure_flow must be path x departure")
    if np.any(dep < -1e-10) or not np.all(np.isfinite(dep)):
        raise ValueError("path_departure_flow must be finite and nonnegative")
    if any(len(path) == 0 for path in path_list):
        raise ValueError("queue backend requires nonempty physical paths")
    if any(any(link < 0 or link >= network.n_links for link in path) for path in path_list):
        raise ValueError("path contains invalid link index")
    if dt_minutes <= 0 or horizon_steps <= dep.shape[1]:
        raise ValueError("invalid time configuration")

    mult = np.asarray(capacity_multiplier, dtype=float)
    if mult.ndim == 0:
        mult = np.full(network.n_links, float(mult))
    if mult.shape != (network.n_links,) or np.any(mult <= 0):
        raise ValueError("capacity multiplier must be positive and link-sized")

    n_departures = dep.shape[1]
    n_cohorts = n_paths * n_departures
    cohort_demand = dep.reshape(-1)
    cohort_path = np.repeat(np.arange(n_paths, dtype=int), n_departures)
    cohort_departure = np.tile(np.arange(n_departures, dtype=int), n_paths)

    # Compact valid-state representation.
    first_state = np.full(n_cohorts, -1, dtype=int)
    state_link: list[int] = []
    state_cohort: list[int] = []
    next_state: list[int] = []
    for cohort, path_index in enumerate(cohort_path):
        path = path_list[int(path_index)]
        previous = -1
        for link_idx in path:
            state = len(state_link)
            state_link.append(int(link_idx))
            state_cohort.append(int(cohort))
            next_state.append(-1)
            if previous < 0:
                first_state[cohort] = state
            else:
                next_state[previous] = state
            previous = state
    state_link_arr = np.asarray(state_link, dtype=int)
    state_cohort_arr = np.asarray(state_cohort, dtype=int)
    next_state_arr = np.asarray(next_state, dtype=int)
    n_states = len(state_link_arr)

    link_members = [np.flatnonzero(state_link_arr == a) for a in range(network.n_links)]
    service = network.capacity * mult * (dt_minutes / 60.0)
    ff_steps = np.maximum(1, np.ceil(network.free_flow_time / dt_minutes).astype(int))
    tail = int(ff_steps.max()) + 2
    schedule_horizon = horizon_steps + tail

    arrival_events: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(schedule_horizon)]
    completion_events: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(schedule_horizon)]
    for cohort, amount in enumerate(cohort_demand):
        if amount > 0:
            departure = int(cohort_departure[cohort])
            arrival_events[departure][int(first_state[cohort])] += float(amount)

    q = np.zeros(n_states, dtype=float)
    queue_hist = np.zeros((network.n_links, horizon_steps), dtype=float)
    inflow_hist = np.zeros((network.n_links, horizon_steps), dtype=float)
    outflow_hist = np.zeros((network.n_links, horizon_steps), dtype=float)
    completed = np.zeros(n_cohorts, dtype=float)
    completion_time_sum = np.zeros(n_cohorts, dtype=float)
    residual = np.zeros(horizon_steps, dtype=float)
    total_demand = float(cohort_demand.sum())

    for t in range(horizon_steps):
        for cohort, amount in completion_events[t].items():
            completed[int(cohort)] += float(amount)
            completion_time_sum[int(cohort)] += float(amount) * (t * dt_minutes)

        for state, amount in arrival_events[t].items():
            q[int(state)] += float(amount)
            inflow_hist[int(state_link_arr[int(state)]), t] += float(amount)

        for link_idx, members in enumerate(link_members):
            if members.size == 0:
                continue
            total_available = float(q[members].sum())
            if total_available <= 0:
                continue
            scale = min(1.0, float(service[link_idx]) / total_available)
            discharged = q[members] * scale
            q[members] -= discharged
            outflow_hist[link_idx, t] = float(discharged.sum())
            arrival_time = min(t + int(ff_steps[link_idx]), schedule_horizon - 1)
            for state, amount in zip(members, discharged, strict=True):
                if amount <= 0:
                    continue
                nxt = int(next_state_arr[int(state)])
                cohort = int(state_cohort_arr[int(state)])
                if nxt >= 0:
                    arrival_events[arrival_time][nxt] += float(amount)
                else:
                    completion_events[arrival_time][cohort] += float(amount)

        for link_idx, members in enumerate(link_members):
            if members.size:
                queue_hist[link_idx, t] = float(q[members].sum())

        future_arrivals = sum(sum(events.values()) for events in arrival_events[t + 1 :])
        future_completions = sum(sum(events.values()) for events in completion_events[t + 1 :])
        accounted = float(q.sum()) + float(completed.sum()) + float(future_arrivals) + float(future_completions)
        residual[t] = accounted - total_demand

    remaining = np.maximum(cohort_demand - completed, 0.0)
    observed_tt = np.zeros(n_cohorts, dtype=float)
    departure_minutes = cohort_departure * dt_minutes
    completed_mask = completed > 1e-12
    observed_tt[completed_mask] = completion_time_sum[completed_mask] / completed[completed_mask] - departure_minutes[completed_mask]
    ff_path = np.asarray([
        float(network.free_flow_time[np.asarray(path_list[int(p)], dtype=int)].sum()) for p in cohort_path
    ])
    observed_tt[~completed_mask] = ff_path[~completed_mask] + incomplete_penalty_minutes
    partial = completed_mask & (remaining > 1e-8)
    if np.any(partial):
        fraction = completed[partial] / np.maximum(cohort_demand[partial], 1e-12)
        observed_tt[partial] += (1.0 - fraction) * incomplete_penalty_minutes
    observed_tt = np.maximum(observed_tt, ff_path)

    future_arrivals = float(sum(sum(events.values()) for events in arrival_events[horizon_steps:]))
    future_completions = float(sum(sum(events.values()) for events in completion_events[horizon_steps:]))
    final_scheduled = future_arrivals + future_completions
    history = pd.DataFrame({
        "time_step": np.arange(horizon_steps),
        "minutes": np.arange(horizon_steps) * dt_minutes,
        "total_queue": queue_hist.sum(axis=0),
        "total_inflow": inflow_hist.sum(axis=0),
        "total_outflow": outflow_hist.sum(axis=0),
        "conservation_residual": residual,
    })
    return SparsePathQueueResult(
        path_departure_travel_time=observed_tt.reshape(n_paths, n_departures),
        queue_by_link_time=queue_hist,
        inflow_by_link_time=inflow_hist,
        outflow_by_link_time=outflow_hist,
        completed_by_path_departure=completed.reshape(n_paths, n_departures),
        conservation_residual=residual,
        total_demand=total_demand,
        final_completed=float(completed.sum()),
        final_queued=float(q.sum()),
        final_scheduled=float(final_scheduled),
        max_total_queue=float(queue_hist.sum(axis=0).max(initial=0.0)),
        max_link_queue=float(queue_hist.max(initial=0.0)),
        dt_minutes=float(dt_minutes),
        history=history,
        metadata={
            "backend": "internal_sparse_event_path_point_queue",
            "spillback": "false",
            "states": int(n_states),
            "cohorts": int(n_cohorts),
            "certificate": "mass_conservation_and_time_refinement_only",
        },
    )
