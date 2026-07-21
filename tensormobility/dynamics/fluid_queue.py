from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tensormobility.dta.sioux_falls import SiouxFallsPathSet


@dataclass(frozen=True)
class FluidQueueResult:
    queue: np.ndarray              # link x time, end-of-step queue
    inflow: np.ndarray             # link x time
    outflow: np.ndarray            # link x time
    completed: np.ndarray          # cumulative completed flow
    conservation_residual: np.ndarray
    total_demand: float
    final_completed: float
    final_queued: float
    final_in_transit: float
    final_unreleased: float
    max_total_queue: float
    max_link_queue: float
    history: pd.DataFrame


def smooth_departure_profile(n_intervals: int = 12, center: float = 5.0, spread: float = 2.1) -> np.ndarray:
    x = np.arange(n_intervals, dtype=float)
    p = np.exp(-0.5 * ((x - center) / spread) ** 2)
    p /= p.sum()
    return p


def build_departure_tensor(path_flow: np.ndarray, profile: np.ndarray) -> np.ndarray:
    f = np.asarray(path_flow, dtype=float)
    p = np.asarray(profile, dtype=float)
    if f.ndim != 1 or p.ndim != 1:
        raise ValueError("path_flow and profile must be one-dimensional")
    if np.any(f < -1e-10) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("Invalid flow/profile")
    return f[:, None] * p[None, :]


def run_fluid_point_queue(
    path_set: SiouxFallsPathSet,
    path_departures: np.ndarray,
    dt_minutes: float = 5.0,
    horizon_steps: int = 96,
    capacity_multiplier: np.ndarray | float = 1.0,
) -> FluidQueueResult:
    """Path-resolved fluid point-queue loading without spillback.

    At each link and time step, the queue composition is discharged by one
    common FIFO fluid scaling factor min(1, service/available). Outflow then
    traverses the link's free-flow delay before entering the next link.
    """
    dep = np.asarray(path_departures, dtype=float)
    n_paths = len(path_set.paths)
    n_links = len(path_set.link_pairs)
    if dep.ndim != 2 or dep.shape[0] != n_paths:
        raise ValueError("path_departures must be path x departure_time")
    if horizon_steps <= dep.shape[1]:
        raise ValueError("horizon must extend beyond departure window")

    mult = np.asarray(capacity_multiplier, dtype=float)
    if mult.ndim == 0:
        mult = np.full(n_links, float(mult))
    if mult.shape != (n_links,) or np.any(mult <= 0):
        raise ValueError("capacity_multiplier has wrong shape")

    links = path_set.links
    service = links["capacity"].to_numpy(float) * mult * (dt_minutes / 60.0)
    ff_steps = np.maximum(1, np.ceil(links["free_flow_time"].to_numpy(float) / dt_minutes).astype(int))
    link_index = {pair: i for i, pair in enumerate(path_set.link_pairs)}

    path_link_sequence: list[list[int]] = []
    next_link = np.full((n_paths, n_links), -2, dtype=int)
    first_link = np.zeros(n_paths, dtype=int)
    for p, nodes in enumerate(path_set.paths):
        seq = [link_index[(u, v)] for u, v in zip(nodes[:-1], nodes[1:])]
        path_link_sequence.append(seq)
        first_link[p] = seq[0]
        for pos, a in enumerate(seq):
            next_link[p, a] = seq[pos + 1] if pos + 1 < len(seq) else -1

    external = np.zeros((n_paths, horizon_steps), dtype=float)
    external[:, :dep.shape[1]] = dep
    # Internal arrivals are future arrivals after traversing a link.
    internal = np.zeros((n_paths, n_links, horizon_steps + int(ff_steps.max()) + 1), dtype=float)
    q = np.zeros((n_paths, n_links), dtype=float)
    queue_hist = np.zeros((n_links, horizon_steps), dtype=float)
    inflow_hist = np.zeros((n_links, horizon_steps), dtype=float)
    outflow_hist = np.zeros((n_links, horizon_steps), dtype=float)
    completed = np.zeros(horizon_steps, dtype=float)
    residual = np.zeros(horizon_steps, dtype=float)
    cumulative_completed = 0.0
    total_demand = float(dep.sum())

    for t in range(horizon_steps):
        arrivals = internal[:, :, t].copy()
        ext = external[:, t]
        arrivals[np.arange(n_paths), first_link] += ext
        inflow_hist[:, t] = arrivals.sum(axis=0)
        available = q + arrivals
        q_new = available.copy()

        for a in range(n_links):
            total_available = float(available[:, a].sum())
            if total_available <= 0:
                continue
            scale = min(1.0, float(service[a]) / total_available)
            discharged = available[:, a] * scale
            q_new[:, a] -= discharged
            outflow_hist[a, t] = discharged.sum()
            active_paths = np.flatnonzero(discharged > 0)
            arrival_time = t + int(ff_steps[a])
            for p in active_paths:
                amount = float(discharged[p])
                nxt = int(next_link[p, a])
                if nxt == -1:
                    cumulative_completed += amount
                elif nxt >= 0 and arrival_time < internal.shape[2]:
                    internal[p, nxt, arrival_time] += amount
                elif nxt >= 0:
                    # Beyond stored horizon; preserve as in-transit tail mass.
                    internal[p, nxt, -1] += amount
        q = np.maximum(q_new, 0.0)
        queue_hist[:, t] = q.sum(axis=0)
        completed[t] = cumulative_completed

        future_external = float(external[:, t + 1 :].sum()) if t + 1 < horizon_steps else 0.0
        future_internal = float(internal[:, :, t + 1 :].sum())
        accounted = cumulative_completed + float(q.sum()) + future_internal + future_external
        residual[t] = accounted - total_demand

    final_unreleased = 0.0
    final_in_transit = float(internal[:, :, horizon_steps:].sum())
    history = pd.DataFrame({
        "time_step": np.arange(horizon_steps),
        "minutes": np.arange(horizon_steps) * dt_minutes,
        "total_queue": queue_hist.sum(axis=0),
        "total_inflow": inflow_hist.sum(axis=0),
        "total_outflow": outflow_hist.sum(axis=0),
        "cumulative_completed": completed,
        "conservation_residual": residual,
    })
    return FluidQueueResult(
        queue=queue_hist,
        inflow=inflow_hist,
        outflow=outflow_hist,
        completed=completed,
        conservation_residual=residual,
        total_demand=total_demand,
        final_completed=float(completed[-1]),
        final_queued=float(queue_hist[:, -1].sum()),
        final_in_transit=final_in_transit,
        final_unreleased=final_unreleased,
        max_total_queue=float(queue_hist.sum(axis=0).max()),
        max_link_queue=float(queue_hist.max()),
        history=history,
    )
