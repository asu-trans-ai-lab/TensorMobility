from __future__ import annotations

"""Canonical well-posedness map for behavior--queue feedback.

This scalar two-route model is not presented as a proof for a full network.
It is an analytically inspectable continuation experiment showing how a
bounded learned substitution term and physical queue feedback change root
multiplicity, local contraction, and numerical fixed-point behavior.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class WellPosednessMap:
    table: pd.DataFrame
    lambda_behavior_values: np.ndarray
    lambda_queue_values: np.ndarray
    metadata: dict[str, float | str]


def two_route_costs(
    share_route_1: float,
    *,
    lambda_behavior: float,
    lambda_queue: float,
    demand: float = 1.0,
    capacity_share: float = 0.44,
    congestion_strength: float = 2.2,
    queue_strength: float = 8.0,
    learned_herding_strength: float = 5.0,
) -> tuple[float, float]:
    s = float(np.clip(share_route_1, 0.0, 1.0))
    x1 = demand * s
    x2 = demand * (1.0 - s)
    c1 = 10.0 + congestion_strength * s
    c2 = 10.0 + congestion_strength * (1.0 - s)
    q1 = max(x1 - demand * capacity_share, 0.0)
    q2 = max(x2 - demand * capacity_share, 0.0)
    c1 += lambda_queue * queue_strength * q1
    c2 += lambda_queue * queue_strength * q2
    # Positive feedback / herding: an alternative becomes more attractive as
    # its current share grows.  This is the exact mechanism that can destroy
    # monotonicity and produce several behavioral fixed points.
    centered = s - 0.5
    c1 -= lambda_behavior * learned_herding_strength * centered
    c2 += lambda_behavior * learned_herding_strength * centered
    return float(c1), float(c2)


def two_route_response(
    share_route_1: float,
    *,
    lambda_behavior: float,
    lambda_queue: float,
    temperature: float = 1.0,
) -> float:
    c1, c2 = two_route_costs(
        share_route_1,
        lambda_behavior=lambda_behavior,
        lambda_queue=lambda_queue,
    )
    z = np.clip((c1 - c2) / temperature, -700.0, 700.0)
    return float(1.0 / (1.0 + np.exp(z)))


def _roots(lambda_behavior: float, lambda_queue: float, temperature: float) -> list[float]:
    grid = np.linspace(0.0, 1.0, 2001)
    values = np.asarray([
        two_route_response(s, lambda_behavior=lambda_behavior, lambda_queue=lambda_queue, temperature=temperature) - s
        for s in grid
    ])
    roots: list[float] = []
    for i in range(len(grid) - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = float(values[i]), float(values[i + 1])
        if abs(fa) < 1e-10:
            roots.append(a)
        if fa * fb < 0.0:
            roots.append(float(brentq(
                lambda s: two_route_response(
                    s,
                    lambda_behavior=lambda_behavior,
                    lambda_queue=lambda_queue,
                    temperature=temperature,
                ) - s,
                a,
                b,
                xtol=1e-13,
                rtol=1e-13,
            )))
    if abs(values[-1]) < 1e-10:
        roots.append(1.0)
    deduped: list[float] = []
    for root in sorted(roots):
        if not deduped or abs(root - deduped[-1]) > 1e-6:
            deduped.append(root)
    return deduped


def _map_derivative(root: float, lambda_behavior: float, lambda_queue: float, temperature: float) -> float:
    eps = 1e-6
    left = max(root - eps, 0.0)
    right = min(root + eps, 1.0)
    if right == left:
        return np.nan
    return float((
        two_route_response(right, lambda_behavior=lambda_behavior, lambda_queue=lambda_queue, temperature=temperature)
        - two_route_response(left, lambda_behavior=lambda_behavior, lambda_queue=lambda_queue, temperature=temperature)
    ) / (right - left))


def _iterate(start: float, lambda_behavior: float, lambda_queue: float, temperature: float) -> tuple[float, bool, int]:
    s = float(start)
    for iteration in range(1, 501):
        raw = two_route_response(s, lambda_behavior=lambda_behavior, lambda_queue=lambda_queue, temperature=temperature)
        # Fixed damping is used only as a numerical diagnostic.  Root count and
        # derivative are evaluated on the undamped map.
        next_s = 0.65 * s + 0.35 * raw
        if abs(next_s - s) < 1e-10:
            return float(next_s), True, iteration
        s = float(next_s)
    return float(s), False, 500


def build_well_posedness_map(
    *,
    lambda_behavior_values: np.ndarray | None = None,
    lambda_queue_values: np.ndarray | None = None,
    temperature: float = 1.0,
) -> WellPosednessMap:
    lb_values = np.asarray(
        lambda_behavior_values if lambda_behavior_values is not None else np.linspace(0.0, 2.0, 21),
        dtype=float,
    )
    lq_values = np.asarray(
        lambda_queue_values if lambda_queue_values is not None else np.linspace(0.0, 1.5, 16),
        dtype=float,
    )
    rows: list[dict[str, float | int | str]] = []
    starts = (0.02, 0.25, 0.50, 0.75, 0.98)
    for lambda_queue in lq_values:
        for lambda_behavior in lb_values:
            roots = _roots(float(lambda_behavior), float(lambda_queue), temperature)
            derivatives = [abs(_map_derivative(r, float(lambda_behavior), float(lambda_queue), temperature)) for r in roots]
            iterates = [_iterate(s, float(lambda_behavior), float(lambda_queue), temperature) for s in starts]
            terminal = np.asarray([v[0] for v in iterates], dtype=float)
            all_converged = all(v[1] for v in iterates)
            spread = float(terminal.max() - terminal.min())
            if len(roots) > 1:
                classification = "multiple_fixed_points"
            elif roots and derivatives[0] < 0.95 and all_converged and spread < 1e-5:
                classification = "locally_contracting"
            elif all_converged and spread < 1e-3:
                classification = "sensitive_single_observed"
            else:
                classification = "numerically_unstable_or_start_sensitive"
            rows.append({
                "lambda_behavior": float(lambda_behavior),
                "lambda_queue": float(lambda_queue),
                "root_count": int(len(roots)),
                "roots": ";".join(f"{r:.8f}" for r in roots),
                "max_abs_raw_map_derivative": float(max(derivatives)) if derivatives else np.nan,
                "min_abs_raw_map_derivative": float(min(derivatives)) if derivatives else np.nan,
                "all_damped_iterations_converged": int(all_converged),
                "terminal_start_spread": spread,
                "max_iterations": int(max(v[2] for v in iterates)),
                "classification": classification,
            })
    return WellPosednessMap(
        table=pd.DataFrame(rows),
        lambda_behavior_values=lb_values,
        lambda_queue_values=lq_values,
        metadata={
            "temperature": float(temperature),
            "interpretation": "canonical_two_route_continuation_not_full_network_proof",
        },
    )
