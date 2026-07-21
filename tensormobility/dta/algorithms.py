from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from tensormobility.core.instance import STBInstance
from tensormobility.core.objective import STBObjective
from tensormobility.core.projections import feasible_residual, project_simplex


@dataclass
class AlgorithmResult:
    name: str
    flow: np.ndarray
    history: pd.DataFrame
    objective: float
    relative_gap: float
    feasibility_residual: float
    active_columns: int
    pricing_calls: int
    wall_time_seconds: float
    metadata: dict[str, float | int | str]


def linear_oracle(gradient: np.ndarray, group_columns: list[np.ndarray], demands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros_like(gradient)
    best = np.zeros(len(group_columns), dtype=int)
    for g, (idx, demand) in enumerate(zip(group_columns, demands, strict=True)):
        j = int(idx[np.argmin(gradient[idx])])
        s[j] = demand
        best[g] = j
    return s, best


def fw_gap(f: np.ndarray, gradient: np.ndarray, group_columns: list[np.ndarray], demands: np.ndarray) -> tuple[float, np.ndarray]:
    s, best = linear_oracle(gradient, group_columns, demands)
    return float(np.dot(gradient, f - s)), best


def relative_gap(gap: float, objective: float) -> float:
    return max(gap, 0.0) / max(abs(objective), 1.0)


def initial_flow(instance: STBInstance) -> np.ndarray:
    obj = STBObjective(instance)
    grad0 = obj.gradient(np.zeros(instance.n_columns))
    f, _ = linear_oracle(grad0, instance.group_columns, instance.demands)
    return f


def _line_search(objective: Callable[[np.ndarray], float], f: np.ndarray, direction: np.ndarray) -> float:
    if np.max(np.abs(direction)) < 1e-14:
        return 0.0
    result = minimize_scalar(
        lambda gamma: objective(f + float(gamma) * direction),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 80},
    )
    if not result.success:
        return 2.0 / 3.0
    return float(np.clip(result.x, 0.0, 1.0))


def solve_exact_fw(
    instance: STBInstance,
    max_iterations: int = 250,
    tolerance: float = 1e-7,
) -> AlgorithmResult:
    obj = STBObjective(instance)
    f = initial_flow(instance)
    rows: list[dict[str, float]] = []
    start = time.perf_counter()
    pricing_calls = 0

    for iteration in range(1, max_iterations + 1):
        value = obj.value(f)
        grad = obj.gradient(f)
        s, _ = linear_oracle(grad, instance.group_columns, instance.demands)
        pricing_calls += instance.n_groups
        gap = float(np.dot(grad, f - s))
        rel = relative_gap(gap, value)
        rows.append(
            {
                "iteration": iteration,
                "objective": value,
                "gap": gap,
                "relative_gap": rel,
                "active_columns": int(np.count_nonzero(f > 1e-9)),
            }
        )
        if rel <= tolerance:
            break
        gamma = _line_search(obj.value, f, s - f)
        f = f + gamma * (s - f)

    elapsed = time.perf_counter() - start
    final_value = obj.value(f)
    final_gap, _ = fw_gap(f, obj.gradient(f), instance.group_columns, instance.demands)
    return AlgorithmResult(
        name="exact_fw",
        flow=f,
        history=pd.DataFrame(rows),
        objective=final_value,
        relative_gap=relative_gap(final_gap, final_value),
        feasibility_residual=feasible_residual(f, instance.group_columns, instance.demands),
        active_columns=int(np.count_nonzero(f > 1e-9)),
        pricing_calls=pricing_calls,
        wall_time_seconds=elapsed,
        metadata={},
    )


def _restricted_projected_step(
    f: np.ndarray,
    active: np.ndarray,
    objective: STBObjective,
    instance: STBInstance,
    initial_step: float,
) -> tuple[np.ndarray, float, bool]:
    grad = objective.gradient(f)
    current = objective.value(f)
    step = initial_step
    for _ in range(24):
        candidate = f.copy()
        for demand, idx in zip(instance.demands, instance.group_columns, strict=True):
            aidx = idx[active[idx]]
            if aidx.size == 0:
                aidx = np.array([idx[np.argmin(grad[idx])]], dtype=int)
                active[aidx] = True
            candidate[idx] = 0.0
            candidate[aidx] = project_simplex(f[aidx] - step * grad[aidx], demand)
        direction = candidate - f
        directional = float(np.dot(grad, direction))
        new_value = objective.value(candidate)
        if new_value <= current + 1e-4 * directional:
            return candidate, step, True
        step *= 0.5
    return f, step, False


def solve_fw_gp(
    instance: STBInstance,
    max_iterations: int = 160,
    gp_sweeps: int = 4,
    tolerance: float = 1e-7,
) -> AlgorithmResult:
    obj = STBObjective(instance)
    f = initial_flow(instance)
    active = f > 1e-12
    rows: list[dict[str, float]] = []
    start = time.perf_counter()
    pricing_calls = 0
    step = 0.5

    for iteration in range(1, max_iterations + 1):
        grad = obj.gradient(f)
        _, best = linear_oracle(grad, instance.group_columns, instance.demands)
        pricing_calls += instance.n_groups
        active[best] = True

        accepted = 0
        for _ in range(gp_sweeps):
            f_new, used_step, ok = _restricted_projected_step(f, active, obj, instance, step)
            step = min(max(used_step * (1.5 if ok else 1.0), 1e-6), 2.0)
            if not ok:
                break
            if np.linalg.norm(f_new - f, ord=1) < 1e-10:
                break
            f = f_new
            accepted += 1

        active |= f > 1e-9
        value = obj.value(f)
        gap, _ = fw_gap(f, obj.gradient(f), instance.group_columns, instance.demands)
        rel = relative_gap(gap, value)
        rows.append(
            {
                "iteration": iteration,
                "objective": value,
                "gap": gap,
                "relative_gap": rel,
                "active_columns": int(np.count_nonzero(active)),
                "gp_steps_accepted": accepted,
            }
        )
        if rel <= tolerance:
            break

    elapsed = time.perf_counter() - start
    value = obj.value(f)
    gap, _ = fw_gap(f, obj.gradient(f), instance.group_columns, instance.demands)
    return AlgorithmResult(
        name="fw_gp",
        flow=f,
        history=pd.DataFrame(rows),
        objective=value,
        relative_gap=relative_gap(gap, value),
        feasibility_residual=feasible_residual(f, instance.group_columns, instance.demands),
        active_columns=int(np.count_nonzero(f > 1e-9)),
        pricing_calls=pricing_calls,
        wall_time_seconds=elapsed,
        metadata={"gp_sweeps": gp_sweeps},
    )


def solve_logit_sue_msa(
    instance: STBInstance,
    temperature: float = 1.5,
    max_iterations: int = 350,
    tolerance: float = 2e-6,
) -> AlgorithmResult:
    obj = STBObjective(instance)
    f = np.zeros(instance.n_columns, dtype=float)
    for demand, idx in zip(instance.demands, instance.group_columns, strict=True):
        f[idx] = demand / len(idx)
    rows: list[dict[str, float]] = []
    start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        grad = obj.gradient(f)
        target = np.zeros_like(f)
        for demand, idx in zip(instance.demands, instance.group_columns, strict=True):
            scores = -grad[idx] / temperature
            scores -= np.max(scores)
            probs = np.exp(scores)
            probs /= probs.sum()
            target[idx] = demand * probs
        step = 1.0 / iteration
        new_f = (1.0 - step) * f + step * target
        residual = float(np.linalg.norm(new_f - f, ord=1) / max(np.sum(instance.demands), 1.0))
        f = new_f
        value = obj.value_entropy(f, temperature)
        gap, _ = fw_gap(f, obj.gradient_entropy(f, temperature), instance.group_columns, instance.demands)
        rel = relative_gap(gap, value)
        rows.append(
            {
                "iteration": iteration,
                "entropy_objective": value,
                "relative_gap": rel,
                "fixed_point_residual": residual,
                "active_columns": int(np.count_nonzero(f > 1e-9)),
            }
        )
        if iteration >= 30 and residual <= tolerance:
            break

    elapsed = time.perf_counter() - start
    base_value = obj.value(f)
    entropy_gap, _ = fw_gap(f, obj.gradient_entropy(f, temperature), instance.group_columns, instance.demands)
    return AlgorithmResult(
        name="logit_sue_msa",
        flow=f,
        history=pd.DataFrame(rows),
        objective=base_value,
        relative_gap=relative_gap(entropy_gap, obj.value_entropy(f, temperature)),
        feasibility_residual=feasible_residual(f, instance.group_columns, instance.demands),
        active_columns=int(np.count_nonzero(f > 1e-9)),
        pricing_calls=0,
        wall_time_seconds=elapsed,
        metadata={"temperature": temperature, "gap_type": "entropy_regularized"},
    )
