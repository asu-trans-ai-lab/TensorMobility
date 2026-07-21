from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os as _os
import torch
# small scientific workloads: default to 1 thread; oversubscription on
# many-core CPUs stalls the grouped training loops (review P0.6)
torch.set_num_threads(int(_os.environ.get(
    'TENSORMOBILITY_TORCH_THREADS', '1')))
from torch import nn

from tensormobility.dta.algorithms import AlgorithmResult, fw_gap, initial_flow, relative_gap
from tensormobility.core.instance import STBInstance
from tensormobility.core.objective import STBObjective
from tensormobility.core.projections import feasible_residual, project_simplex


class RouterNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 48),
            nn.Tanh(),
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class RouterTrainingResult:
    model: RouterNet
    history: pd.DataFrame
    feature_names: list[str]


def _column_features(instance: STBInstance, f: np.ndarray) -> tuple[np.ndarray, list[str]]:
    obj = STBObjective(instance)
    resource_flow = obj.resource_flow(f)
    stress = resource_flow / instance.capacity
    route_time = instance.A.T @ obj.resource_time(f)
    ff = instance.free_flow_column_time
    cols = instance.columns

    features: list[list[float]] = []
    for p, row in cols.iterrows():
        used = np.flatnonzero(instance.A[:, p] > 0)
        mean_stress = float(np.mean(stress[used])) if used.size else 0.0
        max_stress = float(np.max(stress[used])) if used.size else 0.0
        demand = float(instance.demands[int(row["group_index"])])
        route_rank = int(row["route_rank"])
        features.append(
            [
                float(f[p] / max(demand, 1e-9)),
                float(instance.behavior_cost[p] / 40.0),
                float(ff[p] / 35.0),
                float(route_time[p] / 45.0),
                mean_stress,
                max_stress,
                float(row["mode"] == "car"),
                float(row["mode"] == "transit"),
                float(row["period"] == "AM"),
                float(row["period"] == "MD"),
                float(row["period"] == "PM"),
                float(row["destination"] == "cbd"),
                float(row["destination"] == "suburb"),
                float(row["preferred_destination_match"]),
                float(row["target_period_match"]),
                float(row["car_available"]),
                float(row["transit_pass"]),
                float(max(route_rank, -1) / 5.0),
            ]
        )
    names = [
        "flow_share",
        "behavior_cost",
        "free_flow_time",
        "current_route_time",
        "mean_path_stress",
        "max_path_stress",
        "is_car",
        "is_transit",
        "is_am",
        "is_md",
        "is_pm",
        "is_cbd",
        "is_suburb",
        "preferred_destination_match",
        "target_period_match",
        "car_available",
        "transit_pass",
        "route_rank",
    ]
    return np.asarray(features, dtype=np.float32), names


def train_router(
    instance: STBInstance,
    samples: int = 180,
    epochs: int = 30,
    seed: int = 19,
) -> RouterTrainingResult:
    """Train a small proposal model on random feasible states.

    Targets are continuous within-group quality scores derived from exact
    marginal costs. The model ranks columns; it does not certify optimality.
    """

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _ in range(samples):
        capacity_mult = rng.uniform(0.65, 1.25, size=instance.n_resources)
        scenario = instance.with_capacity_multiplier(capacity_mult)
        f = np.zeros(instance.n_columns)
        for demand, idx in zip(scenario.demands, scenario.group_columns, strict=True):
            probs = rng.dirichlet(np.full(len(idx), 0.55))
            f[idx] = demand * probs
        objective = STBObjective(scenario)
        grad = objective.gradient(f)
        features, names = _column_features(scenario, f)
        target = np.zeros(instance.n_columns, dtype=np.float32)
        for idx in scenario.group_columns:
            delta = grad[idx] - np.min(grad[idx])
            scale = max(float(np.std(grad[idx])), 1.0)
            target[idx] = np.exp(-delta / scale).astype(np.float32)
        xs.append(features)
        ys.append(target)

    x = torch.from_numpy(np.vstack(xs))
    y = torch.from_numpy(np.concatenate(ys))
    model = RouterNet(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = torch.sigmoid(model(x))
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            rows.append({"epoch": epoch, "mse": float(loss.detach())})

    return RouterTrainingResult(model=model.eval(), history=pd.DataFrame(rows), feature_names=names)


def solve_neural_certified_fw_gp(
    instance: STBInstance,
    router: RouterNet,
    top_k: int = 4,
    audit_interval: int = 6,
    max_iterations: int = 180,
    tolerance: float = 1e-7,
) -> AlgorithmResult:
    obj = STBObjective(instance)
    f = initial_flow(instance)
    active = f > 1e-12
    rows: list[dict[str, float]] = []
    start = time.perf_counter()
    proposal_evaluations = 0
    exact_audits = 0
    false_negative_audits = 0
    promotions = 0
    step = 0.5

    for iteration in range(1, max_iterations + 1):
        grad = obj.gradient(f)
        features, _ = _column_features(instance, f)
        with torch.no_grad():
            scores = torch.sigmoid(router(torch.from_numpy(features))).numpy()

        proposed_best: list[int] = []
        for idx in instance.group_columns:
            local_k = min(top_k, len(idx))
            local = idx[np.argpartition(scores[idx], -local_k)[-local_k:]]
            best = int(local[np.argmin(grad[local])])
            proposed_best.append(best)
            active[best] = True
            proposal_evaluations += local_k

        did_audit = iteration % audit_interval == 0 or iteration == max_iterations
        if did_audit:
            exact_audits += 1
            _, exact_best = fw_gap(f, grad, instance.group_columns, instance.demands)
            misses = 0
            for proposed, exact in zip(proposed_best, exact_best, strict=True):
                # Count only a materially worse proposal, not harmless ties.
                if float(grad[int(proposed)] - grad[int(exact)]) > 1e-5:
                    misses += 1
                if not active[int(exact)]:
                    active[int(exact)] = True
                    promotions += 1
            if misses > 0:
                false_negative_audits += 1

        current = obj.value(f)
        accepted = False
        used_step = step
        for _ in range(24):
            candidate = f.copy()
            for demand, idx in zip(instance.demands, instance.group_columns, strict=True):
                aidx = idx[active[idx]]
                candidate[idx] = 0.0
                candidate[aidx] = project_simplex(f[aidx] - used_step * grad[aidx], demand)
            direction = candidate - f
            new_value = obj.value(candidate)
            if new_value <= current + 1e-4 * float(np.dot(grad, direction)):
                f = candidate
                accepted = True
                break
            used_step *= 0.5
        step = min(max(used_step * (1.5 if accepted else 1.0), 1e-6), 2.0)
        active |= f > 1e-9

        value = obj.value(f)
        gap, exact_best = fw_gap(f, obj.gradient(f), instance.group_columns, instance.demands)
        rel = relative_gap(gap, value)
        rows.append(
            {
                "iteration": iteration,
                "objective": value,
                "relative_gap": rel,
                "active_columns": int(np.count_nonzero(active)),
                "exact_audit": int(did_audit),
                "promotions": promotions,
            }
        )
        # Final termination always uses the full-space certificate.
        if did_audit and rel <= tolerance:
            break

    elapsed = time.perf_counter() - start
    value = obj.value(f)
    gap, _ = fw_gap(f, obj.gradient(f), instance.group_columns, instance.demands)
    return AlgorithmResult(
        name="neural_certified_fw_gp",
        flow=f,
        history=pd.DataFrame(rows),
        objective=value,
        relative_gap=relative_gap(gap, value),
        feasibility_residual=feasible_residual(f, instance.group_columns, instance.demands),
        active_columns=int(np.count_nonzero(f > 1e-9)),
        pricing_calls=exact_audits * instance.n_groups,
        wall_time_seconds=elapsed,
        metadata={
            "top_k": top_k,
            "audit_interval": audit_interval,
            "proposal_evaluations": proposal_evaluations,
            "exact_audits": exact_audits,
            "false_negative_audits": false_negative_audits,
            "promotions": promotions,
        },
    )
