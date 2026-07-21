from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tensormobility.dta.algorithms import AlgorithmResult, fw_gap, relative_gap
from tensormobility.core.instance import STBInstance
from tensormobility.core.objective import STBObjective
from tensormobility.core.projections import feasible_residual, project_simplex


@dataclass
class AtomRepresentation:
    D: np.ndarray  # original columns x atoms
    atom_group: np.ndarray
    labels: list[str]
    singleton_column: list[int | None]

    @property
    def n_atoms(self) -> int:
        return self.D.shape[1]

    def group_atoms(self, n_groups: int) -> list[np.ndarray]:
        return [np.flatnonzero(self.atom_group == g) for g in range(n_groups)]

    def append_singletons(self, columns: list[int], column_group: np.ndarray) -> "AtomRepresentation":
        existing = {c for c in self.singleton_column if c is not None}
        add = [int(c) for c in columns if int(c) not in existing]
        if not add:
            return self
        extra = np.zeros((self.D.shape[0], len(add)), dtype=float)
        for j, column in enumerate(add):
            extra[column, j] = 1.0
        return AtomRepresentation(
            D=np.column_stack([self.D, extra]),
            atom_group=np.concatenate([self.atom_group, column_group[add]]),
            labels=self.labels + [f"promoted:{c}" for c in add],
            singleton_column=self.singleton_column + add,
        )


def build_route_family_atoms(instance: STBInstance) -> AtomRepresentation:
    """Keep baseline-best car routes explicit and compress other routes."""

    p = instance.n_columns
    atom_vectors: list[np.ndarray] = []
    atom_group: list[int] = []
    labels: list[str] = []
    singleton: list[int | None] = []
    cols = instance.columns
    ff_col = ("total_free_flow_cost" if "total_free_flow_cost" in cols
              else "free_flow_path_time")
    ff = cols[ff_col].to_numpy(dtype=float)

    for g, idx in enumerate(instance.group_columns):
        frame = instance.columns.loc[idx]
        for (destination, mode, period), sub in frame.groupby(["destination", "mode", "period"], sort=False):
            cols = sub["column_index"].to_numpy(dtype=int)
            if mode == "transit" or len(cols) == 1:
                vector = np.zeros(p)
                vector[cols[0]] = 1.0
                atom_vectors.append(vector)
                atom_group.append(g)
                labels.append(f"g{g}:{destination}:{mode}:{period}:singleton")
                singleton.append(int(cols[0]))
                continue

            best = int(cols[np.argmin(ff[cols])])
            vector = np.zeros(p)
            vector[best] = 1.0
            atom_vectors.append(vector)
            atom_group.append(g)
            labels.append(f"g{g}:{destination}:{mode}:{period}:major")
            singleton.append(best)

            minor = [int(c) for c in cols if int(c) != best]
            vector = np.zeros(p)
            # A deliberately static decoder: all hidden route alternatives share mass.
            vector[minor] = 1.0 / len(minor)
            atom_vectors.append(vector)
            atom_group.append(g)
            labels.append(f"g{g}:{destination}:{mode}:{period}:minor_atom")
            singleton.append(None)

    D = np.stack(atom_vectors, axis=1)
    return AtomRepresentation(
        D=D,
        atom_group=np.asarray(atom_group, dtype=int),
        labels=labels,
        singleton_column=singleton,
    )


def _initialize_alpha(instance: STBInstance, rep: AtomRepresentation) -> np.ndarray:
    objective = STBObjective(instance)
    grad0 = objective.gradient(np.zeros(instance.n_columns))
    atom_grad = rep.D.T @ grad0
    alpha = np.zeros(rep.n_atoms, dtype=float)
    for g, demand in enumerate(instance.demands):
        idx = np.flatnonzero(rep.atom_group == g)
        alpha[idx[np.argmin(atom_grad[idx])]] = demand
    return alpha


def _atom_full_gap(
    instance: STBInstance,
    rep: AtomRepresentation,
    alpha: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    obj = STBObjective(instance)
    f = rep.D @ alpha
    grad = obj.gradient(f)
    full_gap_value, best_full = fw_gap(f, grad, instance.group_columns, instance.demands)
    atom_grad = rep.D.T @ grad
    best_atom = np.zeros(instance.n_groups, dtype=int)
    represented_gap = 0.0
    for g, demand in enumerate(instance.demands):
        aidx = np.flatnonzero(rep.atom_group == g)
        j = int(aidx[np.argmin(atom_grad[aidx])])
        best_atom[g] = j
        represented_gap += float(alpha[aidx] @ atom_grad[aidx] - demand * atom_grad[j])
    return represented_gap, full_gap_value, best_full, best_atom


def solve_latent_fw_gp(
    instance: STBInstance,
    representation: AtomRepresentation | None = None,
    adaptive: bool = False,
    max_cycles: int = 10,
    max_iterations_per_cycle: int = 120,
    tolerance: float = 1e-7,
    promotion_tolerance: float = 1e-5,
) -> tuple[AlgorithmResult, AtomRepresentation]:
    rep = representation or build_route_family_atoms(instance)
    obj = STBObjective(instance)
    start = time.perf_counter()
    rows: list[dict[str, float]] = []
    pricing_calls = 0
    promotions = 0
    alpha = _initialize_alpha(instance, rep)
    initial_atoms = rep.n_atoms

    for cycle in range(max_cycles):
        if len(alpha) != rep.n_atoms:
            old = alpha
            alpha = np.zeros(rep.n_atoms)
            alpha[: len(old)] = old
        group_atoms = rep.group_atoms(instance.n_groups)
        active = alpha > 1e-12
        step = 0.5

        for iteration in range(1, max_iterations_per_cycle + 1):
            f = rep.D @ alpha
            grad_f = obj.gradient(f)
            grad_a = rep.D.T @ grad_f
            for g, aidx in enumerate(group_atoms):
                best = int(aidx[np.argmin(grad_a[aidx])])
                active[best] = True
            pricing_calls += instance.n_groups

            current = obj.value(f)
            accepted = False
            used_step = step
            for _ in range(24):
                candidate = alpha.copy()
                for g, demand in enumerate(instance.demands):
                    aidx = group_atoms[g]
                    ridx = aidx[active[aidx]]
                    candidate[aidx] = 0.0
                    candidate[ridx] = project_simplex(alpha[ridx] - used_step * grad_a[ridx], demand)
                direction = candidate - alpha
                new_value = obj.value(rep.D @ candidate)
                if new_value <= current + 1e-4 * float(np.dot(grad_a, direction)):
                    alpha = candidate
                    accepted = True
                    break
                used_step *= 0.5
            step = min(max(used_step * (1.5 if accepted else 1.0), 1e-6), 2.0)

            represented_gap, full_gap_value, _, _ = _atom_full_gap(instance, rep, alpha)
            value = obj.value(rep.D @ alpha)
            rows.append(
                {
                    "cycle": cycle,
                    "iteration": iteration,
                    "objective": value,
                    "represented_relative_gap": relative_gap(represented_gap, value),
                    "full_relative_gap": relative_gap(full_gap_value, value),
                    "atoms": rep.n_atoms,
                    "active_atoms": int(np.count_nonzero(alpha > 1e-9)),
                    "promotions": promotions,
                }
            )
            if relative_gap(represented_gap, value) <= tolerance:
                break

        represented_gap, full_gap_value, best_full, best_atom = _atom_full_gap(instance, rep, alpha)
        value = obj.value(rep.D @ alpha)
        full_rel = relative_gap(full_gap_value, value)
        if not adaptive or full_rel <= tolerance:
            break

        grad = obj.gradient(rep.D @ alpha)
        atom_grad = rep.D.T @ grad
        candidates: list[int] = []
        for g, demand in enumerate(instance.demands):
            full_col = int(best_full[g])
            best_represented = int(best_atom[g])
            improvement = float(atom_grad[best_represented] - grad[full_col])
            if demand * improvement > promotion_tolerance * max(abs(value), 1.0):
                candidates.append(full_col)
        old_atoms = rep.n_atoms
        rep = rep.append_singletons(candidates, instance.column_group)
        new_atoms = rep.n_atoms - old_atoms
        if new_atoms == 0:
            break
        promotions += new_atoms

    elapsed = time.perf_counter() - start
    f = rep.D @ alpha
    value = obj.value(f)
    _, full_gap_value, _, _ = _atom_full_gap(instance, rep, alpha)
    name = "adaptive_latent_fw_gp" if adaptive else "static_latent_fw_gp"
    result = AlgorithmResult(
        name=name,
        flow=f,
        history=pd.DataFrame(rows),
        objective=value,
        relative_gap=relative_gap(full_gap_value, value),
        feasibility_residual=feasible_residual(f, instance.group_columns, instance.demands),
        active_columns=int(np.count_nonzero(f > 1e-9)),
        pricing_calls=pricing_calls,
        wall_time_seconds=elapsed,
        metadata={
            "initial_atoms": initial_atoms,
            "final_atoms": rep.n_atoms,
            "promotions": promotions,
            "compression_ratio": instance.n_columns / rep.n_atoms,
        },
    )
    return result, rep
