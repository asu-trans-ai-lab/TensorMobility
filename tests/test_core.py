from __future__ import annotations

import numpy as np

from tensormobility.dta.algorithms import solve_exact_fw, solve_fw_gp
from tensormobility.core.instance import build_toy_instance, capacity_shock_multiplier
from tensormobility.dta.latent import build_route_family_atoms, solve_latent_fw_gp
from tensormobility.core.objective import STBObjective
from tensormobility.core.projections import feasible_residual


def test_instance_and_mass_conservation() -> None:
    instance = build_toy_instance()
    result = solve_fw_gp(instance, max_iterations=80, tolerance=1e-5)
    assert result.feasibility_residual < 1e-8
    assert np.min(result.flow) >= -1e-10
    assert abs(result.flow.sum() - instance.demands.sum()) < 1e-8


def test_objective_gradient_finite_difference() -> None:
    instance = build_toy_instance()
    objective = STBObjective(instance)
    rng = np.random.default_rng(4)
    f = np.zeros(instance.n_columns)
    direction = np.zeros(instance.n_columns)
    for demand, idx in zip(instance.demands, instance.group_columns, strict=True):
        probs = rng.dirichlet(np.ones(len(idx)))
        f[idx] = demand * probs
        raw = rng.normal(size=len(idx))
        raw -= raw.mean()
        direction[idx] = raw
    eps = 1e-5
    fd = (objective.value(f + eps * direction) - objective.value(f - eps * direction)) / (2 * eps)
    analytic = float(objective.gradient(f) @ direction)
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1.0)
    assert rel < 2e-5


def test_latent_chain_rule() -> None:
    instance = build_toy_instance()
    rep = build_route_family_atoms(instance)
    rng = np.random.default_rng(8)
    alpha = np.zeros(rep.n_atoms)
    for demand, aidx in zip(instance.demands, rep.group_atoms(instance.n_groups), strict=True):
        alpha[aidx] = demand * rng.dirichlet(np.ones(len(aidx)))
    f = rep.D @ alpha
    grad_f = STBObjective(instance).gradient(f)
    grad_alpha = rep.D.T @ grad_f
    direction = rng.normal(size=rep.n_atoms)
    for aidx in rep.group_atoms(instance.n_groups):
        direction[aidx] -= direction[aidx].mean()
    eps = 1e-5
    objective = STBObjective(instance)
    fd = (objective.value(rep.D @ (alpha + eps * direction)) - objective.value(rep.D @ (alpha - eps * direction))) / (2 * eps)
    analytic = float(grad_alpha @ direction)
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1.0)
    assert rel < 2e-5


def test_exact_fw_certificate() -> None:
    instance = build_toy_instance()
    result = solve_exact_fw(instance, max_iterations=500, tolerance=2e-6)
    # Classical FW has the expected O(1/k) tail; this test verifies a valid
    # full-space certificate rather than demanding GP-like local convergence.
    assert result.relative_gap < 2e-4
    assert result.feasibility_residual < 1e-8


def test_capacity_failure_and_adaptive_recovery() -> None:
    base = build_toy_instance()
    shock = base.with_capacity_multiplier(capacity_shock_multiplier(base))
    rep = build_route_family_atoms(base)
    static, _ = solve_latent_fw_gp(
        shock,
        representation=rep,
        adaptive=False,
        max_iterations_per_cycle=80,
        tolerance=1e-5,
    )
    adaptive, _ = solve_latent_fw_gp(
        shock,
        representation=rep,
        adaptive=True,
        max_iterations_per_cycle=80,
        tolerance=1e-5,
    )
    assert adaptive.relative_gap < static.relative_gap
    assert int(adaptive.metadata["promotions"]) > 0
    assert feasible_residual(adaptive.flow, shock.group_columns, shock.demands) < 1e-8
