"""Joint second-order pricing: the calculus is verified against finite
differences and exact finite moves, never against itself.

Covers: analytical gradient and Hessian vs central differences,
positive semidefiniteness under nonnegative curvature, second-order vs
first-order prediction quality, class-mass conservation of replacement
moves, the cubic decay of the second-order remainder, and the response
of the equilibrium to a measured opportunity perturbation.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensormobility.optimization.mobility_tensor_case import (
    MobilityTensorCase,
    Scenario,
)
from tensormobility.optimization.quadratic_pricing import (
    exact_finite_change,
    integrated_hessian,
    quadratic_price,
    replacement_direction,
    third_order_error_bound,
)


@pytest.fixture(scope="module")
def case() -> MobilityTensorCase:
    return MobilityTensorCase()


@pytest.fixture(scope="module")
def base() -> Scenario:
    return Scenario(name="base")


@pytest.fixture(scope="module")
def reference(case: MobilityTensorCase) -> np.ndarray:
    return case.feasible_interior_flow()


def test_gradient_matches_finite_differences(case, base, reference):
    g = case.gradient(reference, base)
    eps = 1e-5 * max(float(np.max(reference)), 1.0)
    fd = np.zeros_like(g)
    for k in range(case.n_columns):
        up = reference.copy()
        down = reference.copy()
        up[k] += eps
        down[k] -= eps
        fd[k] = (case.objective(up, base) - case.objective(down, base)) / (2 * eps)
    scale = max(float(np.max(np.abs(fd))), 1.0)
    assert float(np.max(np.abs(g - fd))) < 1e-6 * scale


def test_hessian_matches_finite_differences(case, base, reference):
    H = case.hessian(reference, base)
    rng = np.random.default_rng(3)
    eps = 1e-5 * max(float(np.max(reference)), 1.0)
    for _ in range(5):
        d = rng.normal(size=case.n_columns)
        d /= np.linalg.norm(d)
        gu = case.gradient(reference + eps * d, base)
        gd = case.gradient(reference - eps * d, base)
        fd = (gu - gd) / (2 * eps)
        Hd = H @ d
        scale = max(float(np.max(np.abs(fd))), 1.0)
        assert float(np.max(np.abs(Hd - fd))) < 1e-5 * scale


def test_hessian_positive_semidefinite(case, base, reference):
    eigenvalues = case.hessian_eigenvalues(reference, base)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    assert float(np.min(eigenvalues)) > -1e-10 * scale


def test_negative_curvature_is_rejected():
    ident = np.eye(3)
    with pytest.raises(ValueError, match="curvature"):
        integrated_hessian(
            ident, ident, ident,
            np.array([1.0, -0.5, 1.0]), np.ones(3), np.ones(3),
        )


def test_replacement_direction_validation():
    with pytest.raises(ValueError):
        replacement_direction(2, 2, 5)
    with pytest.raises(ValueError):
        replacement_direction(7, 0, 5)


def test_replacement_preserves_class_mass(case, base, reference):
    B = case.operators.B.matrix
    rng = np.random.default_rng(11)
    for _ in range(10):
        h = str(rng.choice(sorted(case.origins)))
        idx = case.class_columns(h)
        donor = int(rng.choice(idx))
        candidate = int(rng.choice(idx[idx != donor]))
        d = replacement_direction(candidate, donor, case.n_columns)
        assert float(np.max(np.abs(np.asarray(B @ d).ravel()))) == 0.0
        moved = reference + 10.0 * d
        class_totals = np.asarray(B @ moved).ravel()
        original = np.asarray(B @ reference).ravel()
        assert float(np.max(np.abs(class_totals - original))) < 1e-12


def test_second_order_beats_first_order(case, base):
    _, summary = case.pricing_experiment(base, samples=80)
    mae = summary.set_index("metric")
    first = float(mae.loc["MAE", "first_order"])
    second = float(mae.loc["MAE", "second_order"])
    assert second < 0.05 * first
    assert float(mae.loc["Spearman rank correlation", "second_order"]) > 0.999


def test_second_order_remainder_decays_cubically(case, base, reference):
    """|DeltaF_exact - Q| = O(alpha^3): doubling the step should scale
    the median remainder by ~8 across sampled in-class moves."""
    g = case.gradient(reference, base)
    H = case.hessian(reference, base)
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(20):
        h = str(rng.choice(sorted(case.origins)))
        idx = case.class_columns(h)
        donor = int(rng.choice(idx))
        candidate = int(rng.choice(idx[idx != donor]))
        d = replacement_direction(candidate, donor, case.n_columns)
        errors = []
        for step in (5.0, 10.0):
            price = quadratic_price(g, H, d, step=step)
            exact = exact_finite_change(
                lambda f: case.objective(f, base), reference, d, step=step
            )
            errors.append(abs(exact - price.second_order))
        if errors[0] > 1e-9:
            ratios.append(errors[1] / errors[0])
    assert len(ratios) >= 10
    assert 6.0 < float(np.median(ratios)) < 10.0


def test_third_order_bound_arithmetic():
    d = np.array([1.0, -1.0])
    norm3 = float(np.linalg.norm(d)) ** 3
    assert third_order_error_bound(6.0, d, step=2.0) == pytest.approx(8.0 * norm3)
    with pytest.raises(ValueError):
        third_order_error_bound(-1.0, d)


def test_opportunity_perturbation_shifts_destination_share(case, base):
    shifted = Scenario(
        name="employment shifted to W2",
        work_w1_supply=600.0,
        work_w2_supply=1400.0,
        work_w1_capacity=400.0,
        work_w2_capacity=800.0,
    )
    base_result = case.solve(base)
    shifted_result = case.solve(shifted, initial_flow=base_result.x)
    base_metrics = case.scenario_metrics(base, base_result)
    shifted_metrics = case.scenario_metrics(shifted, shifted_result)
    assert shifted_metrics["W2_share"] > base_metrics["W2_share"] + 0.05
    for certificate in case.certificates(shifted_result.x, shifted):
        assert certificate["passed"], certificate
