"""Observation operators: sensors as linear functionals on the grid.

Certifies the adapter pattern adopted from the computational-graph
TSE codebase (Lu 2021): bilinear point operators reproduce bilinear
fields exactly; trapezoidal segment operators integrate linear fields
exactly; rows of point operators are convex weights.
"""
import numpy as np
import pytest

from tensormobility.adapters.observation_operators import (
    apply_operator, point_operator, segment_integral_operator)

GRID_X = np.linspace(0.0, 5000.0, 51)
GRID_T = np.linspace(0.0, 7200.0, 121)


def bilinear_field(a, b, c, d):
    X, T = np.meshgrid(GRID_X, GRID_T, indexing='ij')
    return a + b * X + c * T + d * X * T


def test_point_operator_exact_on_bilinear_fields():
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 5000, 200)
    ts = rng.uniform(0, 7200, 200)
    W = point_operator(xs, ts, GRID_X, GRID_T)
    f = bilinear_field(3.0, 2e-3, -1e-4, 5e-8)
    exact = 3.0 + 2e-3 * xs - 1e-4 * ts + 5e-8 * xs * ts
    np.testing.assert_allclose(apply_operator(W, f), exact, rtol=1e-12)


def test_point_operator_rows_are_convex_weights():
    W = point_operator([0.0, 2500.0, 5000.0], [0.0, 3600.0, 7200.0],
                       GRID_X, GRID_T)
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(),
                               1.0, rtol=1e-12)
    assert W.min() >= 0.0


def test_point_operator_clamps_out_of_range():
    W = point_operator([-10.0, 6000.0], [-5.0, 9000.0], GRID_X, GRID_T)
    f = bilinear_field(1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(apply_operator(W, f), 1.0, rtol=1e-12)


def test_segment_operator_integrates_linear_field_exactly():
    # density field linear in x at fixed t: k(x) = k0 + s*x
    k0, s = 0.05, 1e-5
    f = bilinear_field(k0, s, 0.0, 0.0)
    x0, x1 = 700.0, 3200.0
    W = segment_integral_operator([x0], [x1], [3600.0], GRID_X, GRID_T)
    exact = k0 * (x1 - x0) + s * (x1**2 - x0**2) / 2
    np.testing.assert_allclose(apply_operator(W, f), [exact],
                               rtol=1e-10)


def test_segment_operator_partial_cells():
    # segment strictly inside one grid cell
    f = bilinear_field(2.0, 0.0, 0.0, 0.0)
    W = segment_integral_operator([120.0], [180.0], [0.0], GRID_X,
                                  GRID_T)
    np.testing.assert_allclose(apply_operator(W, f), [2.0 * 60.0],
                               rtol=1e-10)


def test_time_interpolation_of_segment_counts():
    # field linear in t: integral at t halfway between grid times
    f = bilinear_field(1.0, 0.0, 1e-4, 0.0)
    t_mid = (GRID_T[10] + GRID_T[11]) / 2
    W = segment_integral_operator([0.0], [5000.0], [t_mid], GRID_X,
                                  GRID_T)
    exact = (1.0 + 1e-4 * t_mid) * 5000.0
    np.testing.assert_allclose(apply_operator(W, f), [exact],
                               rtol=1e-10)
