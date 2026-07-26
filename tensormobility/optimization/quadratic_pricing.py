from __future__ import annotations

"""Joint second-order (finite-move) pricing over integrated
activity-tour-path columns.

The integrated convex objective over complete-column flow f is

    F(f; U) = a(U)^T f + Phi_N(Delta f) + Phi_M(E f) + Phi_U(Gamma f; U)

with U the measured land-use / activity-opportunity supply, Delta the
arc-column incidence, E the activity-tour incidence, and Gamma the
grid-activity-state (mobility tensor) incidence.  Gradient and Hessian
decompose along the same three projections:

    g = a(U) + Delta^T grad Phi_N + E^T grad Phi_M + Gamma^T grad Phi_U
    H = Delta^T D_N Delta + E^T D_M E + Gamma^T D_U Gamma

so column interaction has exactly three sources: shared congested arcs,
shared activity-tour alternatives, and shared land-use opportunities.

For a feasible replacement of donor column rho by candidate column pi,

    d = e_pi - e_rho,
    Q_{pi<-rho}(alpha) = alpha g^T d + (alpha^2 / 2) d^T H d,

and under a bounded third derivative (|F'''| <= M3 along the segment)

    |DeltaF_exact(alpha) - Q_{pi<-rho}(alpha)| <= (M3 / 6) alpha^3 ||d||^3.

The exact finite change is always available through
`exact_finite_change`, so every second-order price can be verified
against the objective it approximates -- the pricing analogue of this
repository's certificate discipline.

Orientation follows the canonical rule (docs/ORIENTATION.md): every
operator maps target <- source, so Delta, E, Gamma all carry the
complete-column axis as their source (columns of the matrix) and
`M.T @ (...)` is the one adjoint appearing in the gradient.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse


def _validate_operator(name: str, M, n_columns: int) -> None:
    if M.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got ndim={M.ndim}")
    if M.shape[1] != n_columns:
        raise ValueError(
            f"{name} has {M.shape[1]} columns; expected {n_columns} "
            f"(the complete-column axis is the source axis)"
        )


def _apply_component_gradient(
    name: str, M, f: np.ndarray, component_gradient: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    projected = np.asarray(M @ f).ravel()
    grad = np.asarray(component_gradient(projected), dtype=float).ravel()
    if grad.shape != projected.shape:
        raise ValueError(
            f"{name} gradient returned shape {grad.shape}; expected {projected.shape}"
        )
    if not np.all(np.isfinite(grad)):
        raise ValueError(f"{name} gradient contains non-finite values")
    return np.asarray(M.T @ grad).ravel()


def integrated_gradient(
    f: np.ndarray,
    base_cost: np.ndarray,
    Delta,
    E,
    Gamma,
    network_gradient: Callable[[np.ndarray], np.ndarray],
    tour_gradient: Callable[[np.ndarray], np.ndarray],
    opportunity_gradient: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """g = a(U) + Delta^T grad Phi_N + E^T grad Phi_M + Gamma^T grad Phi_U.

    f: complete-column flow (n_columns,)
    base_cost: linear column cost a(U), already evaluated at the
        measured opportunity supply U
    Delta, E, Gamma: dense or scipy.sparse operators with the column
        axis as source (target <- column orientation)
    network_gradient(x): marginal congestion cost per arc at arc flow x
    tour_gradient(y): marginal tour-choice cost at tour flow y
    opportunity_gradient(X): marginal opportunity-utilization cost at
        the flattened mobility tensor X
    """
    f = np.asarray(f, dtype=float).ravel()
    a = np.asarray(base_cost, dtype=float).ravel()
    if a.shape != f.shape:
        raise ValueError(f"base_cost shape {a.shape} does not match flow {f.shape}")
    for name, M in (("Delta", Delta), ("E", E), ("Gamma", Gamma)):
        _validate_operator(name, M, f.size)
    return (
        a
        + _apply_component_gradient("network", Delta, f, network_gradient)
        + _apply_component_gradient("tour", E, f, tour_gradient)
        + _apply_component_gradient("opportunity", Gamma, f, opportunity_gradient)
    )


def _validate_curvature(name: str, diag: np.ndarray, n_rows: int) -> np.ndarray:
    d = np.asarray(diag, dtype=float).ravel()
    if d.shape != (n_rows,):
        raise ValueError(f"{name} curvature has shape {d.shape}; expected {(n_rows,)}")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{name} curvature contains non-finite values")
    if np.min(d, initial=0.0) < -1e-12:
        raise ValueError(
            f"{name} curvature has negative entries; each Phi must be convex "
            f"so its diagonal curvature is nonnegative"
        )
    return np.maximum(d, 0.0)


def integrated_hessian(
    Delta,
    E,
    Gamma,
    D_network: np.ndarray,
    D_tour: np.ndarray,
    D_opportunity: np.ndarray,
):
    """H = Delta^T D_N Delta + E^T D_M E + Gamma^T D_U Gamma.

    D_network, D_tour, D_opportunity are the diagonal curvatures of
    Phi_N, Phi_M, Phi_U evaluated at the current projections (one entry
    per arc, per tour alternative, per tensor state).  All three must
    be nonnegative -- the objective is a sum of convex separable
    potentials -- which makes H symmetric positive semidefinite by
    construction.

    Returns a scipy CSR matrix when any operator is sparse, otherwise a
    dense symmetric ndarray.
    """
    n_columns = Delta.shape[1]
    for name, M in (("Delta", Delta), ("E", E), ("Gamma", Gamma)):
        _validate_operator(name, M, n_columns)
    D_N = _validate_curvature("network", D_network, Delta.shape[0])
    D_M = _validate_curvature("tour", D_tour, E.shape[0])
    D_U = _validate_curvature("opportunity", D_opportunity, Gamma.shape[0])

    if any(sparse.issparse(M) for M in (Delta, E, Gamma)):
        terms = []
        for M, D in ((Delta, D_N), (E, D_M), (Gamma, D_U)):
            Ms = sparse.csr_matrix(M)
            terms.append(Ms.T @ sparse.diags(D) @ Ms)
        return (terms[0] + terms[1] + terms[2]).tocsr()

    return (
        np.asarray(Delta).T @ (D_N[:, None] * np.asarray(Delta))
        + np.asarray(E).T @ (D_M[:, None] * np.asarray(E))
        + np.asarray(Gamma).T @ (D_U[:, None] * np.asarray(Gamma))
    )


def replacement_direction(
    candidate_index: int, donor_index: int, n_columns: int
) -> np.ndarray:
    """d = e_candidate - e_donor: move one unit of flow from the donor
    column to the candidate column.

    Because both columns belong to the same traveler class, the move
    preserves class mass exactly: B d = 0 for any class-incidence B
    whose rows are indicator rows over columns.  Feasibility of the
    move (donor flow >= step) is the caller's responsibility.
    """
    candidate = int(candidate_index)
    donor = int(donor_index)
    if not (0 <= candidate < n_columns and 0 <= donor < n_columns):
        raise ValueError(
            f"column indices ({candidate}, {donor}) out of range [0, {n_columns})"
        )
    if candidate == donor:
        raise ValueError("candidate and donor columns must differ")
    d = np.zeros(int(n_columns), dtype=float)
    d[candidate] = 1.0
    d[donor] = -1.0
    return d


@dataclass(frozen=True)
class ReplacementPrice:
    """First- and second-order predictions of the objective change for
    the finite move f -> f + step * direction."""

    first_order: float          # step * g^T d
    curvature: float            # d^T H d
    second_order: float         # step * g^T d + (step^2 / 2) d^T H d
    step: float


def quadratic_price(
    gradient: np.ndarray,
    hessian,
    direction: np.ndarray,
    step: float = 1.0,
) -> ReplacementPrice:
    """Price a finite column move with the joint quadratic model.

    Q(step) = step * g^T d + (step^2 / 2) d^T H d.  The Hessian may be
    dense or sparse; only the matrix-vector product H d is formed, so
    pricing one move never materializes column-pair interactions beyond
    the moved columns' rows.
    """
    g = np.asarray(gradient, dtype=float).ravel()
    d = np.asarray(direction, dtype=float).ravel()
    if g.shape != d.shape:
        raise ValueError(f"gradient shape {g.shape} does not match direction {d.shape}")
    Hd = np.asarray(hessian @ d).ravel()
    if Hd.shape != d.shape:
        raise ValueError(f"hessian product shape {Hd.shape} does not match {d.shape}")
    linear = float(step) * float(g @ d)
    curvature = float(d @ Hd)
    return ReplacementPrice(
        first_order=linear,
        curvature=curvature,
        second_order=linear + 0.5 * float(step) ** 2 * curvature,
        step=float(step),
    )


def exact_finite_change(
    objective: Callable[[np.ndarray], float],
    f: np.ndarray,
    direction: np.ndarray,
    step: float = 1.0,
) -> float:
    """DeltaF_exact(step) = F(f + step * d) - F(f).

    The verification quantity: every quadratic price should be compared
    against this exact finite change, never only against another
    approximation.
    """
    f = np.asarray(f, dtype=float).ravel()
    d = np.asarray(direction, dtype=float).ravel()
    if f.shape != d.shape:
        raise ValueError(f"flow shape {f.shape} does not match direction {d.shape}")
    return float(objective(f + float(step) * d)) - float(objective(f))


def third_order_error_bound(
    m3: float, direction: np.ndarray, step: float = 1.0
) -> float:
    """(M3 / 6) * step^3 * ||d||^3, the Taylor remainder bound on
    |DeltaF_exact - second_order| when the third directional derivative
    of F is bounded by M3 along the move segment (Euclidean norm)."""
    if m3 < 0:
        raise ValueError("m3 must be nonnegative")
    d = np.asarray(direction, dtype=float).ravel()
    return float(m3) / 6.0 * abs(float(step)) ** 3 * float(np.linalg.norm(d)) ** 3
