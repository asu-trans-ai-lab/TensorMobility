"""The neural <-> transportation correspondence, executable.

Not analogy -- identity. Each pair below is one computation with two
names, and the tests assert bitwise/numerical equality:

  softmax router @ temperature 1/theta   ==  multinomial logit choice
  router score - capacity duals          ==  negative reduced cost
  top-1 routing (theta -> inf)           ==  all-or-nothing assignment
  admission threshold (quantile dual)    ==  complementary slackness

This module is the executable half of docs/TENSOR_AXES.md.
"""
from __future__ import annotations

import numpy as np


def softmax_router(scores: np.ndarray, temperature: float = 1.0
                   ) -> np.ndarray:
    """MoE gate: p_k = exp(z_k/T) / sum exp(z_j/T)."""
    z = np.asarray(scores, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def logit_choice(costs: np.ndarray, theta: float = 1.0) -> np.ndarray:
    """Multinomial logit: p_k = exp(-theta c_k) / sum exp(-theta c_j)."""
    c = np.asarray(costs, dtype=float)
    z = -theta * (c - c.min())
    e = np.exp(z)
    return e / e.sum()


def router_is_logit(costs: np.ndarray, theta: float) -> bool:
    """The founding identity: a softmax router with scores z = -c and
    temperature 1/theta IS logit choice at scale theta."""
    a = softmax_router(-np.asarray(costs, float), temperature=1.0 / theta)
    b = logit_choice(costs, theta)
    return bool(np.allclose(a, b, rtol=0, atol=1e-15))


def routed_reduced_cost(cost: np.ndarray, incidence: np.ndarray,
                        duals: np.ndarray) -> np.ndarray:
    """Generalized reduced cost c_p + A_p' lambda: the negative of the
    dual-conditioned router score. Admission rule
    (score > threshold) == (reduced cost < 0) == column pricing."""
    return np.asarray(cost, float) + incidence.T @ np.asarray(duals,
                                                              float)


def hard_routing_limit(costs: np.ndarray, thetas=(1., 10., 100., 1000.)
                       ) -> np.ndarray:
    """theta -> inf: logit shares converge to all-or-nothing (the
    zero-temperature limit shared by top-1 MoE routing and AON
    assignment). Returns the share of the argmin option per theta."""
    j = int(np.argmin(costs))
    return np.array([logit_choice(costs, th)[j] for th in thetas])


def admission_complementarity(loads: np.ndarray, capacity: float,
                              dual: float, tol: float = 1e-9) -> bool:
    """BASE/Expert-Choice quantile thresholds are capacity multipliers:
    dual > 0 requires binding capacity (complementary slackness)."""
    slack = capacity - float(np.sum(loads))
    return (dual <= tol) or (abs(slack) <= tol)
