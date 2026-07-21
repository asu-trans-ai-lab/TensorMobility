"""The master fixed-point equation of the graphical abstract, as API:

    x* = B_theta( L(x*) )

B_theta : behavior operator  (utility/choice; probability measure Pi)
L       : loading operator   (congestion / DNL; experienced costs C, T)

"Models emerge from a single unified framework by switching/activating
operators (not programs)" -- this module makes that sentence executable:
compose any behavior operator with any loading operator and hand the
loop to the engine ladder. The special cases are operator collapses:

    logit SUE        B = logit(theta),   L = static BPR
    deterministic UE B = argmin (theta->inf via ladder), L = static BPR
    fluid queue      B frozen,           L = point-queue DNL
    activity-DTA     B = hierarchical chain, L = DNL   (coupled)

Validity domain (stated precisely, review P0.2): EXISTENCE follows
when the composition B(L(.)) is a continuous self-map of a nonempty
compact convex feasible set (Brouwer); flow-independent theta keeps the
inner cost map monotone but does NOT by itself make the composite
contractive. UNIQUENESS and iterative convergence require additional
contraction / strong-monotonicity / spectral conditions -- where
applicable report rho(dT/dx) < 1 locally. Static UE retains the FW
certificate only under separable, integrable, monotone costs. Nonsmooth
L (binding queues) forfeits derivative claims, not the fixed point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tensormobility.engines.equilibrium_engines import (EngineResult,
                                                        solve_fixed_point)


@dataclass
class MasterFixedPoint:
    """x = B(L(x)) with x the behavior/flow state (e.g. path shares)."""
    behavior: Callable[[np.ndarray], np.ndarray]   # costs -> state
    loading: Callable[[np.ndarray], np.ndarray]    # state -> costs
    engine: str = 'auto'

    def composite(self, x: np.ndarray) -> np.ndarray:
        return self.behavior(self.loading(x))

    def solve(self, x0: np.ndarray, tol: float = 1e-10,
              **kw) -> EngineResult:
        return solve_fixed_point(self.composite, np.asarray(x0, float),
                                 engine=self.engine, tol=tol, **kw)

    def self_consistency(self, x: np.ndarray) -> float:
        """Certificate: max |x - B(L(x))| at the returned point."""
        return float(np.abs(np.asarray(x) - self.composite(x)).max())
