"""Equilibrium engines: a formal escalation ladder for computing the
fixed points that pin SYNCHRONIZED axes.

The axis registry (axes.py) declares WHICH indices are pinned by a
fixed point; this module declares HOW that fixed point is computed and
WHEN to change method. The MAGE stiff choice<->wait cycle ("stuck at
0.12") is the motivating instance; the ladder generalizes it.

Failure-mode taxonomy -> remedy (the higher-level position):

  E0 damped Picard        contraction present            cheapest
  E1 adaptive/MSA Picard  mild expansion, period-2       averaging
  E2 Anderson (m-memory)  smooth but stiff composite map extrapolation
  E3 stiff-block Newton   stiffness CONCENTRATED in a    solve the
                          low-dimensional subsystem      smallest
                          (detected, not assumed)        stiffest block
                                                         exactly
  E4 semismooth NCP/VI    nonsmooth complementarity at   declared v0.6
     (PATH class)         scale                          extension

Cross-cutting remedies, applied per failure mode, never silently:
  - smoothing a kink (min/rationing) trades physics for convergence and
    must be stated (POSITION C2 discipline);
  - proximal anchoring ("fix the solution to move forward"):
    iterate on the prox-regularized map when the operator is monotone
    but not contractive;
  - tie-breaking: set-valued responses (argmin ties, degenerate
    rationing) are regularized (entropy/prox) to single-valued maps.

Escalation policy: run the cheapest engine; a CYCLE DETECTOR (state
recurrence over a window) or a stall detector triggers promotion to the
next engine; the engine that closes the residual is recorded in the
result — engine choice is part of the certificate, not a hidden detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root


@dataclass
class EngineResult:
    x: np.ndarray
    residual: float
    iterations: int
    engine: str
    converged: bool
    escalations: list = field(default_factory=list)
    cycle: dict | None = None
    history: list = field(default_factory=list)


def detect_cycle(states: list[np.ndarray], max_period: int = 6,
                 rtol: float = 1e-6) -> dict | None:
    """Detect a period-p recurrence in the iterate sequence: the state
    returns to itself after p steps while consecutive steps stay large
    (i.e., movement without progress)."""
    if len(states) < 2 * max_period + 2:
        return None
    x = states[-1]
    scale = max(float(np.abs(x).max()), 1e-12)
    step = float(np.abs(states[-1] - states[-2]).max())
    if step <= rtol * scale:
        return None                      # converging, not cycling
    for p in range(1, max_period + 1):
        if len(states) <= 2 * p + 1:
            continue
        rec = float(np.abs(states[-1] - states[-1 - p]).max())
        rec2 = float(np.abs(states[-1 - p] - states[-1 - 2 * p]).max())
        # a genuine cycle keeps moving at a NON-DECAYING step; slow
        # geometric convergence also recurs (period 1) but its steps
        # shrink every sweep -- do not flag that as a cycle
        prev_step = float(np.abs(states[-1 - p]
                                 - states[-2 - p]).max())
        if (rec <= 10 * rtol * scale and rec2 <= 10 * rtol * scale
                and step > 0.8 * prev_step):
            return dict(period=p, recurrence=rec, step=step)
    return None


def picard(g, x0, damp=0.5, msa=False, max_iter=500, tol=1e-10):
    """E0/E1: damped fixed-point iteration; msa=True uses 1/k averaging
    (kills period-2 oscillation for nonexpansive maps)."""
    x = np.asarray(x0, dtype=float).copy()
    states = [x.copy()]
    res = np.inf
    for it in range(1, max_iter + 1):
        gx = g(x)
        res = float(np.abs(gx - x).max())
        if res < tol:
            return EngineResult(x, res, it, 'msa' if msa else 'picard',
                                True, history=[res])
        step = (1.0 / it) if msa else damp
        x = (1.0 - step) * x + step * gx
        states.append(x.copy())
        cyc = detect_cycle(states)
        if cyc is not None:
            return EngineResult(x, res, it, 'msa' if msa else 'picard',
                                False, cycle=cyc, history=[res])
    return EngineResult(x, res, max_iter, 'msa' if msa else 'picard',
                        False, history=[res])


def anderson(g, x0, m=5, beta=1.0, max_iter=200, tol=1e-10,
             safeguard_damp=0.2):
    """E2: Anderson acceleration (type II) with a damped-Picard
    safeguard step whenever extrapolation increases the residual."""
    x = np.asarray(x0, dtype=float).copy()
    X, F = [], []                        # iterates and residuals
    res_prev = np.inf
    for it in range(1, max_iter + 1):
        gx = g(x)
        f = gx - x
        res = float(np.abs(f).max())
        if res < tol:
            return EngineResult(x, res, it, 'anderson', True)
        X.append(x.copy())
        F.append(f.copy())
        if len(X) > m:
            X.pop(0)
            F.pop(0)
        k = len(X)
        if k == 1:
            x_new = x + safeguard_damp * f
        else:
            dF = np.stack([F[i + 1] - F[i] for i in range(k - 1)], axis=1)
            dX = np.stack([X[i + 1] - X[i] for i in range(k - 1)], axis=1)
            gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
            x_new = x + beta * f - (dX + beta * dF) @ gamma
        # safeguard: never accept a step that worsens the residual a lot
        res_new = float(np.abs(g(x_new) - x_new).max())
        if not np.all(np.isfinite(x_new)) or res_new > 2.0 * res:
            x_new = x + safeguard_damp * f
        x = x_new
        res_prev = res
    res = float(np.abs(g(x) - x).max())
    return EngineResult(x, res, max_iter, 'anderson', res < tol)


def newton_block(g, x0, bounds=None, tol=1e-10):
    """E3: exact root solve of F(x) = x - g(x) — for the LOW-dimensional
    stiff block identified by diagnostics (design rule: solve the
    smallest stiffest block exactly; keep slow blocks in Picard)."""
    x0 = np.asarray(x0, dtype=float)

    def F(x):
        return x - g(x)

    sol = root(F, x0, method='hybr', options=dict(xtol=1e-13))
    x = sol.x
    if bounds is not None:
        x = np.clip(x, bounds[0], bounds[1])
    res = float(np.abs(F(x)).max())
    return EngineResult(x, res, int(sol.nfev), 'newton', res < tol)


def solve_fixed_point(g, x0, engine='auto', tol=1e-10, bounds=None,
                      **kw) -> EngineResult:
    """Front door. engine in {'picard','msa','anderson','newton','auto'}.
    'auto' walks the escalation ladder and records every promotion."""
    if engine == 'picard':
        return picard(g, x0, tol=tol, **kw)
    if engine == 'msa':
        return picard(g, x0, msa=True, tol=tol, **kw)
    if engine == 'anderson':
        return anderson(g, x0, tol=tol, **kw)
    if engine == 'newton':
        return newton_block(g, x0, bounds=bounds, tol=tol)
    if engine != 'auto':
        raise ValueError(f'unknown engine {engine!r}')

    escalations = []
    r = picard(g, x0, tol=tol, max_iter=200,
               damp=kw.get('damp', 0.5))
    if r.converged:
        r.escalations = escalations
        return r
    escalations.append(dict(engine='picard', residual=r.residual,
                            cycle=r.cycle))
    r = anderson(g, r.x, tol=tol)
    if r.converged:
        r.escalations = escalations
        return r
    escalations.append(dict(engine='anderson', residual=r.residual))
    r = newton_block(g, r.x, bounds=bounds, tol=tol)
    r.escalations = escalations
    return r
