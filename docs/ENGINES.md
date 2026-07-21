# Equilibrium engines: from simple engines to complex extensions

*The higher-level position on the "stuck at 0.12" cycle (2026-07-20).
Executable counterpart: `stb_ftt/equilibrium_engines.py`; the MAGE
profile consumes it via `MAGEConfig.inner_engine`.*

## 1. The principle

The axis registry answers **which** indices are pinned by fixed points
(SYNCHRONIZED axes). It says nothing about **how** those fixed points
are computed. That is a separate, orthogonal design surface — the
**engine ladder** — and conflating the two is what turns a modeling
prototype into a pile of ad-hoc damping constants.

> A cycling iteration is not a tuning problem; it is a diagnosis.
> The remedy is chosen by the failure mode, and the chosen engine is
> part of the certificate.

## 2. Failure-mode taxonomy → remedy

| failure mode | symptom | remedy | engine |
|---|---|---|---|
| contraction holds | monotone residual decay | damped Picard | E0 |
| mild expansion / period-2 | alternating residual | MSA averaging | E1 |
| smooth but stiff composite | slow or cycling Picard, smooth map | Anderson extrapolation (memory m) | E2 |
| stiffness **concentrated** in a small subsystem | huge local Jacobian (MAGE: dw/dshare ≈ 10⁴ near saturation) | **solve the smallest stiffest block exactly** (Newton root on that block; Picard for the rest) | E3 |
| nonsmooth complementarity at scale | kinks, degenerate rationing, GNE coupling | semismooth Newton / PATH-class NCP-VI | E4 (declared v0.6) |

Cross-cutting remedies (never applied silently):

- **Smoothing a kink** (hard `min` → harmonic cap) buys differentiability
  at the price of changed physics — same trade-off as DNL smoothing
  (POSITION C2); it must be stated next to the operator it modifies.
- **Proximal anchoring** ("fix the solution to move forward"): when the
  operator is monotone but not contractive, iterate on the
  prox-regularized map $x_{k+1}=\arg\min F + \tfrac\rho2\|x-x_k\|^2$ —
  anchor at the incumbent, move only as far as the regularization
  allows. The ADMM/consensus benchmarks are this idea at block scale.
- **Tie-breaking**: set-valued responses (argmin ties, degenerate
  rationing) are regularized (entropy/prox) into single-valued maps
  before any engine is asked to find their fixed point.

## 3. The escalation policy

`solve_fixed_point(engine='auto')` runs the cheapest engine first; a
**cycle detector** (period-p state recurrence with non-vanishing step)
or a stall triggers promotion; every promotion is recorded in
`EngineResult.escalations`. Nothing is retried blindly and nothing is
hidden: the final report says *which engine closed the residual and
what failed before it*.

## 4. The MAGE instance, reread at this altitude

The observed history was the taxonomy playing out in order:

1. E0 damped Picard (damp 0.35 → 0.15): period-4 cycle at 0.12 — the
   choice↔wait subsystem has $|g'|\gg 1$.
2. Kink smoothing + E1-style slower fleet timescale: cycle persists —
   the stiffness is *structural* (M/M/1 near saturation), not a kink
   artifact.
3. E3: the stiff block is only $|K|\times|X|$-dimensional in the wait
   vector → exact Newton on that block, Picard (diagonalization) for
   congestion and fleet → **4 outer iterations, inner residual 4e-14**.

The design rule extracted: **identify the smallest subsystem that owns
the stiffness and solve it exactly; keep everything slow in cheap
fixed-point form.** This is the same two-queue/two-timescale insight
as the tensor guide's coordination-vs-synchronization split, applied to
solver construction.

## 5. Extension contract

A new operator profile (transit, freight tours, drones, pricing games)
must declare, per SYNCHRONIZED axis it adds:

1. the fixed-point map and its natural variable (choose the
   low-dimensional dual-like quantity — waits, prices, multipliers —
   not the high-dimensional primal shares);
2. the default engine and the ladder position to escalate to;
3. the certificate reported at the solution (residual + feasibility +
   complementarity + multi-start spread where uniqueness is unproven).

E4 (semismooth NCP/VI over the full stacked system) is the declared
next rung for MAGE-class GNEs and closes the gap to the paper's
GAMS/PATH formulation.
