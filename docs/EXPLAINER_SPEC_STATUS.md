# Explainer spec v1.0 — implementation status

Companion to `EXPLAINER_SPEC_v1.md`. Status after the 2026-07-22
increment. Legend: **live** (deployed at
asu-trans-ai-lab.github.io/TensorMobility/explainer/), **partial**,
**planned**.

## Global layout (spec §3)

| spec element | status | where / gap |
|---|---|---|
| §3.1 prompt & scenario bar | partial | scenario presets + action verbs exist (run/step/slow/scrub); natural-language prompt, audience selector, and depth slider planned |
| §3.2 concept-map navigation | partial | 13-module status strip on the explainer; modules 2/3/5/8/9/10/13 partially covered by home-page cards, docs, and the released suite; 7/11/12 planned |
| §3.3 center canvas | live | network map + solver iteration views; tensor view and traffic-state field planned |
| §3.4 right inspector (What/Why/Math/Compute/Verify) | **live** | five tabs for the scene and for any clicked road; every tab updates during the run |
| §3.5 computational timeline | partial | iteration scrubber + per-iteration ms + column-pool count; operator execution order and cache view planned |

## Interaction loop (spec §4)

| step | status | evidence |
|---|---|---|
| Predict | **live** | each preset opens with a prediction prompt |
| Manipulate | **live** | centralization, demand, β, subcenter clicks |
| Observe | **live** | linked map/readout/inspector updates each iteration |
| Derive | partial | live BPR and kernel equations with current numbers; step-by-step expansion planned |
| Verify | **live** | conservation ledger: Σ𝒟 = Σ_pℱ exactly; link-loading and node residuals ~1e-13 shown live; certificate gap |
| Transfer | partial | article links concepts to Chicago/TRMG2 scale; cross-network replay planned |

## Modules (spec §5, received through §5.6)

| module | status | notes |
|---|---|---|
| 5.1 System Overview | live | operator pipeline on the home page (clickable stages, forward/feedback animation); "break one interface" challenge planned |
| 5.2 Axis & Tensor Lab | partial | axis registry exists in `docs/TENSOR_AXES.md` + package `axes.py`; interactive cube/fiber/slice views planned |
| 5.3 Tensor Operator Lab | partial | continuous↔discrete kernel faces stated on stage click; interactive operation selector planned |
| 5.4 Conservation Lab | **live (core)** | the ledger (demand, link-loading, node flow) with live residuals; queue-storage and activity-time panels + challenge mode planned |
| 5.5 ABM Lab | partial | schedule-delay departure card (home); household/vehicle/tour timeline planned |
| 5.6 BN-DTA Lab | live (static face) | fixed-point iteration timeline, residual (gap) dashboard, MSA-stall lesson; time-space diagram, cumulative curves, queue-cell reverse trace planned |
| 5.7–5.13 | blocked | full spec text pending from the author (received text ends mid-§5.6) |

## Next build order (recommended)

1. §5.6 queue-cell reverse trace (click congested road → columns →
   ODs → zones — the data is already in the column pool).
2. §5.2 axis inspector with a "contract an axis" demonstration.
3. §3.1 depth slider (concept/notation/derivation) gating the Math
   tabs.
4. §5.4 challenge mode ("find the first broken conservation law").
5. ADMM Consensus Lab (§5.7) once its spec text arrives — the
   two-layer price-consensus benchmark already provides the certified
   backend numbers.
