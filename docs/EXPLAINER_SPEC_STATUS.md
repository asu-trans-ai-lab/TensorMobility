# Explainer specs — implementation status

## Spec v3 (ABM–DTA Practice Studio) — `EXPLAINER_SPEC_v3_PRACTICE.md`

Deployed at `explainer/practice.html`, built on the author's v3 mockup
(design and eight views preserved) with the H017 fixture
(`docs/practice_scenario.json`) and every hardcoded value replaced by
live computation:

| view | computed content |
|---|---|
| Tours & schedule | tour residuals \|exp−sched\|; the T3→T4 activity residual max(0, 544−520)=24 min; conflict outlines derived, not asserted |
| Demand transfer | Vehicle Departure Ledger (§7.4) computed from trip records (9 person trips → 6 vehicle departures with the arithmetic shown); V1 overlap detection on scheduled AND experienced times; §7.5 certificates; a real two-step repair (P1→transit, then shift T4) that clears conflicts and demonstrates behavioral-response levels 3–4 |
| DTA network | logit route shares computed from each traveler's VoT and tolls; reroute-on-experienced toggling; stale-cost certificate |
| Dynamic skims | all cells computed from a time-dependent TT(τ); fixed vs sampled vs experienced methods disagree systematically near the 7:50 peak; §9.5 aggregation warning quantified |
| Feedback & convergence | fixture residual record (k=0..3) + a computed direct-vs-MSA cobweb (direct oscillates, α=0.3 converges) |
| Calibration | R² and MAPE computed from the observed/model table |

Pending from v3: en-route rerouting, transit/TNC vehicle formation
beyond the bus, warm/cold start comparison with real runs, export
manifests (§22 #15).

## Spec v2 (Visual Computing Studio) — `EXPLAINER_SPEC_v2_STUDIO.md`

Deployed at `explainer/studio.html`, built directly on the author's
interactive mockup (`tensormobility_visual_explorer_v2`), keeping its
visual design and the eight-axis color system
(`docs/studio_design_tokens.json`), with every view upgraded from
illustrative to computed live:

| chapter (spec §) | status | computed content |
|---|---|---|
| 1 coordinate/fiber/slice (§6) | live | real canonical tensor; click any of 16 cells → coordinate/fiber/slice/ledger update together |
| 2 unfold/fold (§7) | live | real 4×8 unfolding, fold-back verified entry-by-entry, low-rank transportation reading |
| 3 mode product (§8) | live | worked numbers + link to the dedicated Tensor+ADMM Lab |
| 4 contraction studio (§9) | live | predict-shape gate → contract (D=[[18,11],[9,5]], conservation 43=43) → term-by-term entry inspection |
| 5 block-flow canvas (§10) | live | 7 typed blocks; every edge click reveals its contract (contracts/preserves/generates); adjoint direction |
| 6 CP/Tucker (§11) | live | **real Jacobi SVD in-browser**: σ = 41.96/3.73/1.17/0.00 — the canonical tensor is exactly rank 3; rank slider drives the true representation residual; three-residual separation stated |
| 7 sparse (§12) | live | top-s support mask with kept-flow %, dropped-entry norm, and the warning that thresholding breaks conservation until projected |
| 8 ADMM studio (§13) | live | real engine (same certified math as the Lab), ρ slider, "just averaging?" refutation (17.567 vs 17.532) |
| 9 transformer bridge (§14) | live | **real 4-token masked attention** over the fiber with adjacency-mask toggle; softmax-normalization ≠ flow-conservation stated; non-equivalences listed |
| prompt controller (§16) | partial | keyword grounding to chapters; full locate/highlight/execute protocol pending |
| export, TRMG2 example (§21 ph.7) | pending | |

## Spec v1.0 — implementation status

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
| 5.7 ADMM Consensus (Tensor + ADMM Lab) | **live (MVP-1–3)** | dedicated code-ready spec received (`tensormobility_tensor_admm_offline_design`, archived as this repo's governing doc for module 7); implemented at `explainer/tensor-admm.html`: canonical 4-cell→2-zone example, editable B with column-sum checks, fiber/slice/unfolding views, axis-role stack, conservation ledger + break/diagnose chips, null-space information-loss lesson, per-coordinate ADMM (closed-form locals, projected consensus, duals, primal/dual residual chart, ρ slider, balanced/conflict scenarios, diagnostic labels incl. "converged but conservation broken"); MVP-4 prompt parser + instructor mode and MVP-5 TRMG2 super-zone example pending |
| 5.8–5.13 | blocked | full spec text of the master document pending from the author (received text ends mid-§5.6) |

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
