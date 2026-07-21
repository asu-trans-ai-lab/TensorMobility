# Graphical-abstract alignment (STB-FTT: Multi-Axis Flow-Through
# Tensors for Integrated Activity–DTA)

*2026-07-21. Panel-by-panel mapping of the poster to repository
artifacts, plus suggested improvements. The poster's numbers are this
repository's certified outputs — alignment is by construction; the
gaps below are presentation-level.*

## Panel → artifact mapping

| poster panel | repository artifact | status |
|---|---|---|
| 1. Master equation $x = B_\theta(L(x))$ | `engines/master_loop.py` — `MasterFixedPoint(behavior, loading, engine)`; special cases as operator switches (**3 tests**) | **aligned (new)** |
| 2. Multi-axis tensor $\Pi_{o,d,m,a,t,p}$, $F_{l,t}$ | `core/stb_tensor.py` (named axes, contraction, spectator lift); `docs/TENSOR_AXES.md` | aligned |
| 3. Sioux Falls 24/76 | `data/sioux_falls_tcglite` + unified loader | aligned |
| 4. SC1 logit SUE ↔ UE, gap $1.09\times10^{-5}$ | `tests/test_v06_closure.py` — the same number is the closure-test plateau | aligned |
| 5. SC2 fluid queue, base vs capacity shock | `dynamics/fluid_queue` + harness case-2/4 CSVs (peak 8,906→11,330 veh) | aligned |
| 6. SC3 tensor cascade, residuals $4.55/9.09\times10^{-13}$ | `dta/special_cases.solve_case_3` + `neural/tcg_graph` | aligned |
| 7. "9 validation tests" | now **58** in one suite | update poster |
| 8. Information pipeline (6 stages) | `workflow.yml → pipelines.fixed_point` (named below) | **aligned (new)** |
| 9. Axes table | `core/axes.py` registry (+ statuses/semirings the poster omits) | aligned |

Pipeline stage names registered in `workflow.yml`:
`behavior_choice → person_flow → occupancy_conversion → path_to_link →
dnl_queue → cost_update` (iterate to fixed point).

## Suggested improvements to the poster (ranked)

1. **Panel 1 needs the validity domain, one line.** The reviewer
   critique that reshaped this program applies verbatim to the poster:
   $x=B_\theta(L(x))$ is asserted without well-posedness. Add:
   *"$\theta$ flow-independent ⇒ monotone inner map ⇒ fixed point
   exists and certificates below are valid."* Cheap sentence, blocks
   the most damaging referee question.
2. **Add the pool-audit number to Panel 4.** The single strongest
   result in the program is missing: the fixed path pool is exact at
   free flow ($2\times10^{-16}$) and **35.7% inadequate under
   congestion** — it is the reason column generation exists, and it
   turns Panel 4 from "we recover known models" into "we measure what
   static models silently get wrong."
3. **Show the $\tau$ ladder, not one number.** Panel 4's table has a
   single $1.09\times10^{-5}$ standing for both distance and gap. The
   certified ladder ($6.9\times10^{-2}\to8.3\times10^{-6}$ across
   $\theta=0.1\to50$) is more convincing and visually natural (a
   descending curve already exists in the panel).
4. **Name the engine, or the loop is hand-waving.** Panel 8's "iterate
   to FP" hides the hardest computational lesson (the stiff
   choice↔wait cycling; the escalation ladder). One footnote:
   *"fixed points computed by an escalating engine ladder (damped →
   Anderson → block-Newton), cycle-detected."*
5. **One ribbon for the neural identity.** $B_\theta$ carries the
   subscript but the poster never says the founding sentence: *softmax
   router at temperature $1/\theta$ ≡ logit choice; expert path ≡
   behavioral column; backprop ≡ transposed FTT chain.* Without it the
   poster reads as pure DTA — the exact critique this repo just
   answered. A single bottom ribbon fixes it.
6. **Measures need color-coding in Panel 2.** $\Pi$ (probability,
   row-stochastic), $X$ (persons), $F$ (vehicles) with $\Omega_m$
   conversion — Panel 8 has the distinction, Panel 2 loses it; one
   legend line unifies them.
7. **Update counts and jargon**: "9 tests" → 58 (or "certificate suite,
   CI-run"); Panel 6 "HFN / BTCG" → "TCG (transportation computational
   graph)" for consistency with the paper lineage.
8. **Missing-axes footnote.** The tensor shows $(o,d,m,a,t,p)$; the
   grand framework carries group $g$, regime $s$, and the supply-layer
   axes (company, vtype, stage, matching). If the poster's scope is
   deliberately Activity–DTA, one footnote — *"extension axes
   (heterogeneity, regimes, service layers) enter as operator profiles
   without changing the framework"* — turns an omission into a claim.
