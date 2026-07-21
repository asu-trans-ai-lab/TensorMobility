# STB-FTT v0.5 — Streamlined multi-axis tensor design

> **STATUS UPDATE (author decisions, 2026-07-20):** simpler-first
> approved and EXECUTED. F4 state axis deferred to v0.6; F6 map stays
> 2-D; merge performed — this folder is now the integrated codebase
> (v0.4 spine + Passenger_Vehicle v0.5.1 superset + sparse Chicago
> solvers + axis registry + unified grid/SF/Chicago loaders), **34/34
> tests**. See README.md. The feature table below is kept as the
> design record.

*2026-07-20. Sources: TR-Part B and TR-Part C literature atlases
(`structure_is_all_you_need_mapping/`), the structure-layer taxonomy
(S_repr / S_stat / S_mech / S_ctrl + Demand/Supply/Observation), the
v0.4 grid design, and every certified asset in this workspace (SC1–SC3,
Chicago sparse CG, multi-layer LP+BPR price benchmarks). Items marked
**[REVIEW]** need your decision; items marked **[DONE]** are already
implemented and tested in this folder (4/4 tests).*

## 1. The streamline: three declarations per axis, nothing else

The literature scan says the field's methods separate cleanly by *what
they do to an index*, not by application domain. So the multi-axis
design is reduced to: every axis of
$\mathcal F_{o,d,g,m,\tau,p,a,t}\times\text{layers}$ declares exactly

| declaration | values | meaning |
|---|---|---|
| **status** | SPECTATOR / CONTRACTED / SYNCHRONIZED | rides along (Kronecker) / summed by a typed operator / pinned by a fixed point or dual |
| **semiring** | $(+,\times)$ / $(\min,+)$ / $(\max,\times)$ | flow aggregation / pricing / state decode |
| **anchor** | S_repr / S_stat / S_mech / S_ctrl / Observation | which literature layer owns its methods |

Everything else follows mechanically:
- a **model slice** = a set of promotions SPECTATOR→SYNCHRONIZED
  (static UE promotes `link`; DTA adds `departure`; multi-layer adds
  `layer`; state estimation adds `state` under $(\max,\times)$);
- **compression** (atoms/low-rank, S_stat) is admissible only across
  SPECTATOR axes, and only where columns-per-group is large (Chicago
  K̄≈2.2 counterexample is the guardrail);
- **learning** is admissible only if flow-independent along every
  SYNCHRONIZED axis (clean-VI boundary, POSITION file);
- each SYNCHRONIZED axis owes ONE well-posedness argument and ONE
  certificate class (monotonicity→FW/KKT; causality→conservation;
  duality→consensus gap; integrality→report the gap).

**[DONE]** `axes.py` implements the registry, the promotion chains, and
the clean-VI admissibility check as code + tests.

## 2. Feature table (traced to Part B / Part C clusters)

| # | suggested feature | literature trace (Part B/C cluster) | source asset | status |
|---|---|---|---|---|
| F1 | one canonical GMNS spine: grid + Sioux Falls + Chicago through `StaticNetwork` | 系统集成 stage (B); network optimization (C) | v0.4 `gmns_adapter`/`network_core` + our loaders | **[DONE]** `unified_networks.py`: v0.4 CG solver runs unchanged on all three (SF gap 1e-4, Chicago-300 gap 5e-4) |
| F2 | axis registry with status/semiring/anchor | the S_repr/S_stat/S_mech/S_ctrl split itself | tensor guide + this design | **[DONE]** `axes.py` |
| F3 | layer axis + price consensus (passenger/freight/operator blocks) | 多模式协同, 系统优化 (B); 路径协同优化 (C) | STB_multilayer_benchmark (LP duals 3.3e-4; BPR BCD 5.3e-7) | certified, to port |
| F4 | state/regime axis with $(\max,\times)$ decode hook (Viterbi over FD phases) | 交通状态估计 — largest Part-C cluster; 状态估计 (B) | tensor guide §6–7; QVDF phase work | **[REVIEW]** add as v0.5 module or defer to v0.6? |
| F5 | bounded learned residual (tanh, flow-independent), synthetic recovery gate G0 | 物理信息ML/PIML, 可解释AI (C); 机器学习辅助 (B) | v0.4 `bounded_learning` + POSITION G0 | port + wire G0 thresholds |
| F6 | well-posedness continuation map λ_B × λ_Q **× λ_C** (coupling integrality axis added) | 网络韧性, 不确定性量化 (C) | v0.4 `well_posedness` + activity-VRP integer corner | extend **[REVIEW]**: is λ_C in scope for the paper? |
| F7 | certificate gates as permanent tests: analytical anchors + SC1–SC3 + full-space Dijkstra audit | 解释性/鲁棒性 axes (C) | v0.4 `analytical_cases`; STB_special_cases_SF; chicago.py | port audit into v0.4 latent/algorithms (its gaps are still pool-restricted) |
| F8 | compression decision rule: atoms only where K̄ large; NEVER default | 低秩/张量 (S_stat) | Chicago crossover result | doctrine — write into DESIGN_PRINCIPLES |
| F9 | capacity-shock / resilience scenario axis with reoptimization deltas | 网络韧性优化, 应急疏散 (C) | v0.4 grid shock case + SC1 35.7% pool-degradation | have pieces; unify metric names |
| F10 | external DNL exchange (DTALite/DLSim contract) | 系统集成 (B); industry §6 of assessment | v0.4 `create_external_dnl_exchange` | keep as-is |

Explicitly **rejected** for v0.5 (with reason, per the atlases' hype
clusters): unrestricted transformer/GNN routing (fails F5's boundary and
the identifiability critique); RL signal control (out of scope — S_ctrl
methods enter only through prices); dense multi-dimensional tensor
storage (v0.3 Case-3 597x table).

## 3. What was integrated now (kept minimal pending your review)

```
stb_ftt_unified_v0.5/
├── axes.py               # F2: the streamlined registry (executable)
├── unified_networks.py   # F1: grid + SF + Chicago -> StaticNetwork
├── tests/test_v05.py     # 4/4: contracts, all 3 networks, v0.4 CG
│                         #      on SF + Chicago subset, full 93,135-OD load
└── DESIGN_v0_5_REVIEW.md # this file
```

Canonicalization rules made explicit: Chicago's 774 zero-fftt
connectors clamped to 1e-3 min; SF free-flow time = 60·length/speed;
TCGlite 126-path pool and Chicago `ref_volume` carried in `extras` so
the SC1/Chicago certificates remain reproducible through the unified
interface.

## 4. Decisions requested

1. **F4 scope**: include the $(\max,\times)$ state-decode axis in v0.5,
   or hold for v0.6 with the QVDF phase templates?
2. **F6 scope**: extend the well-posedness map with the coupling-
   integrality axis λ_C (needs the multilayer benchmark ported into the
   v0.4 harness), or keep the paper's map 2-D?
3. **Merge direction**: fold these v0.5 files INTO the v0.4 package
   (one repo, one test suite ≈ 25 tests), or keep v0.4 frozen as the
   external build and grow v0.5 as the overlay? My recommendation:
   fold in — v0.4's 21 tests + these 4 + ported SC/Chicago/multilayer
   gates give one spine (DESIGN_PRINCIPLES rule 1).
4. **Naming**: keep "STB-FTT" with the assessment's clarifying subtitle
   ("typed sparse flow-through operator network"), per its §3
   terminology warning?
