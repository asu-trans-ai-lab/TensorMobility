# TensorMobility Explainer — Product and Interface Specification v1.0

> Provided by the author 2026-07-22. NOTE: the received text ends
> mid-Section 5.6 (at the queue-cell trace "… → tours"); modules
> 5.7–5.13 (ADMM Consensus Lab, Tensor Compression and Scaling, PINN
> Traffic-State Lab, Neural Computational Connections,
> Regional-to-National Hierarchy, Model Integration Debugger,
> Experiment and Certification Center) are named in §3.2 but their
> full specifications are pending. Implementation status against this
> spec: `EXPLAINER_SPEC_STATUS.md`.

## 0. Product Definition

**Product name:** TensorMobility Explainer

**Purpose:** An interactive computational website for the
TensorMobility book, integrated paper, and classroom teaching. The
system should help users understand, manipulate, solve, debug, and
scale the mathematical and computational connections among land use,
activity-based modeling (ABM), multidimensional demand, dynamic
traffic assignment (DTA), traffic-state estimation, tensor methods,
ADMM, and modern neural computational operators.

**Primary design principle:**

> Do not visualize equations as decoration. Every visualization must
> expose a modeling decision, a conservation law, a numerical
> difficulty, a failure mode, or a scaling mechanism.

**Narrative arc:**

Represent → Conserve → Connect → Stabilize → Scale → Transfer

---

## 1. Design-Agent Review

The specification reflects six simulated expert reviews.

### 1.1 Transportation modeling agent
- explicit land-use–ABM–DTA lineage;
- household, person, activity, tour, trip, OD, mode, departure, path,
  and link-time states;
- scheduled versus experienced travel-time consistency;
- exact demand, path, node, link, and queue conservation;
- realistic interfaces to CT-RAMP, ActivitySim, GMNS, and TRMG2;
- reverse attribution from congestion back to behavior.

### 1.2 Applied mathematics agent
- named tensor spaces and measures;
- fibers, slices, contractions, matricization, adjoints, projections,
  and fixed points;
- clear distinction between discrete sums and continuous integral
  operators;
- feasible sets, invariants, and residual definitions;
- mathematical derivations that remain visible beside the graphics.

### 1.3 Numerical optimization agent
- solver state rather than only final output;
- ADMM primal and dual residuals;
- penalty-parameter sensitivity;
- convergence, oscillation, infeasibility, and representation-limited
  states;
- comparisons among MSA, Frank–Wolfe, gradient projection, column
  generation, ADMM, and Anderson acceleration;
- stopping criteria and numerical certificates.

### 1.4 Neural-computation agent
- computational-graph interpretation;
- forward and adjoint flows;
- PINN residual terms;
- neural-operator field mappings;
- constrained attention masks;
- explicit separation of structural transportation operators and
  learned corrections;
- no black-box replacement of conservation or equilibrium.

### 1.5 Visualization and HCI agent
- progressive disclosure;
- linked views;
- direct manipulation;
- coordinated highlighting across equations, tensors, networks, and
  residual charts;
- animation only when it explains causality or iteration;
- semantic zoom from city system to one tensor entry;
- accessible color, typography, and keyboard navigation.

### 1.6 Teaching agent
- prediction before revelation;
- small hand-computable examples;
- step-by-step derivations;
- challenge prompts;
- misconception checks;
- comparison mode;
- instructor-authored lessons;
- automatic generation of figures and snapshots for slides and papers.

---

## 2. User Modes

One computational core with three presentation modes.

### 2.1 Paper Mode
concise framework view; exact notation matching the manuscript;
experiment and ablation dashboards; downloadable SVG/PNG/PDF figures;
citation and equation anchors; residual and certificate tables;
reproducible run manifests; "demonstrated / architectural / future"
claim labels.

### 2.2 Book Mode
chapter map; expandable derivations; concept prerequisites; worked
examples; glossary; historical and cross-disciplinary notes; optional
advanced mathematics panels; checkpoints and exercises; linked code
notebooks.

### 2.3 Teaching Mode
instructor-controlled reveal; lecture sequence; quiz and prediction
prompts; simplified and advanced layers; synchronized student
sessions; assignment templates; exportable figures and problem
statements; progress and misconception analytics.

---

## 3. Global Interface Layout

### 3.1 Top Prompt and Scenario Bar
1. Natural-language prompt (e.g. "Trace the queue on link 27 at
   8:15 AM back to the household groups that generated it"; "Show why
   departure interval τ is not the same as DTA simulation time t";
   "Run five ADMM iterations and explain the primal and dual
   residuals"; "Compare sparse storage with CP rank 4 for the current
   demand tensor").
2. Structured action chips: Explain · Derive · Animate · Compare ·
   Break · Diagnose · Scale · Export.
3. Scenario selector: analytical toy case; fixed appointment; corridor
   queue; TRMG2; regional super-zone; multi-region.
4. Audience selector: intuitive; undergraduate; graduate; researcher;
   implementation.
5. Mathematical depth slider: concept; notation; derivation;
   proof / numerical details.

### 3.2 Left Navigation: Concept Map
1. System Overview
2. Axis and Tensor Lab
3. Tensor Operator Lab
4. Conservation Lab
5. Behavioral Mobility / ABM
6. BN-DTA
7. ADMM Consensus Lab
8. Tensor Compression and Scaling
9. PINN Traffic-State Lab
10. Neural Computational Connections
11. Regional-to-National Hierarchy
12. Model Integration Debugger
13. Experiment and Certification Center

### 3.3 Center Canvas
graph view; tensor view; network map; equation derivation; solver
iteration; traffic-state field; scale hierarchy.

### 3.4 Right Inspector — five tabs
- **What** — plain-language definition.
- **Why** — modeling problem solved and consequences of omission.
- **Math** — definition, equation, domains, units, assumptions.
- **Compute** — data structure, algorithm, complexity, runtime,
  memory.
- **Verify** — conservation law, residual, certificate, failure
  modes.

### 3.5 Bottom Computational Timeline
operator execution order; simulation time; ABM feedback iteration;
ADMM iteration; residual histories; active path/column support; cache
hits and recomputation.

---

## 4. Deep-Understanding Interaction Loop

Predict → Manipulate → Observe → Derive → Verify → Transfer

- **Predict** — the user predicts the direction or consequence before
  running (e.g. "If departure demand moves 15 minutes earlier, which
  queue boundary moves first?").
- **Manipulate** — change one meaningful variable: appointment time;
  capacity; path support; ADMM penalty; tensor rank; sensor coverage;
  regional adapter strength.
- **Observe** — linked visual changes in tensor values, network
  state, behavior, residuals, memory/runtime.
- **Derive** — the system expands the exact mathematical
  transformation.
- **Verify** — the system reports invariants and residuals.
- **Transfer** — apply the concept to another network, region, or
  modeling layer.

---

## 5. Module Specifications

### 5.1 System Overview
Learning objective: the complete modular flow
ℒ → ℋ → 𝒜 → 𝒟 → ℱ → 𝒮 → Ŷ.
Interface: clickable operator graph; forward animation;
adjoint/reverse animation; state inspector; master fixed-point
equation; conservation summary; scale meter.
Required challenge — "break one interface": change units; drop an
axis; aggregate departure periods incorrectly; remove lineage. The
interface must visibly show the resulting modeling error.

### 5.2 Axis and Tensor Lab
Learning objective: why a typed tensor is more than a
multidimensional array.
Core state: 𝒵 = {ℒ, ℋ, 𝒜, 𝒟, ℱ, 𝒮, 𝒴, 𝒰}.
Views: axis registry; interactive tensor cube; fiber and slice
selector; sparse-support view; unit and measure view; lineage view.
Interactions: toggle an axis; condition on an axis; contract an axis;
reorder axes; inspect one entry; compare dense and sparse storage;
distinguish generated and observed axes.
Required tensors: 𝒜[h,n,c,r,u,d,m,τ] · 𝒟[o,d,g,m,τ] ·
ℱ[o,d,g,m,τ,p] · 𝒮[a,ℓ,t,ρ].
Misconception checks: matrix vs tensor; logical full state vs dense
storage; departure interval τ vs simulation time t; flow measure vs
probability measure.

### 5.3 Tensor Operator Lab
Learning objective: connect tensor mathematics to transportation
model transformations.
Operations: inner product; outer product; n-mode product;
contraction; matricization; Kronecker product; Khatri–Rao product;
adjoint; projection; integral operator.
Main linked view: user selects a transportation mapping (activities →
demand; demand → path flow; path flow → link-time flow; traffic
state → observations); the site highlights common axes, contracted
axes, output axes, units, computational complexity.
Required equations: Y(ξ) = ∫_Ω K(ξ,η;𝒵) X(η) dη and its discrete
form Y_i = Σ_j K_ij X_j.
Deep-understanding feature: "continuous ↔ discrete" toggle (integral
kernel; discretized tensor contraction; sparse implementation).

### 5.4 Conservation Lab
Learning objective: conservation as the common language of integrated
modeling.
Panels: probability (Σ_ω P_gω = 1); demand (Σ_p ℱ = 𝒟); link
loading (x_at = Σ δ ℱ); node flow (inflow − outflow = b_it); queue
storage (N^in − N^out = q); activity time (Σ durations + Σ TT ≤
T_day).
Interface: a "conservation ledger" shows quantities entering,
leaving, stored, generated, or lost at every operator.
Challenge mode: introduce a missing path flow; wrong
person-to-vehicle factor; queue exceeding storage; incompatible
aggregation — the user must locate the first broken conservation law.

### 5.5 Behavioral Mobility / ABM Lab
Learning objective: household–activity–tour–mode–time–route
continuity.
Behavioral column ω = (c,r,u,d,m,τ,p), x_gω ≥ 0.
Choice model: P_gω = exp(V_gω)/Σ exp(V_gω′); x_gω = q_g P_gω.
Time windows: e_j ≤ T_j ≤ l_j.
Interface: daily activity timeline; household members; shared vehicle
calendar; destination map; mode and path alternatives; utility
decomposition; feasible/infeasible column filter.
Required challenges: fixed appointment; household vehicle conflict;
mode unavailability; travel-time feedback; activity-chain continuity;
schedule infeasibility.

### 5.6 BN-DTA Lab
Learning objective: the behavior–network fixed point.
Main loop: 𝒜 → 𝒟 → ℱ → 𝒮 → C^experienced → 𝒜⁺.
Residuals: r_network = Gap(ℱ,𝒮); r_schedule = ‖TT^experienced −
TT^scheduled‖; r_behavior = ‖x − ℬ(C^experienced)‖; r_physical =
‖conservation/capacity/causality violations‖.
Interface: time-space traffic diagram; cumulative curves; queue
boundary; path costs; departure-time histogram; residual dashboard;
fixed-point iteration timeline.
Deep-understanding feature: click a queue cell and trace queue cell →
link-time flow → paths → OD → tours **[RECEIVED TEXT ENDS HERE —
§§5.7–5.13 pending from the author]**
