# TensorMobility Visual Computing Studio
## Comprehensive Tensor-Oriented Website and Teaching Interface Specification v2.0

**Status:** Code-ready product and interaction specification  
**Deployment:** Fully offline  
**Primary audiences:** paper readers, book learners, instructors, transportation researchers, numerical optimization researchers, and software developers

---

# 1. Why Version 2 Is Different

Version 1 emphasized coordinates and ADMM, but it did not make tensor structure the primary visual language.

Version 2 is organized around seven visual primitives:

1. **typed axes**;
2. **tensor blocks**;
3. **fibers, slices, and unfoldings**;
4. **contractions and mode products**;
5. **operator-flow graphs**;
6. **factorized and sparse representations**;
7. **consensus and trainable computational blocks**.

The interface must never treat a tensor as merely a decorative cube.

Every visual object must answer:

- What are its axes?
- What is its shape?
- What are its units?
- Which indices are fixed?
- Which axes are contracted?
- Which axes survive in the output?
- What is the transportation meaning?
- What is conserved?
- What approximation error is introduced?
- Which solver updates this object?

---

# 2. Product Name

## Full name

**TensorMobility Visual Computing Studio**

## Short name

**TensorMobility Explorer**

## Core page title

**From Tensor Blocks to Transportation Computational Networks**

---

# 3. Central Narrative

\[
\boxed{
\text{Object}
\rightarrow
\text{Axis}
\rightarrow
\text{Block}
\rightarrow
\text{Contraction}
\rightarrow
\text{Operator Graph}
\rightarrow
\text{Optimization}
\rightarrow
\text{Trainable Network}
}
\]

The user should be able to move continuously between:

1. a transportation object;
2. its typed tensor coordinate;
3. a tensor block, slice, or fiber;
4. an algebraic transformation;
5. a graphical tensor-network contraction;
6. an ABM–DTA operator;
7. a numerical optimization step;
8. a modern neural computational block.

---

# 4. Primary Visual Grammar

## 4.1 Axis rails

Each tensor is displayed with colored axis rails.

| Axis type | Color | Example |
|---|---:|---|
| behavior/person | teal | household, person, group |
| activity/tour | green | pattern, tour, trip leg |
| space/network | blue | zone, OD, path, link |
| time | orange | departure interval, simulation time |
| mode/vehicle | cyan | auto, transit, vehicle class |
| observation | purple | sensor, metric, data source |
| latent/rank | magenta | CP component, Tucker core, expert |
| policy/scenario | gold | toll, capacity, scenario |

The same axis must retain the same color across the entire website.

## 4.2 Tensor block

A tensor block is a rectangular card with:

- tensor symbol;
- semantic name;
- shape badge;
- unit badge;
- visible axis rails;
- storage type;
- sparsity indicator;
- lineage indicator.

Example:

```text
┌──────────────────────────────────────────┐
│ 𝓕  Path Flow Tensor                     │
│ shape: O × D × G × M × Τ × P            │
│ unit: person-trips                       │
│ storage: sparse COO | 18,421 active      │
│ axes: [OD][group][mode][time][path]      │
└──────────────────────────────────────────┘
```

## 4.3 Tensor-network node

A mathematical tensor is also represented as a node with one leg per axis.

```text
       group
         │
origin ─ 𝓓 ─ destination
         │
       mode
         │
        time
```

Connecting equal-colored legs means contraction.

## 4.4 Operator node

An operator is a rounded hexagonal or pill-shaped node.

It displays:

- operator name;
- input and output axes;
- contracted axes;
- forward method;
- projection;
- residual;
- certificate.

## 4.5 Flow edge

Every flow edge must show one of:

- `preserve axis`;
- `translate axis`;
- `contract axis`;
- `generate axis`;
- `synchronize axis`;
- `observe axis`;
- `compress axis`.

## 4.6 Four synchronized meanings

Every concept view must contain four synchronized layers:

1. **visual tensor object**;
2. **mathematical equation**;
3. **transportation interpretation**;
4. **numerical computation ledger**.

---

# 5. Global Interface

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ TensorMobility Explorer | Prompt | Lesson | Audience | Math Depth | Export│
├───────────────┬──────────────────────────────────────┬──────────────────────┤
│ Concept Rail  │ Tensor-Network / Block-Flow Canvas   │ Inspector            │
│               │                                      │ - Meaning            │
│ Foundation    │ linked, zoomable, animated           │ - Equation           │
│ Algebra       │                                      │ - Axes               │
│ Contraction   │                                      │ - Transport           │
│ Decomposition │                                      │ - Certificate         │
│ Optimization  │                                      │                      │
│ ADMM          │                                      │                      │
│ Transformer   │                                      │                      │
├───────────────┴──────────────────────────────────────┴──────────────────────┤
│ Computation Ledger | Shapes | Values | Residuals | Memory | Runtime        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5.1 Concept rail

The left rail is not chapter navigation only. It shows the mathematical dependency graph:

```text
Coordinate
  ↓
Fiber / Slice
  ↓
Unfold / Fold
  ↓
Mode Product
  ↓
Contraction
  ↓
CP / Tucker / TT
  ↓
Sparse Optimization
  ↓
ADMM
  ↓
Typed Transformer Block
  ↓
TensorMobility Flow
```

Completed prerequisites are marked.

## 5.2 Main canvas

The center canvas changes among:

- axis rack;
- tensor small multiples;
- tensor-network diagram;
- operator-flow graph;
- decomposition graph;
- ADMM consensus graph;
- structural/neural dual-lane graph.

## 5.3 Inspector

Tabs:

- **Meaning**
- **Equation**
- **Axes**
- **Transport**
- **Numerical**
- **Certificate**
- **Implementation**

## 5.4 Bottom ledger

The ledger shows the actual operation.

Example:

```text
Input shapes:       B[2×4], X[4×2×2×2]
Contracted axis:    source_cell s
Output shape:       Y[2×2×2×2]
Multiplications:    32
Source total:       50
Output total:       50
Conservation:       PASS
```

---

# 6. Interactive Chapter 1 — Coordinate, Fiber, Slice, and Block

## 6.1 Learning objective

Understand that a high-order tensor is explored through typed selections rather than a fictitious 4D cube.

## 6.2 Canvas layout

Left:

- physical map or transportation records.

Center:

- axis rack;
- selected coordinate;
- fiber and slice small multiples.

Right:

- equation and semantics.

## 6.3 Interaction sequence

1. click one physical object;
2. see the full coordinate tuple;
3. expand one coordinate to a fiber;
4. expand to a slice;
5. show the enclosing tensor block;
6. lock or release axes.

## 6.4 Required visual transformations

### Coordinate

\[
x_{i_1,\ldots,i_N}
\]

Display as one illuminated cell across the axis rack.

### Fiber

Fix every index except one:

\[
\mathcal X_{:i_2\cdots i_N}.
\]

Display as a highlighted colored rail.

### Slice

Fix all but two indices:

\[
\mathcal X_{::i_3\cdots i_N}.
\]

Display as one visible matrix panel.

### Block

Select ranges on multiple axes:

\[
\mathcal X_{\mathcal I_1,\ldots,\mathcal I_N}.
\]

Display as a nested small-multiple panel.

## 6.5 Transportation examples

- one household–tour–mode–time coordinate;
- one OD spatial fiber;
- one link-time traffic-state slice;
- one population-group demand block.

---

# 7. Interactive Chapter 2 — Unfold, Fold, and Axis Reordering

## 7.1 Learning objective

Understand matricization as a reversible reindexing operation.

## 7.2 Visual design

The screen has two linked sides:

```text
Tensor axis rack  →  animated fiber routing  →  unfolding matrix
```

Every tensor fiber physically moves into one matrix column.

## 7.3 Mode selector

User selects:

- mode-1;
- mode-2;
- mode-3;
- grouped unfolding;
- custom row/column axis grouping.

## 7.4 Core equation

\[
\mathcal X
\longleftrightarrow
X_{(n)}.
\]

## 7.5 Visual requirement

No values disappear. Animated colored guide lines show the index mapping.

## 7.6 Transportation example

Unfold:

\[
\mathcal D_{o,d,g,m,\tau}
\]

as:

\[
D_{(od)\times(gm\tau)}.
\]

Use the interface to explain:

- rows = OD pairs;
- columns = behavioral market-time states;
- low rank in this unfolding has a transportation interpretation.

## 7.7 Fold action

The user presses **Fold Back** and verifies:

\[
\operatorname{fold}_n(X_{(n)})=\mathcal X.
\]

---

# 8. Interactive Chapter 3 — Mode Product and Spatial Translation

## 8.1 Visual form

```text
       non-spatial spectator axes
             │ │ │
             𝓧
             │ spatial axis s
             │
             B  spatial translation
             │
             𝓨
             │ target spatial axis z
       same spectator axes
```

## 8.2 Equation

\[
\mathcal Y
=
\mathcal X\times_sB.
\]

Expanded:

\[
\mathcal Y_{z,g,m,\tau}
=
\sum_sB_{z,s}\mathcal X_{s,g,m,\tau}.
\]

## 8.3 Required interaction

- select translated axis;
- freeze spectator axes;
- edit matrix coefficients;
- animate contributions;
- inspect shape change;
- verify conservation;
- show information loss.

## 8.4 Axis transformation badge

```text
space: source cell 4 → zone 2
group: preserved 2 → 2
mode: preserved 2 → 2
time: preserved 2 → 2
```

## 8.5 Deep concept

For different modes:

\[
\mathcal X\times_mA\times_nB
=
\mathcal X\times_nB\times_mA.
\]

The user can swap two independent mode products and observe identical output.

For the same mode:

\[
\mathcal X\times_nA\times_nB
=
\mathcal X\times_n(BA).
\]

The animation visually composes the two transforms.

---

# 9. Interactive Chapter 4 — Tensor Contraction Studio

This is a central chapter.

## 9.1 Visual grammar

Each tensor is a node with labeled legs.

Example: BMR-to-demand contraction.

```text
          household/person/tour
                    │
                    │ contracted
                    │
       ┌────────────┴────────────┐
       │                         │
       𝓑                         𝓐
       │                         │
 OD/group/mode/time          behavior state
       └────────────┬────────────┘
                    │
                    𝓓
```

## 9.2 Operation sequence

1. select two tensors;
2. match compatible axis legs;
3. preview output axes;
4. predict output shape;
5. press **Contract**;
6. animate summation along the shared axis;
7. inspect one output entry term-by-term.

## 9.3 Equation

\[
\mathcal C
=
\mathcal A\times_n^m\mathcal B.
\]

## 9.4 Contraction level selector

- one common mode;
- two common modes;
- all modes / inner product;
- outer product / no contraction.

## 9.5 Transportation contraction gallery

### Behavior to demand

\[
\mathcal D_{o,d,g,m,\tau}
=
\sum_\omega
B_{o,d,g,m,\tau,\omega}x_{g,\omega}.
\]

### Path to link-time

\[
x_{a,t}
=
\sum_{o,d,g,m,\tau,p}
\delta_{a,t}^{p,\tau}
\mathcal F_{o,d,g,m,\tau,p}.
\]

### Traffic state to observation

\[
\widehat{\mathcal Y}
=
H(\mathcal S).
\]

### Accessibility feedback

\[
\mathcal C
=
\Phi_{\mathrm{access}}(\mathcal S,\mathcal D,\mathcal F).
\]

## 9.6 Numerical lens

The ledger must show:

- shared axes;
- surviving axes;
- number of summed terms;
- sparse operations;
- output unit;
- conservation effect.

---

# 10. Interactive Chapter 5 — TensorMobility Block-Flow Canvas

## 10.1 End-to-end graph

```text
[Land Use 𝓛]
     ↓ ΦLH
[Population 𝓗]
     ↓ ΦHA
[BMR 𝓐]
     ↓ contraction ΦAD
[Demand 𝓓]
     ↓ generate path axis
[Path Flow 𝓕]
     ↓ path-link-time contraction
[Traffic State 𝓢]
     ↓ observation operator
[Observed Output Ŷ]
```

## 10.2 Node design

Each node exposes:

- tensor block;
- axes;
- units;
- active support;
- local residual;
- representation backend.

## 10.3 Edge design

Each edge states exactly what it does.

Example:

```text
ΦAD
contracts: behavior column ω
preserves: group g
generates: origin o, destination d, mode m, time τ
```

## 10.4 Expand/collapse

At overview level, show blocks only.

On expansion:

- tensor-network legs;
- equations;
- sparse storage;
- numerical example;
- source/target data tables.

## 10.5 Lineage brushing

Hovering over a traffic-state cell highlights:

- contributing paths;
- OD states;
- tours;
- population groups.

## 10.6 Reverse adjoint mode

A toggle changes the graph direction:

```text
network error
→ path marginal cost
→ demand utility
→ tour/activity sensitivity
→ land-use sensitivity
```

---

# 11. Interactive Chapter 6 — CP, Tucker, and Tensor-Train Lab

## 11.1 Design principle

Do not show decomposition only as static factor matrices. Show:

- what is compressed;
- which axes share latent components;
- what each factor means;
- how reconstruction occurs;
- what error is introduced.

## 11.2 CP view

\[
\mathcal X
\approx
\sum_{r=1}^R
a_r^{(1)}
\circ
a_r^{(2)}
\circ
\cdots
\circ
a_r^{(N)}.
\]

Visual:

```text
component r
behavior factor ─┐
space factor ────┼── outer product → rank-one block
time factor ─────┘
```

Rank slider changes the number of component lanes.

## 11.3 Tucker view

\[
\mathcal X
=
\mathcal G
\times_1U_1
\times_2U_2
\cdots
\times_NU_N.
\]

Visual:

- small core tensor in center;
- one factor matrix on each axis;
- animated mode products reconstruct the full block.

Transportation interpretation:

- core = interactions among latent behavioral, spatial, temporal, and network factors;
- factor matrices = mappings from latent factors to observed axes.

## 11.4 Tucker rank view

\[
\operatorname{rank}_T(\mathcal X)
=
\left(
\operatorname{rank}(X_{(1)}),
\ldots,
\operatorname{rank}(X_{(N)})
\right).
\]

Each axis has an independent rank control.

## 11.5 TT view

Visualize the tensor as a chain of cores with shared latent bonds.

Use TT primarily to explain:

- sequential axis factorization;
- memory scaling;
- latent bond dimension.

## 11.6 Required residuals

\[
r_{\mathrm{repr}}
=
\|\mathcal X-\widehat{\mathcal X}\|_F,
\]

\[
r_{\mathrm{phys}}
=
\|\text{physical violations}\|,
\]

\[
r_{\mathrm{eq}}
=
\operatorname{Gap}(\widehat{\mathcal X},\mathcal S).
\]

These must be shown separately.

---

# 12. Interactive Chapter 7 — Sparse Tensor Optimization

## 12.1 Three synchronized views

1. tensor support mask;
2. optimization objective;
3. active-set evolution.

## 12.2 Entry sparsity

\[
\|\mathcal X\|_0\le s.
\]

The user changes \(s\) and sees active coordinates appear/disappear.

## 12.3 Group sparsity

Select grouping by:

- fiber;
- slice;
- OD market;
- path family;
- region;
- time block.

## 12.4 Low-rank matricization

The interface shows:

\[
X_n=M_n(\mathcal X),
\qquad
\operatorname{rank}(X_n)\le r_n.
\]

The user chooses one unfolding and sees the low-rank constraint act on that matrix view.

## 12.5 Decomposition-based optimization

The interface switches variables from:

\[
\mathcal X
\]

to:

\[
(\mathcal G,U_1,\ldots,U_N).
\]

It must visually explain that the optimization variable itself has changed.

## 12.6 Algorithm gallery

- projected gradient;
- proximal gradient;
- hard thresholding;
- alternating minimization;
- ADMM;
- manifold-constrained update.

Each algorithm uses the same active tensor state so users can compare one step.

---

# 13. Interactive Chapter 8 — ADMM Tensor Consensus Studio

This is the optimization centerpiece.

## 13.1 Visual layout

```text
             ABM local tensor 𝓧A
                    │
                    │ primal disagreement
                    ▼
Demand local 𝓧D → Consensus 𝓩 ← DTA local 𝓧N
                    ▲
                    │ projection ΠΩ
                    │
          conservation / capacity /
          nonnegativity / support
```

Dual variables are shown as colored tension arrows between each local block and consensus.

## 13.2 Step sequence

1. update ABM local tensor;
2. update demand local tensor;
3. update DTA local tensor;
4. average local-plus-dual states;
5. project onto feasible set;
6. update dual tensors;
7. compute primal residual;
8. compute dual residual;
9. compute physical certificates.

## 13.3 Equation carousel

Local update:

\[
\mathcal X_q^{k+1}
=
\arg\min_{\mathcal X_q}
f_q(\mathcal X_q)
+
\frac{\rho}{2}
\|\mathcal X_q-\mathcal Z^k+\mathcal U_q^k\|_F^2.
\]

Consensus:

\[
\mathcal Z^{k+1}
=
\Pi_\Omega
\left[
\frac{1}{Q}
\sum_q
(\mathcal X_q^{k+1}+\mathcal U_q^k)
\right].
\]

Dual update:

\[
\mathcal U_q^{k+1}
=
\mathcal U_q^k+\mathcal X_q^{k+1}-\mathcal Z^{k+1}.
\]

## 13.4 Difference views

For each module:

\[
\Delta_q^k
=
\mathcal X_q^k-\mathcal Z^k.
\]

Display as:

- axis-aware heatmap;
- spatial map;
- fiber chart;
- block summary.

## 13.5 Residual dashboard

Separate:

- primal residual;
- dual residual;
- conservation residual;
- schedule residual;
- network equilibrium gap;
- representation residual.

## 13.6 “ADMM is not averaging” demonstration

The interface compares:

- simple average;
- local penalized optimum;
- projected consensus;
- final feasible state.

## 13.7 Failure scenarios

- \(\rho\) too small;
- \(\rho\) too large;
- incompatible constraints;
- representation too low-rank;
- missing path support;
- physically invalid convergence.

---

# 14. Interactive Chapter 9 — Transformer and Modern Neural Network Bridge

This chapter explains connections without claiming identity.

## 14.1 Dual-lane architecture

```text
STRUCTURAL TRANSPORTATION LANE
typed tensor → exact operator → conservation → equilibrium

TRAINABLE NEURAL LANE
block embedding → masked attention → within-block transform → learned residual

                    ↓ merge
             feasibility projection
                    ↓
              certified output
```

## 14.2 Tensor block to token view

A “token” is defined as a computational view of:

- a tensor coordinate;
- a fiber;
- a slice;
- a sparse block;
- a generated column.

The underlying tensor remains the mathematical state.

## 14.3 Axis/type embeddings

The embedding panel shows:

\[
h_i
=
E_{\mathrm{value}}(x_i)
+
E_{\mathrm{axis}}(a_i)
+
E_{\mathrm{type}}(c_i)
+
E_{\mathrm{space}}(s_i)
+
E_{\mathrm{time}}(t_i).
\]

## 14.4 Attention as learned contraction

\[
\alpha_{ij}
=
\operatorname{softmax}_j
\left[
\frac{q_i^\top k_j}{\sqrt d}
+
M^{\mathrm{type}}_{ij}
+
M^{\mathrm{network}}_{ij}
+
M^{\mathrm{time}}_{ij}
+
M^{\mathrm{causal}}_{ij}
+
M^{\mathrm{feasible}}_{ij}
\right].
\]

\[
h_i'
=
\sum_j\alpha_{ij}v_j.
\]

The visual must show:

- query block;
- compatible key blocks;
- masked incompatible blocks;
- weighted contraction;
- output block.

## 14.5 Transformer block mapping

| Transformer element | TensorMobility interpretation |
|---|---|
| token | coordinate, fiber, slice, sparse block, or generated column |
| embedding | typed numerical representation |
| attention | state-dependent masked contraction |
| MLP | within-block nonlinear transformation |
| residual connection | preserve structural state while adding learned correction |
| sparse MoE | select regional, behavioral, or traffic-regime experts |
| output projection | map learned state back to typed tensor space |
| feasibility projection | transportation-specific hard restoration; not ordinary Transformer softmax |

## 14.6 Critical non-equivalences

The interface explicitly warns:

- attention does not automatically conserve flow;
- softmax normalization is not network equilibrium;
- tokenization is not the tensor state itself;
- Transformer residuals are not ADMM dual residuals;
- neural convergence is not physical feasibility.

## 14.7 Structural-plus-learned equation

\[
\boxed{
\mathcal Z^\star
=
\Pi_\Omega
\left[
\mathcal K_{\mathrm{struct}}(\mathcal Z^\star,\mathcal U)
+
\mathcal N_\theta(\mathcal Z^\star,\mathcal U)
\right].
}
\]

---

# 15. Interactive Chapter 10 — Full TensorMobility Computational Network

## 15.1 Overview

The final chapter combines all previous visual grammars.

## 15.2 Expandable levels

### Level 0 — System blocks

Land use → population → BMR → demand → path flow → traffic state → observations.

### Level 1 — Tensor axes

Expand every block to reveal axes.

### Level 2 — Operator contractions

Expand every edge to reveal the contracted and generated axes.

### Level 3 — Representation backend

Choose sparse, CP, Tucker, TT, or full.

### Level 4 — Solver

Choose direct, FW, column generation, ADMM, PGD, or neural surrogate.

### Level 5 — Certificate

Show conservation, equilibrium, schedule, observation, representation, and gradient checks.

## 15.3 Trace mode

Select one:

- household;
- tour;
- OD state;
- path;
- link-time state;
- observation.

The graph shows forward lineage and reverse attribution.

---

# 16. Prompt as an Interface Controller

The prompt is grounded in the active graph.

## 16.1 Examples

- “Show the spatial fiber containing this coordinate.”
- “Unfold demand with OD as rows.”
- “Contract BMR columns into OD-mode-time demand.”
- “Which axes disappear in this contraction?”
- “Show the Tucker core and reconstruct rank (3,2,2).”
- “Run one projected-gradient step.”
- “Run one ADMM consensus iteration.”
- “Compare attention with this exact path-link contraction.”
- “Highlight every place where conservation is enforced.”
- “Show why this low-rank solution is representation-limited.”

## 16.2 Response sequence

1. locate graph objects;
2. highlight axes;
3. show operation;
4. execute values;
5. verify shapes;
6. verify units;
7. verify certificates;
8. ask one prediction question.

---

# 17. Deep-Learning Pedagogy

Each concept follows:

\[
\boxed{
\text{Predict}
\rightarrow
\text{Construct}
\rightarrow
\text{Transform}
\rightarrow
\text{Inspect}
\rightarrow
\text{Certify}
\rightarrow
\text{Generalize}
}
\]

## 17.1 Predict

Predict output axes and shape.

## 17.2 Construct

Connect tensor legs or choose an operator.

## 17.3 Transform

Run contraction, unfolding, factorization, or solver step.

## 17.4 Inspect

Inspect values, blocks, ranks, or residuals.

## 17.5 Certify

Check conservation, feasibility, and equilibrium.

## 17.6 Generalize

Transfer to a different transportation operator.

---

# 18. Design-Agent Review

The design is reviewed through six simulated expert roles.

## 18.1 Tensor mathematician

Requirements:

- semantic axes;
- exact shapes;
- fibers/slices/unfolding before decomposition;
- tensor-network contraction as primary visual grammar;
- no decorative high-dimensional cubes.

## 18.2 Transportation modeler

Requirements:

- person versus vehicle units;
- departure period versus simulation time;
- conservation at every interface;
- BMR–demand–path–link lineage;
- schedule and network consistency.

## 18.3 Numerical optimizer

Requirements:

- distinguish variable transformation from solver splitting;
- show projected, proximal, alternating, and ADMM steps;
- separate primal, dual, physical, representation, and equilibrium residuals;
- diagnose infeasibility.

## 18.4 Neural-network researcher

Requirements:

- distinguish tensor state from token view;
- explain attention as learned masked contraction;
- show structural and neural lanes;
- avoid false one-to-one equivalence.

## 18.5 Visualization/HCI specialist

Requirements:

- linked brushing;
- focus plus context;
- progressive disclosure;
- consistent colors;
- low visual clutter;
- animated index movement rather than unexplained shape changes.

## 18.6 Instructor

Requirements:

- prediction gates;
- hand-computable examples;
- misconception checks;
- guided and sandbox modes;
- exportable lesson states.

---

# 19. Offline Implementation Architecture

## 19.1 Recommended stack

- SvelteKit or React + TypeScript;
- D3 or custom SVG for tensor-network and block-flow canvas;
- local KaTeX bundle;
- Web Workers;
- no remote services required.

## 19.2 Core components

```text
TensorMobilityStudio
├── PromptController
├── ConceptDependencyRail
├── TensorNetworkCanvas
│   ├── TensorBlockNode
│   ├── AxisLeg
│   ├── OperatorNode
│   ├── ContractionEdge
│   ├── DecompositionGraph
│   ├── ADMMConsensusGraph
│   └── NeuralBridgeGraph
├── LinkedViews
│   ├── CoordinateView
│   ├── FiberView
│   ├── SliceView
│   ├── UnfoldingView
│   ├── SparseSupportView
│   └── DifferenceHeatmap
├── Inspector
└── ComputationLedger
```

## 19.3 State schema

```ts
type TensorAxis = {
  id: string;
  label: string;
  values: string[];
  unit?: string;
  semanticType: string;
  color: string;
};

type TensorBlock = {
  id: string;
  symbol: string;
  name: string;
  axes: string[];
  shape: number[];
  unit: string;
  storage: "dense" | "coo" | "cp" | "tucker" | "tt" | "implicit";
  values?: number[] | SparseEntry[];
};

type TensorOperator = {
  id: string;
  name: string;
  inputBlocks: string[];
  outputBlocks: string[];
  contractedAxes: string[];
  generatedAxes: string[];
  preservedAxes: string[];
  projection?: string;
  certificate?: string[];
};

type SolverState = {
  iteration: number;
  blocks: Record<string, TensorBlock>;
  residuals: Record<string, number>;
  objective?: number;
};
```

---

# 20. Acceptance Criteria

The complete interface is successful when a learner can:

1. identify axes and units of every block;
2. select a coordinate, fiber, slice, and block;
3. watch an unfolding without losing index lineage;
4. connect tensor-network legs to define a contraction;
5. predict output axes and shape;
6. map BMR to demand and path flow to link-time state;
7. compare full, sparse, CP, Tucker, and TT representations;
8. distinguish representation error from physical and equilibrium error;
9. execute PGD, proximal, alternating, and ADMM steps;
10. explain primal and dual tensor residuals;
11. explain attention as learned masked contraction;
12. state where neural methods do not preserve transportation structure;
13. trace a transportation outcome forward and backward;
14. export a paper figure and reproducible state.

---

# 21. Development Order

## Phase 1 — Visual grammar

- axis rails;
- tensor block cards;
- tensor-network legs;
- contraction edges;
- shape and unit ledger.

## Phase 2 — Foundations

- coordinate;
- fiber;
- slice;
- unfolding/folding;
- mode product.

## Phase 3 — Transportation contractions

- BMR to demand;
- path to link-time;
- traffic state to observations.

## Phase 4 — Decomposition and sparsity

- CP;
- Tucker;
- sparse support;
- low-rank matricization.

## Phase 5 — Optimization

- PGD;
- proximal/hard thresholding;
- alternating minimization;
- ADMM.

## Phase 6 — Neural bridge

- block embedding;
- masked attention;
- learned residual;
- sparse experts.

## Phase 7 — Full TensorMobility graph

- TRMG2 example;
- book lessons;
- paper export;
- regional scaling.

---

# 22. Final Design Principle

> The website must make tensor mathematics visible as a living computational network: axes become colored legs, contractions become executable connections, decompositions become alternative internal representations, ADMM becomes coordinated tensor consensus, and Transformer components become trainable block operators embedded inside—not substituted for—the transportation system.
