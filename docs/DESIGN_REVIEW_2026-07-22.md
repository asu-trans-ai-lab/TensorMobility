# TensorMobility Design Review

## Scope

This review covers the current repository design, code architecture, visual explainers, and the proposed self-instructing book/course. The central recommendation is to keep:

\[
\text{network science} \rightarrow \text{typed tensor state} \rightarrow
\text{land use/ABM/DTA operators} \rightarrow
\text{graph attention/MoE} \rightarrow
\text{projection/equilibrium} \rightarrow
\text{adjoint optimization}
\]

as one coherent architecture.

## 1. Current strengths

1. **Certificate-first computation.** The repository consistently distinguishes convergence, feasibility, conservation, queue causality, and calibration.
2. **Executable identities.** Logit/softmax, forward/adjoint, flow-through contraction, and fixed-point correspondences are tested rather than stated only as analogies.
3. **Strong DTA and scalable assignment core.** The `StaticNetwork`, path generation, sparse assignment, queue, and engine-ladder components provide a credible computational foundation.
4. **Excellent failure-mode pedagogy.** Static path-pool failure, compression crossover, and fixed-point cycling are valuable teaching cases.
5. **Visual explainers already expose intermediate quantities.** The inspector tabs, ledgers, prediction-before-reveal interaction, and certification views are strong foundations.
6. **The repository states limitations honestly.** Point queues, no spillback, external data, and learned proposal versus exact certificate are distinguished.

## 2. Highest-priority architectural gaps

### 2.1 Land use is present in the narrative but not yet a first-class computational package

Current browser pages include land-use shading and accessibility language, but the Python package lacks a formal `landuse` module.

Add:

```text
tensormobility/landuse/
    state.py
    stock_flow.py
    accessibility.py
    population.py
    location_choice.py
    development.py
    market_clearing.py
    multirate.py
```

Minimum equations:

\[
L_{z,k,y+1}=L_{z,k,y}+D_{z,k,y}-R_{z,k,y},
\]

\[
H_{z,g,y+1}=H_{z,g,y}
+M^{\mathrm{in}}_{z,g,y}-M^{\mathrm{out}}_{z,g,y}
+B_{z,g,y}-E_{z,g,y},
\]

\[
P_{n,z}
=
\frac{\exp(V^{LU}_{n,z})}
{\sum_{j\in C_n}\exp(V^{LU}_{n,j})},
\]

subject to land, housing, employment, zoning, and market-capacity constraints.

### 2.2 Network science should be executable, not only a book theme

`StaticNetwork` is effective for assignment but deliberately avoids a graph-tensor abstraction. Add a network-science layer without replacing the efficient assignment object:

```text
tensormobility/network/
    directed.py
    multilayer.py
    temporal.py
    bipartite.py
    incidence.py
    laplacian.py
    line_graph.py
    communities.py
    centrality.py
    resilience.py
```

Use network science for model construction:

- zone–activity bipartite network;
- household–vehicle resource graph;
- OD–path–link multilayer graph;
- time-expanded activity and DTA graphs;
- land-use–accessibility interaction network;
- observation network;
- policy-intervention network.

### 2.3 `STBTensor` needs a richer semantic contract

The current `STBTensor` correctly enforces named axes, but it should evolve from:

```python
(data, axes, measure)
```

to:

```python
TensorState(
    data=...,
    schema=TensorSchema(
        axes=...,
        measure="person_trip",
        units="persons",
        support=...,
        clock="departure_interval",
        spatial_support="od_pair",
        role="state",
    ),
    lineage=...,
    mask=...,
    backend="numpy|scipy|torch|jax",
)
```

Required additions:

- coordinates and labels for each axis;
- units and measure conversion;
- sparse support and feasibility mask;
- time-scale and clock identity;
- spatial support;
- lineage;
- scenario/version ID;
- backend/device/dtype;
- optional differentiability status.

### 2.4 Axis roles should be operator-relative

The current global axis status is useful pedagogically but too rigid mathematically. An OD axis may be contracted by one operator, retained by another, and synchronized in OD estimation.

Replace one global status with:

```python
AxisUse(
    input_role="preserved|contracted|generated|observed",
    output_role="...",
    synchronization="none|fixed_point|dual|consensus",
)
```

attached to an operator contract.

### 2.5 The repository needs one general block-system object

`MasterFixedPoint` implements \(x=B(L(x))\), which is ideal for the behavioral-loading loop, but integrated land use–ABM–DTA requires multiple clocks and residual blocks.

Add:

```python
class BlockSystem:
    states: dict[str, TensorState]
    operators: list[Operator]
    residual_blocks: list[ResidualBlock]
    clocks: dict[str, Clock]
    scheduler: MultiRateScheduler

    def forward(...)
    def residual(...)
    def jvp(...)
    def vjp(...)
    def solve(...)
```

The system equation becomes:

\[
G(Z;\theta,u)=
\begin{bmatrix}
G_{LU}\\
G_{ABM}\\
G_{DTA}\\
G_{QUEUE}\\
G_{OBS}
\end{bmatrix}=0.
\]

## 3. Modern neural-network integration

### 3.1 Separate three neural roles

Do not use one generic `neural` label.

1. **Representation:** block encoders and low-rank embeddings.
2. **Interaction:** graph attention or cross-attention.
3. **Computation selection:** sparse expert routing.

### 3.2 Implement graph attention as a real package module

The explainers demonstrate masked attention, but the Python package needs:

```text
tensormobility/neural/
    tokenization.py
    graph_attention.py
    cross_attention.py
    positional_encoding.py
    moe.py
    expert_registry.py
    neural_operator.py
```

Transportation attention:

\[
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt d}
+M^{type}_{ij}
+M^{graph}_{ij}
+M^{time}_{ij}
+M^{causal}_{ij}
+M^{feasible}_{ij}
\right).
\]

Masks must be generated from the authoritative graph/tensor mappings, not manually copied into the neural module.

### 3.3 Be precise about the current neural router

`behavior/neural_router.py` is a learned column proposer/ranker. It is not yet a full MoE system. Keep the current claim:

> learned proposal, exact audit, certified optimization.

Then add a true typed MoE interface:

```python
class ComputationalExpert(Protocol):
    input_schema: TensorSchema
    output_schema: TensorSchema
    eligibility: EligibilityRule

    def forward(self, state, context): ...
    def certificate(self, result): ...

class ExpertRouter(nn.Module):
    def route(self, tokens, eligibility_mask, top_k): ...
```

Experts may be:

- logit/choice expert;
- activity scheduling expert;
- shortest-path/pricing expert;
- DNL/queue expert;
- land-use transition expert;
- observation expert;
- neural residual;
- neural-operator surrogate.

### 3.4 Distinguish behavioral probability from computational routing

Softmax routing and logit choice share an algebraic identity, but their semantics differ.

- Logit shares represent traveler choice probabilities.
- MoE weights represent computational allocation.
- Column scores represent proposal priorities.
- Reduced costs represent optimization admission conditions.

The visual explainer should show the common mathematics and the different meanings side by side.

## 4. Adjoint and joint optimization

### 4.1 Move from one-chain backprop to system-level implicit adjoints

The current hand-adjoint tests are an excellent witness. The next object should solve:

\[
G_Z^\top\lambda=J_Z^\top
\]

without constructing \(G_Z\) explicitly.

Add:

```text
tensormobility/optimization/
    block_system.py
    adjoint.py
    linear_solvers.py
    bilevel.py
    policy.py
    active_set.py
```

Required interfaces:

```python
operator.jvp(state, direction)
operator.vjp(state, cotangent)
operator.nonsmooth_status(state)
operator.active_set(state)
```

### 4.2 Separate three optimization problems

The book, code, and explainer should distinguish:

**Calibration**
\[
\min_\theta L(Y,H(Z^\star(\theta,u))).
\]

**Planner optimization**
\[
\min_u J(Z^\star(\theta,u),u).
\]

**Equilibrium**
\[
G(Z^\star,\theta,u)=0.
\]

Calling all three “joint optimization” without this nesting will confuse readers.

### 4.3 Nonsmooth traffic dynamics

Point queues, capacity minima, path-set changes, and discrete schedules are not globally smooth. Report:

- smooth-region adjoint;
- active-set derivative;
- directional derivative;
- generalized derivative;
- finite-difference audit;
- nondifferentiable-event flag.

## 5. DTA and queue design

### 5.1 Keep the point queue as the first certified baseline

The current fluid-queue pulse is ideal for teaching:

\[
q^{k+1}=\max\{0,q^k+\Delta t(u^k-v^k)\}.
\]

Do not call it a complete DTA model by itself.

### 5.2 Add an explicit DNL hierarchy

```text
Level 0: static BPR assignment
Level 1: vertical/point queue
Level 2: path-resolved event queue
Level 3: link transmission or CTM with spillback
Level 4: external DNL adapter (DTALite/DLSim)
```

Every level should implement the same:

```python
DynamicLoadingOperator.forward()
DynamicLoadingOperator.residual()
DynamicLoadingOperator.certificate()
DynamicLoadingOperator.vjp_or_status()
```

### 5.3 Preserve two time axes

Never merge:

- departure interval;
- network simulation time.

The explainer should display both clocks simultaneously.

## 6. Visual explainer recommendations

### 6.1 Replace four independent pages with one shared state engine

Current HTML pages are strong prototypes but contain duplicated CSS, data, and algorithms. Create:

```text
explainer/
    app/
        state-model.ts
        scenario-loader.ts
        certificate-model.ts
        axis-colors.ts
        panels/
        views/
    scenarios/
        gridcity_small.json
        abm_dta_household.json
        landuse_feedback.json
        attention_moe.json
```

Use one authoritative scenario JSON exported by Python.

### 6.2 Provide two modes

**Sandbox mode:** small computations run in the browser.

**Certified replay mode:** the browser loads Python-produced states, histories, and certificates.

The UI must label which mode is active. This prevents browser algorithms from drifting away from package algorithms.

### 6.3 Use four synchronized views

Every lesson should synchronize:

1. **Network view** — nodes, links, layers, agents.
2. **Tensor view** — axes, slice, contraction, support.
3. **Operator graph** — forward dependencies and active experts.
4. **Certificate/influence view** — residuals, gradients, policy effects.

### 6.4 Add a complete land-use feedback explainer

One panel should show:

\[
\text{land stock}
\rightarrow
\text{households/jobs}
\rightarrow
\text{activities}
\rightarrow
\text{OD}
\rightarrow
\text{DTA}
\rightarrow
\text{accessibility}
\rightarrow
\text{location/development}.
\]

Controls:

- housing capacity;
- employment centralization;
- zoning;
- link capacity;
- transit frequency;
- relocation sensitivity;
- development lag.

Outputs:

- annual land-use stocks;
- accessibility;
- household relocation;
- OD changes;
- queues;
- equity effects;
- fixed-point residual by layer.

### 6.5 Add a graph-attention explainer

Show:

- graph-derived eligible neighbors;
- Q/K/V values;
- semantic mask;
- spatial mask;
- time mask;
- final attention matrix;
- resulting tensor update;
- comparison with exact system sensitivity.

### 6.6 Add a true MoE explainer

Show:

- expert eligibility;
- router logits;
- Top-\(K\) activation;
- expert capacity;
- load balance;
- output mixture;
- rejected proposals;
- exact audit/certificate;
- expert influence on the final objective.

### 6.7 Add reverse adjoint animation

Animate:

```text
policy objective
← queue/link state
← path flow
← OD/mode/time
← activity schedule
← household/location
← land use
```

At every edge display:

- local derivative;
- adjoint value;
- accumulated total influence;
- differentiability status.

## 7. Book chapter recommendations

### Part I — Network science and tensor foundations

1. Urban systems as multilayer networks
2. Incidence, Laplacian, paths, cuts, and conservation
3. Temporal and time-expanded networks
4. Typed tensor states and axis calculus
5. Sparse support, mappings, and flow-through lineage

### Part II — Land use and behavior

6. Urban stocks, transitions, and multiple clocks
7. Accessibility and location choice
8. Population synthesis and households
9. Random utility, choice sets, and behavioral operators
10. Activity patterns, tours, schedules, and household resources

### Part III — Traffic flow and equilibrium

11. OD, paths, columns, and static assignment
12. Fluid queues and cumulative flow
13. Dynamic network loading and DTA
14. ABM–DTA consistency and integrated residuals
15. Land-use–ABM–DTA feedback equilibrium

### Part IV — Modern neural computation

16. Graph neural networks and message passing
17. Attention as state-dependent tensor contraction
18. Transformer blocks for multi-rate urban states
19. Sparse computational experts and column generation
20. Neural operators and physics-informed residuals

### Part V — Optimization, observation, and control

21. Projection, invariants, and certificates
22. Observation operators and identifiability
23. Implicit differentiation and adjoints
24. Calibration, policy optimization, and control
25. Scaling from GridCity to multi-city pretraining

Every chapter should follow:

```text
Question → network object → tensor state → operator →
worked GridCity example → code → visual explainer →
certificate → failure case → scaling note.
```

## 8. Code-quality and release recommendations

1. Do not distribute `.git`, `.pytest_cache`, `__pycache__`, logs, or generated outputs in release ZIPs.
2. Mark tests:
   - `unit`
   - `integration`
   - `slow`
   - `external_data`
   - `regional`
3. Keep the default CI suite fast; run regional cases separately.
4. Export all certificates in one JSON schema.
5. Add static type checking and formatting.
6. Separate generated benchmark outputs from source-controlled fixtures.
7. Add versioned scenario manifests and data provenance.
8. Avoid pandas objects inside differentiable kernels; convert to array/tensor state at the boundary.
9. Add a backend abstraction before scaling neural models.
10. Track active versus total parameters, memory, tokens/states, cities, and solver calls—not parameter count alone.

## 9. Recommended implementation order

### P0 — Align the architecture

- Introduce `TensorSchema`, `OperatorContract`, `Certificate`.
- Add minimal land-use stock/accessibility/location modules.
- Add `BlockSystem` and multi-rate clocks.
- Export one Python-generated scenario JSON to all explainers.

### P1 — Make the modern neural extension real

- Implement graph-derived attention masks.
- Implement typed cross-attention.
- Implement typed MoE expert registry and router.
- Preserve exact audits and proposal/certificate separation.

### P2 — Close the mathematics

- Implement matrix-free system adjoint.
- Add active-set/nonsmooth diagnostics.
- Demonstrate calibration and planner optimization on GridCity.

### P3 — Integrate and scale

- Run land-use–ABM–DTA feedback on a 10x10/50x50 grid.
- Connect to regional GMNS.
- Add multi-city pretraining only after the same contracts work across cities.

## 10. Most important principle

\[
\boxed{
\text{Network science defines relations;}
\quad
\text{tensors define states;}
\quad
\text{operators define mechanisms;}
\quad
\text{attention learns interactions;}
\quad
\text{MoE selects computation;}
\quad
\text{projection restores feasibility;}
\quad
\text{equilibrium closes feedback;}
\quad
\text{adjoints enable calibration and policy design.}
}
\]
