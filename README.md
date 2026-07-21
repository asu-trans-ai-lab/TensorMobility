# TensorMobility

**Mobility systems as flow-through tensors — space, time, and behavior
in one certified computational graph.**

The object of study is the space–time–behavior demand tensor and its
supply companions:

**𝓕**[o, d, group, mode, departure, path, link, time] (persons) ·
**𝓨**[company, vtype, stage, …] (vehicles) · **𝓠**[link, time, regime]
(queues)

Everything in this repository is an **operator acting on named axes of
these tensors** — and every operator carries three readings at once:

| neural | optimization | transportation |
|---|---|---|
| softmax router @ temp 1/θ | entropy-regularized program | **logit choice** (identical, tested) |
| MoE expert path | column of the master | behavioral chain / route |
| router score − duals | negative reduced cost | column pricing |
| deep-equilibrium layer h\*=T(h\*) | fixed point / VI | traffic equilibrium |
| backprop (transposed forward) | adjoint chain | FTT Jacobian B·A·diag(φ′)·Aᵀ·Bᵀ |
| masked-softmax stack | conserved flow chain | zone→OD→path→link (TCG) |
| tanh residual head | admissible utility term | bounded behavioral calibration |
| latent atoms f = Dα | low-rank feasible coordinates | compression on spectator axes |

**The division of responsibility is fixed: neural architecture learns
the search and representation; optimization architecture enforces
feasibility and equilibrium; transportation networks provide physical
meaning and exact verification.**

The axes are real objects (`STBTensor`, named-axis contraction with
Kronecker-lifted spectators), the correspondence is executable
(`tensormobility.neural`, equality-tested), and every equilibrium claim
is certificate-checked (**55 tests**). See
[docs/TENSOR_AXES.md](docs/TENSOR_AXES.md) — *what the mobility,
behavior, time, and space axes are* — the front door of the framework.
TensorMobility is the software home of the STB-FTT research line and
companions DTALite / TAPLite / TCGlite / path4gmns in the GMNS
ecosystem.

```python
from tensormobility.core.stb_tensor import STBTensor
from tensormobility.neural import router_is_logit

F = STBTensor(demand, axes=('od', 'group', 'departure'), measure='persons')
F_paths = F.contract('od', B, new_axis='path')   # spectators ride along
assert router_is_logit(costs, theta=2.0)          # identity, not analogy
```

## Sub-name architecture

| sub-name | import | what it owns |
|---|---|---|
| **TensorMobility.Core** | `tensormobility.core` | axis calculus (spectator / contracted / synchronized × semiring), typed sparse contracts, unified GMNS networks (grid, Sioux Falls, Chicago Sketch) |
| **TensorMobility.DTA** | `tensormobility.dta` | the DTA core: column generation, full-space-certified sparse assignment, latent atoms, classical special cases (logit SUE ↔ UE) |
| **TensorMobility.Dynamics** | `tensormobility.dynamics` | fluid point queues, path/cohort queue loading (the time seam) |
| **TensorMobility.Behavior** | `tensormobility.behavior` | activity chains, activity-DTA coupling, bounded learning residuals |
| **TensorMobility.Engines** | `tensormobility.engines` | the equilibrium engine escalation ladder (Picard → MSA → Anderson → stiff-block Newton → NCP/VI) with cycle detection |
| **TensorMobility.Profiles** | `tensormobility.profiles` | layered equilibria: passenger–vehicle service coupling `Rx ≤ Sy`, mixed-autonomy ride-hailing (MAGE) |
| **TensorMobility.Harness** | `tensormobility.harness` | experiment harnesses, analytical anchors, well-posedness maps |

Future sub-names slot in without touching the canon (the axis registry
extends, never mutates): `TensorMobility.Phase` (phase–time XYZ DTA),
`TensorMobility.Transit`, `TensorMobility.Freight`.

## Quickstart

```python
from tensormobility import load_case, network_from_case, solve_fw

case = load_case('chicago_sketch')          # 933 nodes, 93,135 ODs, GMNS
result = solve_fw(network_from_case(case))  # full-space certified gap
print(result.relative_gap)                  # 8.3e-05 in ~7 s
```

Every network — generated grid, Sioux Falls, Chicago Sketch — loads
through one `StaticNetwork` contract; every solver call is identical
across networks; every reported gap is priced over the full network by
all-origin Dijkstra, never over the enumerated pool.

## Run

```bash
python -m pytest -q                  # 55 tests: tensor+neural identity, contracts, closure, profiles
python cases/run_mage_grid.py        # mixed-autonomy equilibrium + sweeps
python cases/run_unified_grid_harness.py
python cases/run_passenger_vehicle_harness.py
```

## Certified highlights (all reproduced by the suite / case scripts)

- Logit SUE = entropy-master KKT exactly; Wardrop gap 6.9e-2 → 8.3e-6
  along the τ→0 ladder; fixed-pool audit: exact at free flow (2e-16),
  35.7% inadequate under congestion — measured, not assumed.
- Newell point queue matches the closed-form pulse to ≤7e-5; network
  conservation 9e-12 veh.
- Chicago Sketch full network: CG Frank–Wolfe gap 8.3e-5 in 7.4 s;
  link flows correlate 1.0000 with reference volumes; the atom
  compression boundary (K̄ ≈ 2.2 ⇒ atoms cannot pay) reported honestly.
- Two-layer passenger–freight price consensus certified against exact
  LP duals (gap 3.3e-4, price agreement ≤1.6%); BPR version: congestion
  alone prices freight out with no explicit dual.
- MAGE mixed-autonomy profile: 4-outer-iteration equilibrium; patience,
  complementarity, fleet-cap violations 0; the engine ladder's honest
  negative (Picard limit-cycles, detected) is part of the suite.

## Teaching

`teaching/` carries the T1–T7 ladder (FTT chain by hand → logit/UE →
fluid queues → pool audit → compression boundary → prices as
synchronization → mixed autonomy + engines). See
[teaching/README.md](teaching/README.md).

## Workflows

Seven named workflows compose every use of the package — see
[public_gui/workflow.html](public_gui/workflow.html) (self-contained, open locally)
and [docs/ECOSYSTEM_DESIGN.md](docs/ECOSYSTEM_DESIGN.md):

**W1 generate** (self-contained networks) · **W2 assign** (certified
DTA core) · **W3 import** (external kernel audit: TAPLite/DTALite) ·
**W4 calibrate** (dynamic ODME profile) · **W5 visualize** (gui4gmns
dashboards + certificate panel) · **W6 teach** (executable notebook
ladder) · **W7 profile** (layered equilibria).

Pipelines: quickstart W1→W2→W5 · MPO W1→W3→W5 · corridor
W1→W3→W4→W5 · research W1→W7→W5 · classroom W6.

## Papers & docs

- `research_papers/main_v06.tex` — the framework wrap-up draft (compiles clean).
- `docs/ENGINES.md` — the solver escalation ladder position paper.
- `docs/DESIGN_PRINCIPLES_V0_5.md`, `docs/GMNS_DNL_INTEGRATION*.md` —
  passenger/vehicle measures, external DNL exchange contract.

Naming note: **TensorMobility** is the software; **STB-FTT /
Flow-Through Tensors** is the framework name used in the papers. Both
names appear here so each finds the other.
