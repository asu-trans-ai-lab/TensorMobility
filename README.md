# TensorMobility

**Mobility systems as flow-through tensors — space, time, and behavior
in one certified computational graph.**

TensorMobility implements the Space–Time–Behavior Flow-Through Tensor
(**STB-FTT**) framework: origin–destination flows, behavioral columns,
path probabilities, link times, queues, and service layers as chained
typed tensors, with every equilibrium certificate checked in the test
suite (**49 tests**). It is the software home of the FTT research line
(Flow-Through Tensors) and companions DTALite / TAPLite / TCGlite /
path4gmns in the GMNS ecosystem.

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
python -m pytest -q                  # 49 tests: contracts, closure, profiles
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

## Papers & docs

- `paper/main_v06.tex` — the framework wrap-up draft (compiles clean).
- `docs/ENGINES.md` — the solver escalation ladder position paper.
- `docs/DESIGN_PRINCIPLES_V0_5.md`, `docs/GMNS_DNL_INTEGRATION*.md` —
  passenger/vehicle measures, external DNL exchange contract.

Naming note: **TensorMobility** is the software; **STB-FTT /
Flow-Through Tensors** is the framework name used in the papers. Both
names appear here so each finds the other.
