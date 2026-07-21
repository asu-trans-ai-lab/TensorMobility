# Data inventory — integrated ABM + DTA (+ land use) demonstrations

One table of every dataset the TensorMobility program can draw on, what
axes of the typed state each covers, and which layer or demo it powers.
Ordering follows the GridCity ladder: synthetic grid first, then the
instrumented local network (Tempe), then metropolitan and regional
scales. Certified-case numbers live in
`cases/outputs/`; this file is the map, not the numbers.

## In the package today (loadable via `load_case`)

| dataset | scale | contents | axes covered | powers |
|---|---|---|---|---|
| **GridCity / grid** (generated) | G0 3×3 → G2 50×50 (2,500 nodes / 9,800 links) | parametric GMNS network + synthetic demand; land-use and activity layers synthetic by construction | space, time, behavior, control — all, by design | every explainer card; the self-contained demo suite row; book Ch. 1–16 labs |
| **Sioux Falls** | 24 nodes / 76 links / 528 OD | classic benchmark + path sets, demand scaling | space, behavior | logit SUE ↔ UE ladder, rank-economy sweep (K = 2…16) |
| **Chicago Sketch** | 933 nodes / 2,950 links / 93,135 OD | GMNS network, demand, reference volumes | space, behavior | full-network certified assignment (corr 1.0000 vs reference); the K̄ ≈ 2.2 compression boundary |
| **IEEE TrafficFlowBench I405N** (local only, `TENSORMOBILITY_TFB_DATA`) | 225 nodes / 313 links corridor | detector inflow episodes with dates, released path–link incidence (2,145 chains, rebuilt contiguous), FD parameters for 65 detector chains, departure profiles (288 intervals) | space, **time (real)**, observation | the PINN data face: queue certificates on real inflow, FD priors, physics-informed training |
| **TRMG2 AM** (local only, `TENSORMOBILITY_TRMG2_DATA`) | 33,963 nodes / 75,939 links / 3,247 zones / 1,039,117 OD | full regional GMNS bundle, class demand (sov/hov2/hov3) | space, behavior (classes) | the million-OD certified assignment; regional-scale gate (G3) |

## Instrumented local network: Tempe / ASU (integration next)

Source: `C:\source_codes\0_source_code_new\IRF_Training\ASU_training_material_dashboards\`
(single-file teaching dashboards; data embedded in each HTML).

| dataset | contents | axes covered | powers |
|---|---|---|---|
| **Apache Blvd UTDF** (3 intersections: College / McAllister / Rural) | cycle 110 s, offsets 85/40/40, phase greens, actuated min/max/passage, 12-movement L/T/R counts | space, time, **control** | signal-control layer on a real corridor; DTA node model; the fixed-vs-actuated shared-seed comparison pattern |
| **ASU CV movements** | GMNS nodes/links; **102 intersections, 682 movements, ~20,400 connected-vehicle journeys** with delay mean/median/p95 and travel-time distributions; OD flows | space, time, behavior, **observation** | the CV observation axis 𝒴: ODME calibration targets, delay validation of the queue core, ABM calibration evidence |
| **Tempe citywide timing** | **206 signalized intersections** × timing plans (NEMA phases, splits, offsets), 15 corridors | space, control | city-scale supply model for DTA; signal-optimization future work (the two-column min F(A_x x)+G(y) bridge) |
| **Ped / special-event extension** | WALK/FDW timings, crosswalk geometry, 2× game-day demand | time, behavior | demand-surge and multimodal-signal demos |

## Planned large cases (the two big additions)

| dataset | scale | role |
|---|---|---|
| **DC / NVTA transit** | Northern Virginia multimodal | the transit axis at scale: schedule-based itinerary columns (waiting-band design), rail-incremental and stress cases already prototyped in the column-pool suite |
| **Atlanta ARC** | regional activity-based model network | second million-OD regional gate with an ABM demand chain — the full ℒ→𝒜→𝒟→ℱ→𝒮 loop at G3 |

## Axis-coverage view (what still lacks real data)

- **space** — covered at every scale (grid → corridor → city → region).
- **time** — real dynamics only at the corridor (IEEE detectors) and
  intersection (CV journeys) scales; regional time axis still model-generated.
- **behavior** — demand classes exist (TRMG2), but activity chains and
  departure profiles are synthetic or aggregate; ARC brings real ABM chains.
- **control** — rich real data (UTDF, 206-signal timing) not yet wired
  into the package; first integration target: Apache Blvd as a GMNS
  case with signal-aware link costs.
- **observation (𝒴)** — the CV journey set is the highest-value unwired
  asset: 20,400 ground-truth delay observations for certificate
  validation.

Data-governance notes: TrafficFlowBench and TRMG2 stay local (env-var
paths, tests skip in CI). CV data are aggregates (no trajectories in
the repo). Nothing in this inventory is committed to the public repo
beyond what its license already permits.
