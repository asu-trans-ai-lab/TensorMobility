# TensorMobility ABM–DTA Practice Studio
## Realistic Activity-Based Demand and Dynamic Traffic Assignment Interface Specification v3.0

**Status:** Code-ready offline product, teaching, and implementation specification  
**Relationship to v2:** Companion system. Version 2 teaches tensor mathematics; Version 3 grounds those tensor concepts in realistic ABM–DTA practice.

---

# 1. Product Mission

The **TensorMobility ABM–DTA Practice Studio** is an offline interactive environment for understanding, designing, implementing, testing, and teaching an integrated activity-based model (ABM) and dynamic traffic assignment (DTA) system.

The operational chain is:

\[
\boxed{
\text{Synthetic Population}
\rightarrow
\text{Long-Term Choices}
\rightarrow
\text{Person-Day Patterns}
\rightarrow
\text{Tours and Stops}
\rightarrow
\text{Trips and Vehicles}
\rightarrow
\text{Time-Dependent Paths}
\rightarrow
\text{Dynamic Network Loading}
\rightarrow
\text{Dynamic Skims}
\rightarrow
\text{ABM Feedback}
}
\]

The interface must represent real modeling practice:

- households and persons;
- heterogeneous passenger types;
- work and school locations;
- auto ownership and vehicle availability;
- mandatory, maintenance, discretionary, joint, and work-based tours;
- intermediate stops and activity durations;
- tour mode, trip mode, departure time, and route choice;
- person-trip to vehicle-trip conversion;
- time-dependent paths and generalized cost;
- queues, spillback, signals, tolls, lane restrictions, and parking;
- dynamic skims at configurable temporal resolution;
- scheduled versus experienced travel time;
- ABM, DTA, and cross-model convergence;
- calibration, validation, warm starts, and reproducibility.

---

# 2. Design Thesis

> A true ABM–DTA interface must preserve household, person, tour, trip, vehicle, path, and link-time lineage. Aggregated OD matrices and period skims are useful derived views, but cannot be the sole representation.

The system therefore provides two linked views:

## 2.1 Practice view

Household, person-day, tour, trip, vehicle, path, and network records.

## 2.2 Typed tensor view

Multi-axis states, contractions, generated path supports, residuals, projections, and scalable representations.

Every screen must answer:

1. Which behavioral object is being modeled?
2. Which axes and units define it?
3. Which choices are endogenous?
4. Which constraints are active?
5. What is sent to the next model?
6. What returns through feedback?
7. Which convergence or physical certificate applies?

---

# 3. Canonical Typed States

## 3.1 Household and person state

\[
\mathcal H_{h,p,r,i,a,v,\ell},
\]

where:

- \(h\): household;
- \(p\): person;
- \(r\): person/passenger type;
- \(i\): income group;
- \(a\): auto ownership or availability;
- \(v\): value-of-time class;
- \(\ell\): home/location class.

## 3.2 Person-day and activity state

\[
\mathcal A_{h,p,d,c,\omega,\eta,\tau_s,\tau_e,z,m},
\]

where:

- \(d\): modeled day;
- \(c\): purpose;
- \(\omega\): tour;
- \(\eta\): activity or stop;
- \(\tau_s,\tau_e\): start and end time;
- \(z\): activity location;
- \(m\): mode state.

## 3.3 Tour state

\[
\mathcal T_{h,p,\omega,q,c,z_o,z_d,m,\tau_o,\tau_r,j},
\]

where:

- \(q\): tour structure/type;
- \(c\): primary purpose;
- \(m\): main tour mode;
- \(\tau_o,\tau_r\): outbound and return timing;
- \(j\): joint-tour participant set.

## 3.4 Trip state

\[
\mathcal R_{h,p,\omega,u,o,d,c,m,\tau,\kappa},
\]

where:

- \(u\): trip leg;
- \(o,d\): activity locations;
- \(c\): trip purpose;
- \(m\): trip mode;
- \(\tau\): departure time;
- \(\kappa\): occupancy or sharing state.

## 3.5 Vehicle departure state

\[
\mathcal D^{V}_{o,d,r,c,\nu,\kappa,\tau},
\]

where \(\nu\) is vehicle class.

## 3.6 Path-flow state

\[
\mathcal F_{o,d,r,c,\nu,\kappa,\tau,p},
\]

where \(p\) is a time-dependent route or generated path column.

## 3.7 Dynamic network state

\[
\mathcal S_{a,\lambda,\xi,t,\rho,\nu},
\]

where:

- \(a\): link;
- \(\lambda\): lane or lane group;
- \(\xi\): longitudinal cell;
- \(t\): simulation time;
- \(\rho\): traffic regime or state variable;
- \(\nu\): vehicle class.

A network state can contain:

\[
(q,k,v,w,N^{in},N^{out},Q).
\]

## 3.8 Skim and generalized-cost state

\[
\mathcal C_{o,d,r,c,m,\tau,\mu},
\]

where \(\mu\) includes:

- travel time;
- distance;
- toll;
- parking;
- operating cost;
- reliability;
- generalized cost;
- accessibility logsum.

---

# 4. Passenger and Household Types

The taxonomy must be configurable.

## 4.1 Default person/passenger types

- full-time worker;
- part-time worker;
- university student;
- K–12 student;
- preschool child;
- non-working adult;
- senior/retired adult;
- mobility-limited traveler.

## 4.2 Household dimensions

- household size;
- income category;
- number of workers;
- number of children;
- licensed drivers;
- auto ownership;
- transit-pass availability;
- home zone or parcel;
- telework eligibility;
- shared vehicle availability.

## 4.3 Attributes affecting DTA response

- value of time;
- occupancy;
- toll transponder;
- route-information access;
- willingness to reroute;
- parking eligibility;
- energy/fuel class;
- mobility constraints.

## 4.4 Passenger Type Matrix

The interface displays:

```text
person type × income × auto availability × value of time × accessibility needs
```

Selecting one type updates:

- daily pattern;
- tour frequency;
- mode availability;
- departure-time distribution;
- toll response;
- path choice;
- network outcomes.

---

# 5. Real ABM Choice Hierarchy

## 5.1 Long-term choices

- usual workplace;
- usual school;
- household auto ownership;
- transit pass;
- telework frequency;
- optional residence/location feedback.

## 5.2 Daily pattern

- mandatory participation;
- non-mandatory participation;
- number of tours;
- joint-tour participation;
- work-based subtours.

## 5.3 Tour-level choices

- primary destination;
- main mode;
- departure and arrival window;
- parking location/cost;
- joint participants;
- household vehicle assignment.

## 5.4 Stop-level choices

- stop generation;
- stop purpose;
- stop location;
- stop sequence;
- activity duration.

## 5.5 Trip-level choices

- trip mode;
- departure time;
- access and egress;
- pickup/drop-off;
- route/path;
- en-route rerouting where supported.

## 5.6 Hierarchy view

```text
Household
├── Long-term choices
│   ├── work/school location
│   ├── auto ownership
│   └── telework/transit pass
├── Person-day pattern
│   ├── mandatory tours
│   ├── maintenance tours
│   ├── discretionary tours
│   └── joint tours
├── Tour
│   ├── destination
│   ├── main mode
│   ├── time window
│   └── household vehicle
├── Intermediate stops
└── Trips
    ├── trip mode
    ├── departure time
    └── route
```

Lower-level logsums must be visible as they flow upward.

---

# 6. Tour Taxonomy and Constraints

## 6.1 Mandatory tours

- work;
- school;
- university.

## 6.2 Maintenance tours

- escort;
- shopping;
- personal business;
- medical;
- household maintenance.

## 6.3 Discretionary tours

- social;
- recreation;
- meal;
- visiting;
- other discretionary.

## 6.4 Structural tour types

- individual;
- joint household;
- partially joint;
- work-based subtour;
- chained tour;
- external/boundary;
- visitor.

## 6.5 Tour integrity constraints

### Activity continuity

\[
d^{arr}_{u}=o^{dep}_{u+1}.
\]

### Temporal continuity

\[
\tau^{arr}_{u}+\Delta^{activity}_{u}
\leq
\tau^{dep}_{u+1}.
\]

### Tour closure

\[
z^{final}_{\omega}=z^{anchor}_{\omega}.
\]

### Household vehicle availability

\[
\sum_{\omega:\,v\text{ used at }t}1
\leq A_{h,v,t}.
\]

### Joint-tour synchronization

\[
\tau^{dep}_{p,\omega}
=
\tau^{dep}_{p',\omega},
\qquad p,p'\in J_\omega.
\]

### Main-mode consistency

Different tour modes impose different allowable trip-mode sequences.

The interface must flag invalid tours before DTA loading.

---

# 7. Person-Trip to Vehicle-Trip Conversion

This is a first-class integration page.

## 7.1 ABM output

Person-level trips with identity, tour, purpose, mode, time, and occupancy intent.

## 7.2 Vehicle formation

The conversion reconciles:

- household auto drivers;
- household auto passengers;
- joint/HOV participants;
- escort chaining;
- transit passengers;
- TNC solo/shared rides;
- empty TNC repositioning;
- park-and-ride;
- walk and bike;
- external vehicles;
- freight and service vehicles.

## 7.3 Conversion operator

\[
\mathcal D^V_{o,d,\nu,\kappa,\tau}
=
\sum_{h,p,\omega,u,r,c,m}
G_{\nu,\kappa}
(h,p,\omega,u,r,c,m)
\mathcal R_{h,p,\omega,u,o,d,c,m,\tau,\kappa}.
\]

## 7.4 Vehicle Departure Ledger

For each interval:

```text
person trips
- nonmotorized trips
- transit passenger trips
- auto passengers paired with drivers
+ empty TNC repositioning
+ external vehicles
+ freight/service vehicles
= network vehicle departures
```

## 7.5 Certificates

- no orphan auto passenger;
- no duplicated driver;
- vehicle availability;
- occupancy consistency;
- tour continuity;
- person-to-vehicle conservation;
- feasible connector/loading point.

---

# 8. DTA Supply Features

## 8.1 Network representation

- nodes;
- links;
- lanes/lane groups;
- turn movements;
- centroids and activity-location connectors;
- transit and walk interfaces;
- signals and phases;
- ramp meters;
- toll facilities;
- parking;
- HOV/HOT restrictions;
- dynamic message signs;
- incidents and capacity states.

## 8.2 Vehicle classes

- SOV;
- HOV2;
- HOV3+;
- TNC/taxi;
- light commercial;
- medium truck;
- heavy truck;
- transit vehicle;
- emergency/service;
- EV or other energy class.

## 8.3 Route choice

- pre-trip time-dependent path;
- stochastic or deterministic route choice;
- path-size/commonality;
- traveler-specific value of time;
- toll and parking sensitivity;
- HOV eligibility;
- route-information access;
- en-route rerouting;
- DMS response;
- incident response;
- path-set generation and auditing.

## 8.4 Dynamic network loading

The design supports macro, meso, and micro engines. The default teaching engine is mesoscopic.

Required phenomena:

- queue formation;
- queue dissipation;
- spillback;
- bottleneck activation;
- signal delay;
- lane restrictions;
- turning delay;
- travel-time propagation;
- trajectory ordering;
- FIFO and causality.

## 8.5 Path-flow to link-time contraction

\[
x_{a,t,\nu}
=
\sum_{o,d,r,c,\kappa,\tau,p}
\delta_{a,t}^{p,\tau,\nu}
\mathcal F_{o,d,r,c,\nu,\kappa,\tau,p}.
\]

---

# 9. Dynamic Skim Studio

## 9.1 Skim tensor

\[
\mathcal C_{o,d,r,c,m,\tau,\mu}.
\]

Controls:

- mode;
- traveler type;
- purpose;
- departure interval;
- metric.

## 9.2 Methods

### Experienced travel time

Average experienced time for travelers departing in the interval.

### Fixed-departure TDSP

Time-dependent shortest path at one representative departure time.

### Sampled TDSP

Average of several TDSP calculations inside the interval.

### Path-distribution skim

Expected generalized cost over used/available paths.

### Reliability-aware skim

Travel time plus a reliability penalty.

## 9.3 Active and inactive OD cells

The interface displays:

- active OD-time cells;
- inactive OD-time cells;
- TDSP/imputation coverage;
- fallback static skims;
- missing-data warnings.

## 9.4 Time resolutions

- DTA simulation: seconds;
- link travel-time aggregation: 5 minutes;
- ABM skim intervals: 10, 15, 30, or 60 minutes;
- peak-period DTA with off-peak fallback;
- full-day DTA.

## 9.5 Aggregation warning

One period-level value may conceal:

- short temporal peaks;
- route heterogeneity;
- toll changes;
- incidents;
- schedule infeasibility.

---

# 10. ABM–DTA Feedback

## 10.1 Sequential loop

\[
\mathcal A^k
\rightarrow
\mathcal D^k
\rightarrow
\mathcal F^k
\rightarrow
\mathcal S^k
\rightarrow
\mathcal C^k
\rightarrow
\mathcal A^{k+1}.
\]

## 10.2 Direct feedback

\[
\mathcal C^{k+1}_{ABM}
=
\mathcal C^k_{DTA}.
\]

## 10.3 MSA/damped feedback

\[
\widetilde{\mathcal C}^{k+1}
=
(1-\alpha_k)\widetilde{\mathcal C}^{k}
+
\alpha_k\mathcal C^k_{DTA}.
\]

## 10.4 Parallel/co-evolution mode

ABM scheduling and traffic operations exchange synchronized states during the day. This is an advanced mode and must be distinguished from sequential feedback.

## 10.5 Multirate structure

- fast loop: paths and dynamic loading;
- middle loop: departure time, mode, destination, tour, schedule;
- slow loop: auto ownership, workplace/school, accessibility, land use.

---

# 11. Schedule Reconciliation

The central inconsistency is:

\[
TT^{experienced}
\neq
TT^{scheduled}.
\]

## 11.1 Trip residual

\[
r^{trip}_{h,p,\omega,u}
=
\tau^{arr,exp}_{h,p,\omega,u}
-
\tau^{arr,sched}_{h,p,\omega,u}.
\]

## 11.2 Activity feasibility residual

\[
r^{activity}
=
\max\left(
0,
\tau^{arr,exp}
+
\Delta^{activity}
-
\tau^{next\,dep}
\right).
\]

## 11.3 Tour return residual

\[
r^{tour}
=
\left|
\tau^{return,exp}
-
\tau^{return,sched}
\right|.
\]

## 11.4 Visual comparison

Overlay:

- planned schedule;
- experienced DTA trajectory;
- propagated delay;
- revised ABM schedule.

Conflicts appear as collisions on a 24-hour timeline.

---

# 12. Convergence Dashboard

No single metric is sufficient.

## 12.1 DTA residuals

- relative gap;
- trip gap;
- departure-interval gap;
- route-switching stability;
- link-flow change;
- link travel-time change;
- queue/storage residual.

## 12.2 ABM residuals

- trip-table PRMSE/PMAE;
- tour-frequency change;
- destination-share change;
- mode-share change;
- departure-time change;
- activity-pattern change;
- trip-length distribution change.

## 12.3 Cross-model residuals

- scheduled versus experienced time;
- infeasible trip count;
- tour closure;
- vehicle availability;
- skim change;
- demand–supply consistency;
- generalized-cost consistency.

## 12.4 Residual vector

\[
\mathbf r^k
=
[
r_{route},
r_{link},
r_{skim},
r_{trip},
r_{tour},
r_{schedule},
r_{vehicle},
r_{physical}
].
\]

## 12.5 Status classes

- converged;
- stable but not converged;
- oscillatory;
- DTA-incomplete;
- ABM-incomplete;
- schedule-infeasible;
- representation-limited;
- network-coding-limited;
- calibration-limited;
- unresolved.

---

# 13. Calibration and Validation

## 13.1 Data layers

- traffic counts;
- speeds;
- travel times;
- turns;
- queues;
- signal timing;
- toll transactions;
- transit boardings;
- probe trajectories;
- household travel survey;
- synthetic-population controls.

## 13.2 Time intervals

- 5 minutes;
- 15 minutes;
- 30 minutes;
- daily totals.

## 13.3 Network QA/QC

- centroid connectors;
- loading-point distribution;
- facility type;
- lane count;
- speed/capacity;
- turn restrictions;
- signals;
- toll/HOV rules;
- disconnected paths.

## 13.4 Validation views

- observed/model time series;
- corridor heatmap;
- bottleneck timeline;
- OD-time residual map;
- passenger-type residuals;
- tour-purpose residuals;
- queue episode comparison.

## 13.5 Warm start

Store and reuse:

- path sets;
- link travel times;
- route policies;
- network equilibrium state.

Compare warm and cold starts.

---

# 14. Policy and Operations

## 14.1 Scenario controls

- capacity;
- managed lanes;
- dynamic tolling;
- parking;
- signal timing;
- transit service;
- incidents;
- telework;
- land use;
- mode availability;
- auto ownership.

## 14.2 Dynamic pricing

Price can vary by:

- time;
- facility;
- zone;
- trip or destination;
- speed/occupancy threshold.

## 14.3 Behavioral response depth

The interface distinguishes:

1. fixed-demand post-processing;
2. route response only;
3. departure/mode response through ABM feedback;
4. full tour/activity rescheduling;
5. long-term response.

---

# 15. Main Interface

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ ABM–DTA Practice Studio | Prompt | Scenario | Iteration | Run | Export    │
├──────────────────┬──────────────────────────────────┬───────────────────────┤
│ Practice Rail    │ Integrated Model Canvas          │ Active Inspector      │
│                  │                                  │                       │
│ System           │ household / schedule / network   │ axes, equations,      │
│ Passengers       │ / skim / feedback view           │ records, residuals    │
│ Tours            │                                  │                       │
│ Demand Transfer  │                                  │                       │
│ DTA              │                                  │                       │
│ Skims            │                                  │                       │
│ Feedback         │                                  │                       │
│ Calibration      │                                  │                       │
├──────────────────┴──────────────────────────────────┴───────────────────────┤
│ Integration Ledger: persons | tours | trips | vehicles | paths | residuals│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 16. Interactive Pages

## 16.1 System Map

Complete model and data flow. Every edge declares records, axes, time resolution, units, conversion, residual, and certificate.

## 16.2 Passenger Explorer

Households, person types, long-term choices, auto ownership, value of time, and daily patterns.

## 16.3 Daily Tour Schedule

Multi-person 24-hour view of activities, tours, joint travel, household vehicle use, planned times, and experienced times.

## 16.4 Demand Transfer

Tour records → trip records → person trips → vehicle departures → connectors/loading points.

## 16.5 DTA Network

Route alternatives, path costs, signals, queues, bottlenecks, link-time state, and vehicle classes.

## 16.6 Skim Studio

OD-time heatmaps, fixed and sampled TDSP, experienced time, generalized cost, active/inactive OD coverage.

## 16.7 Feedback and Convergence

One integrated iteration shown step-by-step.

## 16.8 Calibration

Observed/model comparisons and network-coding diagnostics.

---

# 17. Prompt Controller

Examples:

- “Show all mandatory tours for full-time workers.”
- “Highlight joint tours with household vehicle conflicts.”
- “Trace this work tour into vehicle departures and paths.”
- “Show all TNC trips and empty repositioning.”
- “Display the queue encountered by this traveler.”
- “Compare fixed-departure and sampled TDSP skims.”
- “Find trips that became schedule-infeasible after DTA.”
- “Run one direct-feedback iteration.”
- “Change to MSA with alpha 0.3.”
- “Show convergence by tour purpose.”
- “Compare SOV and HOV toll response.”
- “Validate 15-minute speeds on this corridor.”
- “Warm-start from the previous scenario.”

Execution sequence:

\[
\boxed{
\text{Filter}
\rightarrow
\text{Trace}
\rightarrow
\text{Execute}
\rightarrow
\text{Diagnose}
\rightarrow
\text{Certify}
}
\]

---

# 18. Prototype Household

## Household H017

- medium income;
- home in Zone 3;
- one household vehicle;
- three persons.

### P1

- full-time worker;
- work tour;
- lunch work-based subtour;
- auto driver;
- high value of time.

### P2

- part-time worker;
- escort + shopping chain;
- driver/passenger depending on vehicle availability.

### P3

- K–12 student;
- escorted outbound school tour;
- school-bus return.

The example creates a household-vehicle conflict after DTA delay so schedule reconciliation can be demonstrated.

---

# 19. Data Schemas

## 19.1 Household

```ts
type Household = {
  householdId: string;
  homeZone: string;
  incomeGroup: string;
  vehicles: Vehicle[];
  persons: Person[];
};
```

## 19.2 Person

```ts
type Person = {
  personId: string;
  personType: string;
  age: number;
  valueOfTime: number;
  license: boolean;
  transitPass: boolean;
  usualWorkZone?: string;
  usualSchoolZone?: string;
};
```

## 19.3 Tour

```ts
type Tour = {
  tourId: string;
  personIds: string[];
  structure: "individual" | "joint" | "work_subtour" | "external";
  purpose: string;
  anchorLocation: string;
  primaryDestination: string;
  mainMode: string;
  scheduledStart: number;
  scheduledEnd: number;
  experiencedEnd?: number;
  trips: Trip[];
};
```

## 19.4 Trip

```ts
type Trip = {
  tripId: string;
  tourId: string;
  personIds: string[];
  origin: string;
  destination: string;
  purpose: string;
  mode: string;
  departure: number;
  occupancy?: number;
  vehicleId?: string;
  pathId?: string;
};
```

## 19.5 Dynamic path

```ts
type DynamicPath = {
  pathId: string;
  origin: string;
  destination: string;
  departureInterval: number;
  vehicleClass: string;
  linkIds: string[];
  expectedTime: number;
  experiencedTime?: number;
  toll: number;
  probability?: number;
  flow?: number;
};
```

## 19.6 Link-time state

```ts
type LinkTimeState = {
  linkId: string;
  interval: number;
  volume: number;
  inflow: number;
  outflow: number;
  queue: number;
  speed: number;
  travelTime: number;
  capacity: number;
  regime: string;
};
```

## 19.7 Integrated iteration

```ts
type IntegratedIteration = {
  iteration: number;
  abmRunId: string;
  dtaRunId: string;
  skimVersion: string;
  feedbackMethod: "direct" | "msa" | "adaptive";
  residuals: Record<string, number>;
  certificates: Record<string, boolean>;
};
```

---

# 20. Software Components

```text
ABMDTAPracticeStudio
├── PromptController
├── SystemMap
├── PassengerExplorer
├── TourSchedule
│   ├── PersonTimeline
│   ├── TourCard
│   ├── JointTourConnector
│   └── VehicleUseBand
├── DemandTransfer
│   ├── PersonTripTable
│   ├── OccupancyMatcher
│   ├── VehicleDepartureLedger
│   └── ConnectorLoader
├── DTANetworkView
│   ├── RouteAlternatives
│   ├── LinkTimeHeatmap
│   ├── QueueAnimation
│   ├── SignalState
│   └── VehicleClassFilter
├── SkimStudio
├── FeedbackRunner
├── ConvergenceDashboard
├── CalibrationWorkspace
└── ExportManager
```

---

# 21. Required Certificates

## ABM

- household/person completeness;
- activity continuity;
- tour closure;
- time-window feasibility;
- main-mode consistency;
- joint-tour synchronization;
- vehicle availability.

## Demand transfer

- person-trip conservation;
- occupancy consistency;
- no orphan passenger;
- vehicle departure balance;
- connector feasibility.

## DTA

- path connectivity;
- nonnegative flow;
- OD-path conservation;
- node/link conservation;
- queue storage;
- capacity;
- causality/FIFO;
- route gap.

## Cross-model

- scheduled/experienced consistency;
- skim coverage;
- departure-interval consistency;
- trip-table stability;
- tour-pattern stability;
- transparent direct or averaged feedback.

---

# 22. Acceptance Criteria

The version is successful when a user can:

1. inspect a household and its person types;
2. view a full-day activity and tour schedule;
3. distinguish mandatory, maintenance, discretionary, joint, and at-work tours;
4. identify mode and vehicle availability constraints;
5. convert person trips to vehicle departures;
6. detect a household vehicle conflict;
7. trace one trip to a time-dependent path;
8. observe queue and spillback effects;
9. compare scheduled and experienced travel time;
10. compare multiple skim methods;
11. run an integrated feedback iteration;
12. inspect ABM, DTA, and cross-model residuals;
13. validate counts, speeds, and travel times;
14. distinguish route-only, mode/time, tour, and long-term responses;
15. export records and a reproducible run manifest.

---

# 23. Final Positioning

> TensorMobility ABM–DTA Practice Studio is a realistic, typed, and traceable environment for examining how heterogeneous households and persons generate daily activities, tours, trips, and vehicle demand; how vehicle departures are routed and propagated through a time-dependent network; and how experienced network conditions return to activity schedules and travel choices. Its purpose is not simply to connect two software packages, but to make every behavioral, temporal, spatial, vehicle, path, and convergence interface explicit, inspectable, and certifiable.
