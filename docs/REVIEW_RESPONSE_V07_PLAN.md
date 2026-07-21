# Response to the full technical review (2026-07-21) and v0.7 plan

Reviewer verdict accepted: **strong prototype — not yet a paper-ready
unified artifact.** Scorecard and all eight technical issues
acknowledged. Status of the P0 items (this commit):

| P0 item | status |
|---|---|
| 1 Orientation standardization | adapter + docs/ORIENTATION.md now; module migration v0.7 |
| 2 Well-posedness statement | corrected (master_loop docstring; Brouwer existence / contraction for uniqueness; rho(dT/dx)<1 reporting) — figure Panel 1 wording should follow |
| 3 Typed measure-aware master loop | DEFERRED to v0.7 (STBState + operator signatures; design accepted as written in the review) |
| 4 MAGE final-state consistency | DONE: full x*->D*->z*->w*->T*->x*' pass, r_x/r_z/r_T reported + tested; op_cost now drives fleet priority; claim remains "operator-profile demonstration", NOT solved GNE |
| 5 Data in the wheel | DONE: tensormobility/data + package-data + wheel-install CI job |
| 6 Torch oversubscription | DONE: default 1 thread via TENSORMOBILITY_TORCH_THREADS |
| 7 Unsupported claims | freight benchmark demoted to companion-archive note; Chicago 1.0000 correlation now an executable test; NCP/VI marked "declared, not executable"; minimal `tm` CLI added |
| 8 Counts/versions | synchronized at 66 tests / version 0.6.2 |

## Accepted v0.7 architecture (from the review, recorded as the plan)

- Four-tensor system: behavioral demand X[h,n,c,r,u,m,tau,p]; vehicle
  service Y[k,v,sigma,omega,tau,p,a,t]; traffic state
  Q[a,l,xi,t,s] with regime axis and stability classification
  (converged / damped oscillation / limit cycle / divergent /
  horizon-truncated); performance/observation y.
- Compressed axis profiles as formal contract: od := origin x
  destination; behavioral_column := activity x destination x mode x
  departure x route; link_time := link x simulation_time.
- Discrete adjoint of the causal loading process (lambda_t backward
  recursion) distinguished from generic autodiff and from equilibrium
  differentiation.
- Super-columns with multiple incidence patterns (link, activity,
  vehicle, schedule); fixed appointments as degenerate time windows
  e_j = l_j.
- Two-sided column generation (passenger columns x_omega, vehicle
  columns y_nu; service prices lambda_i + network prices mu_at).
- Four capacity meanings separated: physical / service / fleet /
  behavioral.
- Four implementation modes solving one canonical test: dense / sparse
  coordinate / factorized / implicit-operator.
- Multilevel Frank-Wolfe (super-zone -> district -> full) with
  full-space certification and super-queue disaggregation checks.

## Experiment ladder (Cases A–F)

A fixed-appointment behavioral example; B two-sided passenger-vehicle
(MILP benchmark vs two-sided CG vs ADMM vs fixed point vs duty atoms);
C corridor queue-regime test with adjoint check; D Sioux Falls
activity-DTA; E Chicago Sketch static scalability; **F TRMG2 regional
tensor benchmark** (gmns_transfer; stages F1 transfer validation, F2
super-zone FW warm starts, F3 sparse multi-axis assignment with
axis-wise aggregation, F4 column generation + latent atoms, F5
time-dependent subarea, F6 queue stability + CBI outputs). The partner MPO application remains
private-data, shown through aggregates only.

## Paper positioning (accepted)

"TensorMobility is a typed sparse operator framework with certified
special-case closure, scalable path column generation, and extensible
activity and passenger-vehicle profiles" — not yet "a general solver
for integrated multimodal dynamic equilibrium / GNE".
