# Computational-graph TSE — what TensorMobility adopts

Source studied: the traffic-state-estimation computational-graph
codebase (Jiawei Lu, ASU, 2021, MIT license; the Lu–Li–Wu–Zhou TR-C
research line). A ~700-line TensorFlow-2 prototype estimating a
continuous corridor state field k(x,t), v(x,t), q = k·v from
heterogeneous sensors on the **Mobile Century** experiment (I-880,
Feb 8 2008: 168 loop records, 19,229 GPS probe pings, 192 vehicle
travel-time traversals, 6 ramp locations), on a 51×121 space–time
grid.

## The five adoptions

1. **Observation operators — ADOPTED, IMPLEMENTED.** In the source,
   every sensor is compiled offline into a sparse linear weight matrix
   over grid collocation points (bilinear interpolation for point
   probes, trapezoidal integration for section counts), so every data
   term is `W @ field`. This is exactly a typed-axis contraction
   against the observation axis 𝒴. Now in
   `tensormobility/adapters/observation_operators.py` with 6 tests
   certifying exactness on bilinear fields, convex row weights, and
   trapezoidal exactness on linear densities.
2. **Book Ch. 15 witness: the LWR adjoint IS backprop.** The source
   computes the conservation residual ∂q/∂x + ∂k/∂t by nested
   autodiff on the (x,t) inputs while jointly training six physical
   scalars (vf, kj, bottleneck-discharge curve parameters) — a
   compact runnable proof that PDE-constrained estimation gradients
   are backprop. Queue as the Ch. 15 GridCity-lab witness.
3. **Differentiating through the queue fixed point.** The source's
   travel-time loss runs six fixed-point iterations of a Newell-style
   delay w = (Q−D)/μ *inside* the graph and differentiates through
   them — the same fluid-queue map as `tensormobility.dynamics`.
   Planned: a test differentiating our queue map and an explainer
   panel-12 (adjoint influence) demo.
4. **Ramp-masked conservation priors.** Conservation residuals are
   masked at ramp columns (merge cells violate the closed-corridor
   PDE). Directly applicable to the I405N corridor face: the FD/PDE
   priors for the 65 detector chains must be switched off at known
   ramp postmiles before PINN training.
5. **Mobile Century as a second corridor benchmark.** Small, real,
   multi-sensor, MIT-licensed, in-repo data — a natural
   cross-validation companion to TrafficFlowBench I405N (which stays
   local). Planned as `load_case('mobile_century')`.

## Pitfalls noted for the port

Windows-backslash relative data path (breaks on Linux); corridor
constants hardcoded in code (t0=2100 s, t3=6180 s, 5,000 m, μ cap)
rather than configured; no requirements.txt, seed control, or
shuffled batching; per-step Python-side clamps on two shape
parameters silently bound their learning rate. The port keeps the
mathematical patterns and leaves the prototype's engineering behind.
