# Learning from the PINN framework (JSQE) and integrating it

*Teaching note + integration design. Source: Lu, Li, Wu & Zhou (2023),
"Physics-informed neural networks for integrated traffic state and
queue profile estimation" (TR-C 153, 104224) and its companion
computational-graph code (`refenrences/`), plus the TrafficFlowBench
corridor data now wired in via `adapters/trafficflowbench.py`.*

## 1. What the JSQE-PINN actually is (anatomy lesson)

Three design moves, visible directly in the companion `net.py`:

1. **Continuous fields with a shared trunk.** One MLP trunk over
   space–time (x, t) feeds two heads: density k(x,t) and speed v(x,t),
   with flow defined *structurally* as q = k·v — one identity is
   hard-wired, not learned.
2. **Physical parameters are trainable variables next to the network
   weights**: free-flow speed v_f, jam density k_j (the fundamental
   diagram), and the queue-discharge profile parameters (t1, μ(t1),
   γ). The network learns the *field*; interpretable physics learns
   the *regime*. This is why the output is a queue **profile** (Newell
   fluid approximation with analytic delay/travel time), not just a
   heat-map.
3. **Physics enters as residual loss terms** — FD consistency and the
   conservation PDE penalize violations — and the whole layered graph
   is trained by a forward–backward pass (the discrete adjoint we
   equality-tested against autograd this week).

## 2. The lesson, stated as our axis calculus

| JSQE-PINN element | TensorMobility reading |
|---|---|
| shared trunk + heads over (x,t) | S_repr operator on the **state axes** (link-cell × time) of tensor Q |
| q = k·v hard-wired | *structural* constraint — same doctrine as our conserved chain: build identities in, don't penalize them |
| conservation as a *penalty* | our conserved operators make it *exact* — the key difference: **PINN penalizes what typed operators can enforce**; use penalties only for constraints that cannot be structural (the PDE across cells) |
| trainable v_f, k_j, μ(t) | bounded interpretable parameters — exactly the `bounded_learning` discipline (learn few named physical quantities, not free fields) |
| physics residual value | a **certificate**: report it like a gap, per state cell |
| estimation task | the (max,×)/state face — NOT the equilibrium face, so the clean-VI monotonicity boundary is not at risk; what applies instead is the **identifiability gate** on (v_f, k_j, μ): sensors must make them recoverable (G0 logic) |

So "learning from PINN" ≑ three imports: (i) continuous-field heads on
state axes, (ii) trainable named physics with priors/bounds, (iii)
residual-as-certificate. And one export back: move every constraint
that CAN be structural out of the loss — the typed chain does
conservation exactly at 1e-13; a penalty would do it at 1e-3.

## 3. Where TrafficFlowBench plugs in (done this commit)

`adapters/trafficflowbench.py` loads the IEEE five-corridor panels
(I405N first): GMNS corridor network → `StaticNetwork`; released path
set + base OD; 5-min detector states; queue episodes (T0/T2/T3 —
Newell profile material); per-detector FD parameters (v_f, capacity,
k_crit, k_j — the *priors* for the trainable physics); historical
profiles → the **simple departure-time profile** (the author-sequenced
first behavior step).

Certified now (5 tests, local-data skip in CI):
- corridor loads through the canonical contract (225 nodes, 313 links);
- the released path–link incidence is reconstructed **exactly** from
  the link sequences;
- the DTA core certifies on the routable OD set (full-space gap <1e-4);
- the queue core runs point-queue certificates (conservation,
  causality, queue formation) on real detector inflow at the largest
  episode's bottleneck under a documented 30% capacity reduction;
- the departure profile normalizes and peaks in a plausible window.

**Measured data characteristic, reported not hidden**: 2,068 of 2,145
released link_seq chains are detector-incidence sequences, NOT
contiguous walks (1,547 not even ordered along the corridor), and some
anchor sets cannot lie on one directed walk. The adapter therefore
carries BOTH faces: the released incidence exactly (ODME face), and
`rebuild_contiguous_paths()` — graph-ordered anchors, shortest-connector
stitching, branch anchors dropped with counts — which rebuilds ALL
2,145 paths as verified contiguous walks at 76.9% anchor coverage
(28,435/36,980; 47,879 connector links inserted). Unroutable/self ODs
in the raw endpoint aggregation are likewise dropped with counts.

## 4. Integration roadmap (JSQE face on TFB data)

1. **Physics-residual certificate module** (v0.7): given detector
   states + FD priors, evaluate per-cell FD residual |q − Q_fd(k)| and
   discrete conservation residual on the link chain — a Task-3-style
   certificate we can attach to ANY reconstruction, learned or not.
2. **JSQE profile**: trunk+heads over (milepost, t) per corridor;
   trainable (v_f, k_j) initialized at the released FD values with
   bounds; μ(t) from queue episodes; losses = masked-data fit +
   PDE residual; conservation via the typed chain where cells are
   linked. Certificates: residuals + identifiability report of
   (v_f, k_j, μ) under the sensor layout.
3. **Behavior step**: replace the exogenous base OD with departure-
   profile-weighted demand (the adapter's profile), then ODME through
   the released incidence (the Task-4 face) using the typed-chain
   gradient machinery — the dynamic-odme-lab design lands here.
