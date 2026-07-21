# The STB tensor: what the axes are

*The direct answer to: what are the mobility axes? the behavior axes?
the time and space axes? — and where the neural network lives in each.*

The object of study is the demand tensor and its supply companions:

$$\mathcal F_{o,d,g,m,\tau,p,a,t} \quad\text{(persons)},\qquad
\mathcal Y_{k,x,\sigma,\ldots} \quad\text{(vehicles)},\qquad
\mathcal Q_{a,t,s} \quad\text{(queue state)}$$

Everything else in this repository — solvers, profiles, workflows — is
an operator acting on these tensors along named axes.

## The axes, by family

### Space axes
| axis | index | neural reading | optimization reading | transportation reading |
|---|---|---|---|---|
| origin/destination | $o,d$ | token identity (which demand token) | commodity / conservation block $Bf=d$ | zone pair |
| path / column | $p$ | **expert path through the layer stack** (MoE) | column of the master program | route or behavioral chain |
| link / cell | $a$ | shared computation unit experts route through | resource row, capacity dual $\lambda_a$ | road segment (or link-period cell) |

### Time axes
| axis | index | neural | optimization | transportation |
|---|---|---|---|---|
| departure period | $\tau$ | positional channel (spectator until promoted) | block index of time-expanded program | departure-time choice |
| entry time | $t$ | recurrent state unrolled by the loading operator | fixed-point variable of the queue recursion | when flow occupies a cell |
| regime / phase | $s$ | hidden state (HMM); decoded by $(\max,\times)$ | discrete mode of a piecewise program | free-flow / congested / spillback phase |

### Behavior axes
| axis | index | neural | optimization | transportation |
|---|---|---|---|---|
| traveler group | $g$ | batch / conditioning feature | block-diagonal replication (Kronecker lift) | demographic class |
| mode | $m$ | routing gate at the mode layer | column family | car / transit / walk … |
| activity position | (chain) | **depth in the expert-path stack** | precedence structure inside a column | activity-travel chain |

### Mobility (supply-layer) axes
| axis | index | neural | optimization | transportation |
|---|---|---|---|---|
| layer / commodity | — | one tower of a multi-tower network | block of a block-coordinate scheme | passenger vs freight vs operator |
| company | $k$ | competing routers over shared experts | player in a generalized Nash game | TNC |
| vehicle type | $x$ | expert *capability* class | bounded resource pool ($N_{k,x}$, AV cap) | AV / HV / SV |
| operational stage | $\sigma$ | two-phase computation (fetch, serve) | separate incidence into shared cells | pickup vs service |
| matching queue | — | attention bottleneck between demand and supply tokens | complementarity variable (wait ⊥ slack) | customer waiting, patience-capped |

Each axis carries a **status** — spectator (rides along, Kronecker
lift), contracted (summed by a typed operator), synchronized (pinned by
a fixed point) — a **semiring** ($(+,\times)$ flow, $(\min,+)$ pricing,
$(\max,\times)$ decode), and an **anchor** layer. A *model* is a chain
of promotions; a *solver* is a way of computing the promoted axes'
fixed points. This is executable: `tensormobility.core.axes` and
`tensormobility.core.stb_tensor`.

## Where the neural network is — not decoration, identity

The correspondence is exact, not analogical, and it is *tested*
(`tensormobility.neural`):

| neural object | = | transportation object | where certified |
|---|---|---|---|
| softmax router at temperature $1/\theta$ | ≡ | multinomial logit choice | `neural.correspondence` (bitwise-equal test) |
| MoE expert path through masked layers | ≡ | behavioral super-column | bijection on the layered choice DAG |
| router score minus capacity duals | ≡ | negative generalized reduced cost $-\bar c_p$ | pricing = admission rule |
| load-balancing quantile threshold | ≡ | capacity multiplier (complementary slackness) | passenger–vehicle profile |
| deep-equilibrium layer $h^*=T_\theta(h^*)$ | ≡ | traffic equilibrium $f^*=T(f^*)$ | engine ladder computes both |
| backprop (transpose of forward) | ≡ | FTT Jacobian chain $B A\,\mathrm{diag}(\phi')A^{\!\top}B^{\!\top}$ | `neural.tcg_graph` FD-certified |
| masked-softmax layer stack | ≡ | zone→OD→path→link conserved chain (TCG) | analytic backprop, FD 2e-7 |
| bounded residual head (tanh) | ≡ | admissible learned utility term (clean-VI side) | `behavior.bounded_learning` |
| latent atoms $f = D\alpha$ | ≡ | feasible low-rank coordinates on spectator axes | `dta.latent`, crossover law |

The division of responsibility is fixed and is the repository's motto:
**neural architecture learns the search and representation;
optimization architecture enforces feasibility and equilibrium;
transportation networks provide physical meaning and exact
verification.**

## Why the DTA workflows exist at all

W1–W7 are the *operational* view for agencies (kernels, GMNS files,
dashboards). They are one profile of the tensor framework — the
promotion chain {link, departure} under $(+,\times)$/$(\min,+)$ — not
the framework. The identity of the project is this document plus
`stb_tensor` + `neural`; the workflows are how that identity earns its
keep in practice.
