# TensorMobility teaching ladder

Seven cases, each a runnable script/notebook target with exercises;
solutions reference package APIs so the material can never rot (CI runs
them). Grounded in the tensor coordination/synchronization study guide
and the certified results of the main suite.

| # | case | teaches | package anchor |
|---|---|---|---|
| T0 | the mobility-tensor paper case (**runnable now**: `python cases/run_mobility_tensor_paper.py`) | complete activity-tour-path columns; typed projections y=Ef, q=Rf, x=Δf, X=Γf; measured opportunity supply U[i,j,m]; the joint Hessian and second-order column pricing verified against exact finite moves | `optimization.mobility_tensor_case`, `optimization.quadratic_pricing` |
| T1 | FTT chain by hand | mode contraction, transpose-symmetry gradients, the 4-OD micro-example | `core.flow_through`, `core.tensor_contracts` |
| T2 | logit ↔ UE ladder | entropy-master KKT = logit; τ→0 limit; self-consistency residuals | `dta.special_cases.solve_case_1` |
| T3 | fluid queue pulse | cumulative counts, closed-form peak/clearing/delay, causality | `dynamics.fluid_queue` |
| T4 | the pool audit | why static path pools fail under congestion (2e-16 → 35.7%); pricing as the fix | `dta.sparse_assignment.full_space_gap` |
| T5 | the compression boundary | route-family atoms on Chicago: K̄ ≈ 2.2 ⇒ compression cannot pay — a certified negative result | `dta.latent`, `dta.sparse_assignment.solve_atom_gp` |
| T6 | prices as synchronization | two layers, one capacity: LP duals vs iterative price consensus; congestion as an implicit price | `profiles` + the multilayer case scripts |
| T7 | mixed autonomy + engines | MAGE profile; two queues (matching vs road); why damped iteration limit-cycles and stiff-block Newton closes it | `profiles.mage_profile`, `engines.equilibrium_engines` |

Design rule for every case: run → certify → explain. Each case prints
its certificate table first; the narrative explains what the
certificates mean and what breaks without them. T4, T5, and T7 are
deliberate *failure-mode* lessons — the pool that degrades, the
compression that cannot pay, the iteration that cycles — because the
diagnosis discipline is the curriculum.

T0 is the first fully runnable rung: the 5×5 grid case of the TRB
paper draft (`research_papers/trb_mobility_tensor/`), hand-checkable
end to end, CI-tested (`tests/test_quadratic_pricing.py`,
`tests/test_landuse_activity_dta_integration.py`), and exporting its
own `certificates.json`. Every number in the paper comes from that one
script — the teaching case and the paper evidence are the same
artifact, so neither can rot without the other noticing.
