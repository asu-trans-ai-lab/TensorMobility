# Choice Graph — related work (annotated)

*For the Choice_Graph revision and the behavior-axis core
(`tensormobility/behavior/choice_graph.py`). Grouped by the role each
line plays; one line each on what to take and how we differ.*

## Activity-based microsimulation (the microstructure lineage)

1. **Kitamura, Chen, Pendyala & Narayanan (2000)**, *Micro-simulation of
   daily activity-travel patterns for travel demand forecasting*,
   Transportation. Sequential, history- and time-of-day-dependent
   pattern generation; "rigidities in daily schedules matter." The
   direct ancestor of the choice graph's microstructure claim — cite
   first.
2. **Bowman & Ben-Akiva (2001)**, activity-based day pattern schedules
   (TR-A). The nested day-pattern logit = a two-level choice graph.
3. **Recker (1995)**, HAPP — household activity pattern problem as MILP
   routing. The optimization face of the same chain object.
4. **Arentze & Timmermans (2004)**, ALBATROSS. Rule-based sequential
   choices = a decision-tree choice graph.

## Supernetworks (the graph-expansion lineage)

5. **Nagurney & Dong (2002)**, multimodal supernetworks. Mode+route in
   one expanded graph, equilibrium on the expansion.
6. **Liao, Arentze & Timmermans (2010, 2013)**, multistate supernetworks
   for activity-travel scheduling (TR-B). Closest existing formalism to
   the choice graph — state (vehicle location, activity progress) ×
   space expansion; our difference: typed measures, column/tensor face,
   certificates, and the equilibrium loop x = B(L(x)).
7. **Mahmoudi & Zhou (2016)**, state-space-time networks for
   pickup/delivery (TR-B). The lab's own state-expansion machinery; the
   choice graph is its demand-side sibling.
8. **Liu, Zhou et al.**, space-time activity networks (foundational
   activity-network line already flagged in the Paper-2 notes).

## Sequential choice = flat choice (the identity we test)

9. **Dial (1971)**, STOCH — logit assignment without enumeration.
10. **Fosgerau, Frejinger & Karlström (2013)**, the recursive logit
    (TR-B). The sequential face of the choice graph; our
    `sequential_chain_shares == logit_chain_shares` test is exactly the
    link-additivity equivalence, extended to activity chains.
11. **Mai, Fosgerau & Frejinger (2015)**, nested recursive logit —
    the road-map for relaxing arc-additivity (scale parameters per
    stage) in a future choice-graph version.

## ABM–DTA integration and consistency (the loop lineage)

12. **Waller & Castiglione (2018 draft)**, *Considerations in Linking
    Advanced Transportation Demand and Supply Modeling* — consistency
    as the first-class requirement; the choice graph's equilibrium
    closure is the formal answer.
13. **Ruiz Juri, James, Jiang, Duthie, Pinjari & Bhat**, *On the
    computation of skims for large-scale ABM-DTA* (Austin CTR). Two
    findings to import: (a) skim time-step granularity (10–30 min)
    materially changes integrated-model convergence — in our terms, the
    resolution of L's output measure is a convergence parameter of the
    master loop, worth a sweep in the harness; (b) simulation noise
    swamps sophisticated skim estimators — supports the certificate
    view (report residuals, don't over-engineer the estimator).
14. **Lin, Eluru, Waller & Bhat (2008)**, CEMDAP–VISTA integration
    (TRR); the sequential-linkage baseline the loop improves on.
15. **Pendyala et al. (2012)**, SimTRAVEL (TRR) — minute-by-minute
    tight coupling; **Auld et al. (2016)**, POLARIS (TR-C) — integrated
    agent platform; **Balmer et al. (2009)**, MATSim architecture.
16. **MPO ABM–DTA Integration Vision** (2020, internal agency report) —
    "essential features of integrable ABM and DTA"; the software-level
    counterpart of which the choice graph is the open algebraic form.
17. **Zockaie, Saberi, Mahmassani et al. (2015)**, ABM+DTA with
    heterogeneous users for toll forecasting — the policy use case the
    co-opetition experiments mirror.
18. **Halat et al. (2016)**, dynamic network equilibrium for daily
    activity-trip chains (Transportation) — chain-level equilibrium
    precedent.

## Shared mobility interaction (the application)

19. **Mahmoudi, Tong, Garikapati, Pendyala & Zhou**, *How many trip
    requests could we support?* — passenger activity-travel graphs
    coupled to vehicle networks by Rx ≤ Sy with dual trip prices; the
    integer corner of the choice-graph master.
20. **Hou, Wang, Li & Pang (TRB-D-26-00426)**, MAGE mixed-autonomy
    ride-hailing GNE — competition/cooperation of AV/HV fleets with
    customer patience; the equilibrium cross-response methodology for
    the co-opetition experiments (already an operator profile in this
    repo).

## Positioning sentence (for the revision)

The choice graph formalizes the behavior axes of the STB tensor as a
layered DAG whose complete chains are behavioral columns; it inherits
the sequential face of recursive logit (Dial → Fosgerau) and the
expansion face of multistate supernetworks (Liao–Timmermans), closes
the ABM–DTA loop demanded by the consistency literature
(Waller–Castiglione; Ruiz Juri et al.), and measures cross-mode
competition/cooperation as equilibrium cross-response signs rather than
fixed-supply derivatives.
