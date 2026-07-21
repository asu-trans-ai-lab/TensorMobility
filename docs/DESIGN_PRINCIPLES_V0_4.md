# STB-FTT v0.4 Design Principles

## 1. One paper, one computational spine

The implementation must support all experiments through the same objects:

```text
Demand groups
    -> feasible behavioral/path columns
    -> path flow
    -> link flow
    -> path-departure cohorts
    -> link-time queue state
    -> experienced cost
```

A special case is created by freezing or replacing an operator, not by writing a disconnected program.

## 2. Typed sparse operators, not anonymous tensors

Every linear map follows:

```text
y_target = M[target <- source] x_source
```

Each value declares:

- axis;
- measure;
- unit;
- semantic name.

Each operator declares:

- source and target axes;
- input and output measures;
- orientation;
- stochastic or conservation contract;
- sparse layout.

## 3. Column generation is the spatial scaling mechanism

The grid solver never constructs all feasible paths. At each iteration it:

1. calculates current link costs;
2. solves one exact shortest-path pricing problem per OD;
3. forms the all-or-nothing direction;
4. performs exact line search on the Beckmann objective;
5. stores newly generated paths and their flow weights.

This is simultaneously a link-flow FW algorithm and an implicit path column-generation algorithm.

## 4. Analytical anchors before large cases

Two anchors must pass before scaling:

- a two-parallel-link BPR network with a root-solved analytical UE;
- a single bottleneck with a closed-form fluid queue buildup and clearance curve.

The analytical cases verify that low residuals correspond to the intended equations, not merely to self-consistency within the code.

## 5. Learning is bounded by the certified region

The first AI component is:

```text
interpretable cost + bounded alternative-specific residual
```

The residual is:

- bounded by a `tanh` layer;
- independent of current flow during the static inner assignment;
- trained on a controlled synthetic path-choice recovery task;
- not allowed to replace shortest-path pricing or conservation.

## 6. Queue state preserves path and departure identity

The dynamic backend stores state by:

```text
(path-departure cohort, ordered position along path)
```

Arrival events are sparse dictionaries indexed by simulation time. This avoids a dense `path x departure x link x time` schedule cube.

## 7. Well-posedness is mapped, not assumed

The continuation experiment changes:

- learned behavioral feedback strength;
- physical queue feedback strength.

For each parameter cell it reports:

- number of scalar fixed points;
- local derivative of the raw response map;
- convergence from several initial shares;
- final-start spread.

This is an interpretable canonical map, not a proof for the full network.

## 8. GMNS is the exchange boundary

The canonical output separates:

- OD flow;
- path flow;
- route flow;
- link flow and travel time;
- departure profiles;
- path-departure flow;
- link-time inflow, outflow, and queue.

The same folder can feed TAPLite-style analysis or a thin DTALite/DLSim adapter.

## 9. Certificate discipline

| Slice | Valid certificate |
|---|---|
| Static deterministic assignment | Beckmann objective and FW gap |
| Static Logit SUE | entropy/fixed-point residual |
| Fixed path queue | conservation and time refinement |
| Coupled activity-DTA | fixed-point residual and multi-start spread |
| Learned proposal | held-out recovery plus exact solver audit |

No result is allowed to borrow a stronger certificate from another slice.
