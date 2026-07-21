# GMNS, TAPLite, and DTALite/DLSim Integration Contract

## Canonical input

```text
node.csv
link.csv
demand.csv
path.csv
departure_profile.csv
```

### `node.csv`

Required: `node_id`.

Recommended: `zone_id`, `x_coord`, `y_coord`, `geometry`.

### `link.csv`

Required canonical fields:

```text
link_id
from_node_id
to_node_id
free_flow_time
capacity
```

Optional: `length`, `lanes`, `facility_type`, `geometry`.

### `demand.csv`

```text
od_id
origin_node_id
destination_node_id
volume
```

### `path.csv`

```text
path_id
route_id
od_id
origin_node_id
destination_node_id
path_flow
node_sequence
link_sequence
link_index_sequence
```

### `departure_profile.csv`

```text
departure_interval
share
```

Shares must be nonnegative and sum to one for the declared demand period.

## Canonical output

### Static/TAPLite-style

```text
od_flow.csv
path_flow.csv
route_flow.csv
link_flow.csv
solver_history.csv
```

### Dynamic/DNL-style

```text
path_departure_flow.csv
path_departure_time.csv
link_time_flow.csv
network_queue_summary.csv
```

`link_time_flow.csv` uses:

```text
link_id
time_step
minutes
inflow
outflow
queue
travel_time   # external backend or later internal extension
```

## Backend interface

The DNL backend receives:

```text
network
path flow by OD/path
departure profile or path-departure flow
capacity/control scenario
```

It returns:

```text
experienced path-departure travel time
link-time inflow/outflow/queue/storage
completion and conservation diagnostics
```

## Required cross-backend checks

1. Path and departure IDs survive the round trip.
2. Loaded path-departure demand equals the input demand after person-to-vehicle conversion.
3. Per-link queue change equals inflow minus outflow, subject to the backend's storage definition.
4. Completed + queued + in-transit mass equals total loaded vehicles.
5. Free-flow no-congestion travel times agree within the declared time discretization.
6. Capacity shock directions are qualitatively consistent before comparing exact magnitudes.

## Storage constraint treatment

A storage constraint should be physical in a finite-storage DNL:

```text
occupancy(link, time) <= storage_capacity(link)
```

An observation loss may be added when storage or queue is measured:

```text
L_storage = ||observed_storage - modeled_storage||^2
```

The loss calibrates discrepancy; it should not replace the physical receiving/storage constraint.
