# GMNS / TAPLite / DTALite Exchange for Passenger–Vehicle STB v0.5

## Input identities

The v0.5 adapter retains four ID domains:

1. passenger group and behavioral column;
2. mobility service request;
3. vehicle service pattern/route;
4. physical path, link, departure interval and simulation time.

## Passenger and service outputs

```text
passenger_group.csv
passenger_column_flow.csv
passenger_mode_share.csv
passenger_departure_share.csv
activity_mode_share.csv
service_request_flow.csv
```

`passenger_column_flow.csv` contains one feasible activity/mode/departure/route alternative with probability, person flow and generalized cost. `service_request_flow.csv` aggregates shared-ride and transit demand and reports dual price, average service price, supplied capacity and slack.

## Vehicle and road outputs

```text
vehicle_route_flow.csv
road_link_vehicle_flow.csv
multimodal_queue_history.csv
```

`vehicle_route_flow.csv` contains service mode, departure interval, served request IDs/capacities, ordered physical link sequence, operating cost and continuous vehicle flow. `road_link_vehicle_flow.csv` aggregates private-auto access and generated service vehicles before BPR/DNL loading.

## Explicit coupling table

```text
passenger_vehicle_coupling.csv
```

The table serializes nonzeros of:

\[
R(\text{request}\leftarrow\text{passenger column}),
\qquad
S(\text{request}\leftarrow\text{vehicle pattern}).
\]

This makes passenger-to-service and vehicle-to-capacity mappings auditable outside Python.

## External DNL contract

A DTALite/DLSim backend should accept vehicle path-departure flows, not passenger flows. For private auto, one passenger column may generate a private vehicle path. For shared ride and bus, service vehicle route columns generate the road paths. Park-and-ride produces both a private access vehicle segment and a passenger transit request.

Expected external inputs therefore include:

```text
node.csv
link.csv
vehicle_path.csv
vehicle_path_departure_flow.csv
```

Recommended returned tables are:

```text
vehicle_path_departure_time.csv
link_time_inflow.csv
link_time_outflow.csv
link_time_queue.csv
```

The STB passenger cost operator then maps those vehicle-generated times back to the passenger itinerary segments. Passenger, service-request, vehicle-route and physical-path IDs must remain stable across the exchange.
