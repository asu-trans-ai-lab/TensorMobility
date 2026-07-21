# STB-FTT v0.5 Design Principles

1. **Passenger and vehicle are different measures.** Passenger/person flow cannot enter road capacity without an explicit private-vehicle or service-vehicle mapping.
2. **Requests and vehicles are synchronized through capacity.** Use `R x <= S y`; do not encode sharing through a single global occupancy scalar.
3. **Behavior and operation have different costs.** Passenger utility includes waiting, transfer, schedule and fare; vehicle cost includes route time, dispatch, pooling and later deadheading/energy.
4. **Dual prices are interpretable interfaces.** The multiplier on service capacity is a scarcity/service price that can enter passenger choice and vehicle reduced cost.
5. **Private and service vehicles share the physical network.** Road flow is the sum of private access/auto vehicles and generated fleet/transit vehicle routes.
6. **Feasible columns, not anonymous tensors.** Passenger activity chains and vehicle service routes are sparse typed columns; coupling matrices declare target/source orientation.
7. **Two column-generation mechanisms are kept distinct.** Exact shortest-path pricing generates road paths; finite-pool reduced-cost pricing generates compatible vehicle service patterns.
8. **Each layer retains its own conservation audit.** Passenger mass, service capacity, vehicle/link flow and queue mass are checked separately.
9. **Static and dynamic certificates are not mixed.** Static FW gap, vehicle candidate-pool reduced cost, queue conservation and coupled fixed-point residual have different meanings.
10. **The first paper uses a continuous vehicle master.** Integer dispatch, exact matching, fleet inventory and timetable constraints are declared future extensions.
11. **Learning cannot bypass synchronization.** Learned utility residuals may alter passenger scores; they cannot create capacity, delete requests or certify vehicle routing.
12. **External DNL is replaceable.** DTALite/DLSim may replace the internal queue while passenger, vehicle, request and path identifiers remain unchanged.
13. **Connected choice is an operator, not only a diagram.** The direct mode-interaction matrix is a share Jacobian with zero row sums; supply-mediated cooperation/complementarity requires re-solving service capacity and congestion, and must not be inferred from a fixed-supply Logit derivative alone.
