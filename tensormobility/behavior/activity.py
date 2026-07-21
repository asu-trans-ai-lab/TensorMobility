from __future__ import annotations

"""Activity-expanded STB behavioral columns on the Sioux Falls path set.

The implementation stores feasible alternatives as a column table.  It does
not materialize a dense origin x class x activity x destination x mode x time
x route tensor.  Named sparse operators provide the same tensor semantics with
explicit axis orientation and unit contracts.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from tensormobility.dta.sioux_falls import SiouxFallsPathSet
from tensormobility.core.tensor_contracts import Axis, TypedOperator, TypedVector, one_hot_operator

TRAVELER_CLASSES = ("worker", "student", "nonworker")
CLASS_SHARES = np.asarray((0.55, 0.25, 0.20), dtype=float)
ACTIVITIES = ("mandatory", "maintenance", "discretionary")
MODES = ("drive_alone", "shared_ride")


@dataclass(frozen=True)
class BehaviorParameters:
    activity_temperature: float = 1.2
    destination_temperature: float = 1.5
    mode_temperature: float = 1.0
    departure_temperature: float = 1.0
    route_temperature: float = 0.8
    value_of_time: float = 1.0
    destination_prior_weight: float = 2.0
    schedule_penalty_weight: float = 0.45
    safe_residual_bound: float = 0.75


@dataclass(frozen=True)
class ActivitySTBModel:
    path_set: SiouxFallsPathSet
    groups: pd.DataFrame
    columns: pd.DataFrame
    group_axis: Axis
    column_axis: Axis
    physical_path_axis: Axis
    departure_axis: Axis
    path_departure_axis: Axis
    column_to_path: TypedOperator
    column_to_path_departure_person: TypedOperator
    column_to_path_departure_vehicle: TypedOperator
    group_columns: tuple[np.ndarray, ...]
    path_departure_shape: tuple[int, int]
    od_demand: np.ndarray
    path_ff_time: np.ndarray
    n_departures: int

    @property
    def demands(self) -> np.ndarray:
        return self.groups["demand"].to_numpy(float)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def demand_vector(self) -> TypedVector:
        return TypedVector(self.group_axis, self.demands, "person_flow", "persons", "group_demand")


@dataclass(frozen=True)
class BehaviorResult:
    probability: np.ndarray
    person_column_flow: np.ndarray
    vehicle_column_flow: np.ndarray
    path_departure_person: np.ndarray
    path_departure_vehicle: np.ndarray
    mass_residual: float
    shares: pd.DataFrame


def _softmin(cost: np.ndarray, temperature: float) -> tuple[np.ndarray, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    c = np.asarray(cost, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("cost must be a nonempty vector")
    shifted = -(c - np.min(c)) / temperature
    expv = np.exp(np.clip(shifted, -700.0, 0.0))
    prob = expv / expv.sum()
    inclusive_cost = float(np.min(c) - temperature * np.log(expv.sum()))
    return prob, inclusive_cost


def build_activity_stb_model(
    path_set: SiouxFallsPathSet,
    *,
    n_departures: int = 12,
    traveler_classes: tuple[str, ...] = TRAVELER_CLASSES,
    activities: tuple[str, ...] = ACTIVITIES,
    modes: tuple[str, ...] = MODES,
    demand_scale: float = 1.0,
) -> ActivitySTBModel:
    if n_departures < 2:
        raise ValueError("n_departures must be at least two")
    if len(traveler_classes) != len(CLASS_SHARES):
        raise ValueError("default class-share vector requires exactly three traveler classes")

    sf_groups = path_set.instance.groups.reset_index(drop=True)
    sf_columns = path_set.instance.columns.reset_index(drop=True)
    origins = sorted(sf_groups["origin"].unique().tolist())
    od_by_origin = {
        int(origin): sf_groups.index[sf_groups["origin"] == origin].to_numpy(dtype=int)
        for origin in origins
    }
    origin_total = sf_groups.groupby("origin")["demand"].sum().to_dict()

    group_rows: list[dict[str, object]] = []
    for origin in origins:
        total = float(origin_total[origin]) * demand_scale
        for class_index, traveler_class in enumerate(traveler_classes):
            g = len(group_rows)
            group_rows.append({
                "group_index": g,
                "group_id": f"o{origin}:{traveler_class}",
                "origin": int(origin),
                "traveler_class": traveler_class,
                "class_index": class_index,
                "demand": total * float(CLASS_SHARES[class_index]),
            })
    groups = pd.DataFrame(group_rows)

    # Pre-index the four physical paths belonging to each OD group.
    paths_by_od = {
        od: sf_columns.index[sf_columns["group_index"] == od].to_numpy(dtype=int)
        for od in range(len(sf_groups))
    }
    path_ff = sf_columns["free_flow_path_time"].to_numpy(float)

    column_rows: list[dict[str, object]] = []
    for group in group_rows:
        origin = int(group["origin"])
        g = int(group["group_index"])
        for activity_index, activity in enumerate(activities):
            for od_index in od_by_origin[origin]:
                destination = int(sf_groups.iloc[od_index]["destination"])
                od_prior = float(sf_groups.iloc[od_index]["demand"]) / float(origin_total[origin])
                for mode_index, mode in enumerate(modes):
                    vehicle_factor = 1.0 if mode == "drive_alone" else 0.60
                    for departure_index in range(n_departures):
                        for physical_path_index in paths_by_od[int(od_index)]:
                            route_rank = int(sf_columns.iloc[physical_path_index]["route_rank"])
                            c = len(column_rows)
                            column_rows.append({
                                "column_index": c,
                                "column_id": (
                                    f"g{g}:a{activity_index}:od{od_index}:m{mode_index}:"
                                    f"t{departure_index}:p{physical_path_index}"
                                ),
                                "group_index": g,
                                "origin": origin,
                                "traveler_class": group["traveler_class"],
                                "class_index": int(group["class_index"]),
                                "activity": activity,
                                "activity_index": activity_index,
                                "od_index": int(od_index),
                                "destination": destination,
                                "destination_prior": od_prior,
                                "mode": mode,
                                "mode_index": mode_index,
                                "vehicle_factor": vehicle_factor,
                                "departure_index": departure_index,
                                "physical_path_index": int(physical_path_index),
                                "route_rank": route_rank,
                                "free_flow_path_time": float(path_ff[physical_path_index]),
                            })
    columns = pd.DataFrame(column_rows)

    group_axis = Axis("traveler_group", tuple(groups["group_id"].astype(str)), "origin x traveler class")
    column_axis = Axis("behavioral_super_column", tuple(columns["column_id"].astype(str)), "activity-destination-mode-departure-route")
    physical_path_axis = Axis("physical_path", tuple(sf_columns["column_id"].astype(str)), "Sioux Falls route columns")
    departure_axis = Axis("departure_interval", tuple(f"t:{i}" for i in range(n_departures)), "departure time bins")
    path_departure_axis = Axis(
        "path_departure",
        tuple(f"p:{p}|t:{t}" for p in range(len(sf_columns)) for t in range(n_departures)),
        "physical path x departure interval",
    )

    path_index = columns["physical_path_index"].to_numpy(int)
    dep_index = columns["departure_index"].to_numpy(int)
    path_dep_index = path_index * n_departures + dep_index
    vehicle_factor = columns["vehicle_factor"].to_numpy(float)

    column_to_path = one_hot_operator(
        name="column_to_physical_path",
        source=column_axis,
        target=physical_path_axis,
        target_index_for_source=path_index,
        input_measure="person_flow",
        output_measure="person_flow",
        conservation="mass_preserving",
        description="Drops behavioral attributes while preserving person mass.",
    )
    column_to_path_departure_person = one_hot_operator(
        name="column_to_path_departure_person",
        source=column_axis,
        target=path_departure_axis,
        target_index_for_source=path_dep_index,
        input_measure="person_flow",
        output_measure="person_flow",
        conservation="mass_preserving",
        description="Aggregates behavioral columns to path-departure person flows.",
    )
    column_to_path_departure_vehicle = one_hot_operator(
        name="column_to_path_departure_vehicle",
        source=column_axis,
        target=path_departure_axis,
        target_index_for_source=path_dep_index,
        weights=vehicle_factor,
        input_measure="person_flow",
        output_measure="vehicle_flow",
        conservation="none",
        description="Converts persons to vehicles using explicit mode occupancy factors.",
    )

    cg = columns["group_index"].to_numpy(int)
    group_columns = tuple(np.flatnonzero(cg == g) for g in range(len(groups)))

    return ActivitySTBModel(
        path_set=path_set,
        groups=groups,
        columns=columns,
        group_axis=group_axis,
        column_axis=column_axis,
        physical_path_axis=physical_path_axis,
        departure_axis=departure_axis,
        path_departure_axis=path_departure_axis,
        column_to_path=column_to_path,
        column_to_path_departure_person=column_to_path_departure_person,
        column_to_path_departure_vehicle=column_to_path_departure_vehicle,
        group_columns=group_columns,
        path_departure_shape=(len(sf_columns), n_departures),
        od_demand=sf_groups["demand"].to_numpy(float),
        path_ff_time=path_ff,
        n_departures=n_departures,
    )


def _activity_base_cost(class_index: int, activity_index: int) -> float:
    # Costs, not utilities: lower is preferred.  These synthetic values are
    # deliberately interpretable and are not presented as calibrated behavior.
    table = np.asarray([
        [0.0, 1.1, 1.5],   # worker
        [0.3, 0.8, 1.2],   # student
        [1.6, 0.2, 0.6],   # nonworker
    ])
    return float(table[class_index, activity_index])


def _mode_base_cost(class_index: int, activity_index: int, mode_index: int) -> float:
    if mode_index == 0:  # drive alone
        return float((0.05, 0.65, 0.35)[class_index] + 0.10 * activity_index)
    return float((0.55, 0.15, 0.05)[class_index] + 0.05 * max(0, 1 - activity_index))


def _schedule_cost(activity_index: int, departure_index: int, n_departures: int, weight: float) -> float:
    targets = np.asarray((0.28, 0.53, 0.72)) * (n_departures - 1)
    scale = max(n_departures / 6.0, 1.0)
    return float(weight * abs(departure_index - targets[activity_index]) / scale)


def hierarchical_behavior_choice(
    model: ActivitySTBModel,
    path_time: np.ndarray | None = None,
    *,
    parameters: BehaviorParameters = BehaviorParameters(),
    safe_column_residual: np.ndarray | None = None,
) -> BehaviorResult:
    """Compute a recursive-logit choice over activity->destination->mode->time->route.

    ``safe_column_residual`` is an exogenous bounded generalized-cost residual.
    Because it does not depend on current flow or queue state, it does not alter
    monotonicity of the physical cost mapping.  Flow-dependent attention is not
    part of the certified core.
    """
    n_paths, n_dep = model.path_departure_shape
    if path_time is None:
        path_time = np.repeat(model.path_ff_time[:, None], n_dep, axis=1)
    path_time = np.asarray(path_time, dtype=float)
    if path_time.shape != (n_paths, n_dep):
        raise ValueError(f"path_time has shape {path_time.shape}, expected {(n_paths, n_dep)}")
    if not np.all(np.isfinite(path_time)) or np.any(path_time < 0):
        raise ValueError("path_time must be finite and nonnegative")

    residual = np.zeros(model.n_columns, dtype=float)
    if safe_column_residual is not None:
        residual = np.asarray(safe_column_residual, dtype=float)
        if residual.shape != (model.n_columns,):
            raise ValueError("safe_column_residual has wrong shape")
        if np.max(np.abs(residual)) > parameters.safe_residual_bound + 1e-12:
            raise ValueError("safe_column_residual exceeds declared bound")

    probability = np.zeros(model.n_columns, dtype=float)
    cols = model.columns

    for g, group_idx in enumerate(model.group_columns):
        group_cols = cols.iloc[group_idx]
        class_index = int(group_cols.iloc[0]["class_index"])
        activity_ids = sorted(group_cols["activity_index"].unique().tolist())
        activity_conditional: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
        activity_costs: list[float] = []

        for a in activity_ids:
            idx_a_local = np.flatnonzero(group_cols["activity_index"].to_numpy(int) == a)
            cols_a = group_cols.iloc[idx_a_local]
            od_ids = sorted(cols_a["od_index"].unique().tolist())
            destination_costs: list[float] = []
            destination_struct: list[tuple[int, np.ndarray, np.ndarray, list[object]]] = []

            for od in od_ids:
                idx_d_local = idx_a_local[np.flatnonzero(cols_a["od_index"].to_numpy(int) == od)]
                cols_d = group_cols.iloc[idx_d_local]
                mode_ids = sorted(cols_d["mode_index"].unique().tolist())
                mode_costs: list[float] = []
                mode_struct: list[tuple[int, np.ndarray, np.ndarray, list[object]]] = []

                for m in mode_ids:
                    idx_m_local = idx_d_local[np.flatnonzero(cols_d["mode_index"].to_numpy(int) == m)]
                    cols_m = group_cols.iloc[idx_m_local]
                    dep_ids = sorted(cols_m["departure_index"].unique().tolist())
                    dep_costs: list[float] = []
                    dep_struct: list[tuple[int, np.ndarray, np.ndarray]] = []

                    for dep in dep_ids:
                        idx_t_local = idx_m_local[np.flatnonzero(cols_m["departure_index"].to_numpy(int) == dep)]
                        leaf_global = group_idx[idx_t_local]
                        leaf = cols.iloc[leaf_global]
                        pidx = leaf["physical_path_index"].to_numpy(int)
                        route_cost = (
                            parameters.value_of_time * path_time[pidx, dep]
                            + residual[leaf_global]
                        )
                        route_prob, route_inclusive = _softmin(route_cost, parameters.route_temperature)
                        total_dep_cost = route_inclusive + _schedule_cost(
                            int(a), int(dep), model.n_departures, parameters.schedule_penalty_weight
                        )
                        dep_costs.append(total_dep_cost)
                        dep_struct.append((int(dep), leaf_global, route_prob))

                    dep_prob, dep_inclusive = _softmin(np.asarray(dep_costs), parameters.departure_temperature)
                    total_mode_cost = dep_inclusive + _mode_base_cost(class_index, int(a), int(m))
                    mode_costs.append(total_mode_cost)
                    mode_struct.append((int(m), np.asarray(dep_ids, int), dep_prob, dep_struct))

                mode_prob, mode_inclusive = _softmin(np.asarray(mode_costs), parameters.mode_temperature)
                prior = float(cols_d.iloc[0]["destination_prior"])
                destination_prior_cost = -parameters.destination_prior_weight * np.log(max(prior, 1e-12))
                destination_costs.append(mode_inclusive + destination_prior_cost)
                destination_struct.append((int(od), np.asarray(mode_ids, int), mode_prob, mode_struct))

            destination_prob, destination_inclusive = _softmin(
                np.asarray(destination_costs), parameters.destination_temperature
            )
            activity_costs.append(destination_inclusive + _activity_base_cost(class_index, int(a)))
            activity_conditional[int(a)] = (
                np.asarray(od_ids, int), destination_prob, np.asarray(destination_struct, dtype=object)
            )

        activity_prob, _ = _softmin(np.asarray(activity_costs), parameters.activity_temperature)

        for a_pos, a in enumerate(activity_ids):
            od_ids, dest_prob, dest_struct_arr = activity_conditional[int(a)]
            for d_pos, _od in enumerate(od_ids):
                _, mode_ids, mode_prob, mode_struct = dest_struct_arr[d_pos]
                for m_pos, _m in enumerate(mode_ids):
                    _, dep_ids, dep_prob, dep_struct = mode_struct[m_pos]
                    for t_pos, _dep in enumerate(dep_ids):
                        _, leaf_global, route_prob = dep_struct[t_pos]
                        probability[leaf_global] = (
                            activity_prob[a_pos]
                            * dest_prob[d_pos]
                            * mode_prob[m_pos]
                            * dep_prob[t_pos]
                            * route_prob
                        )

    person_flow = probability * np.repeat(model.demands, [len(idx) for idx in model.group_columns])
    # The repeat above relies on columns being group-contiguous by construction.
    if person_flow.shape != (model.n_columns,):
        raise RuntimeError("Internal group-contiguity assumption failed")
    vehicle_factor = model.columns["vehicle_factor"].to_numpy(float)
    vehicle_flow = person_flow * vehicle_factor

    person_vector = TypedVector(model.column_axis, person_flow, "person_flow", "persons", "behavioral_column_person_flow")
    path_dep_person = model.column_to_path_departure_person.apply(
        person_vector, output_name="path_departure_person_flow", unit="persons"
    ).values.reshape(model.path_departure_shape)
    path_dep_vehicle = model.column_to_path_departure_vehicle.apply(
        person_vector, output_name="path_departure_vehicle_flow", unit="vehicles"
    ).values.reshape(model.path_departure_shape)

    group_mass = np.asarray([person_flow[idx].sum() for idx in model.group_columns])
    mass_residual = float(np.max(np.abs(group_mass - model.demands)))

    share_rows: list[dict[str, object]] = []
    total = max(float(person_flow.sum()), 1.0)
    for field in ("traveler_class", "activity", "destination", "mode", "departure_index", "route_rank"):
        aggregation = pd.DataFrame({field: model.columns[field], "flow": person_flow}).groupby(field, as_index=False)["flow"].sum()
        for row in aggregation.itertuples(index=False):
            share_rows.append({"dimension": field, "alternative": str(row[0]), "flow": float(row[1]), "share": float(row[1]) / total})

    return BehaviorResult(
        probability=probability,
        person_column_flow=person_flow,
        vehicle_column_flow=vehicle_flow,
        path_departure_person=path_dep_person,
        path_departure_vehicle=path_dep_vehicle,
        mass_residual=mass_residual,
        shares=pd.DataFrame(share_rows),
    )
