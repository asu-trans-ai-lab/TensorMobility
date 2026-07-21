from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd


PERIODS = ("AM", "MD", "PM")
DESTINATIONS = ("cbd", "suburb")


@dataclass(frozen=True)
class STBInstance:
    """A finite Space-Time-Behavior super-column assignment instance.

    Rows of A are link-period resources; columns are complete behavioral
    alternatives (destination, mode, departure period, physical route).
    """

    name: str
    columns: pd.DataFrame
    groups: pd.DataFrame
    A: np.ndarray
    resource_labels: tuple[str, ...]
    free_flow_time: np.ndarray
    capacity: np.ndarray
    behavior_cost: np.ndarray
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0

    def __post_init__(self) -> None:
        p = len(self.columns)
        r = len(self.resource_labels)
        if self.A.shape != (r, p):
            raise ValueError(f"A has shape {self.A.shape}, expected {(r, p)}")
        if len(self.free_flow_time) != r or len(self.capacity) != r:
            raise ValueError("Resource vectors do not match A rows")
        if len(self.behavior_cost) != p:
            raise ValueError("behavior_cost does not match columns")
        if np.any(self.capacity <= 0):
            raise ValueError("All capacities must be positive")

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def n_resources(self) -> int:
        return len(self.resource_labels)

    @property
    def demands(self) -> np.ndarray:
        return self.groups["demand"].to_numpy(dtype=float)

    @property
    def column_group(self) -> np.ndarray:
        return self.columns["group_index"].to_numpy(dtype=int)

    @property
    def group_columns(self) -> list[np.ndarray]:
        cg = self.column_group
        return [np.flatnonzero(cg == g) for g in range(self.n_groups)]

    @property
    def free_flow_column_time(self) -> np.ndarray:
        return self.A.T @ self.free_flow_time

    def with_capacity_multiplier(
        self,
        multiplier: np.ndarray | float,
        name: str | None = None,
    ) -> "STBInstance":
        mult = np.asarray(multiplier, dtype=float)
        if mult.ndim == 0:
            mult = np.full(self.n_resources, float(mult))
        if mult.shape != (self.n_resources,):
            raise ValueError("capacity multiplier has the wrong shape")
        return replace(
            self,
            name=name or f"{self.name}_capacity_changed",
            capacity=self.capacity * mult,
        )


def _period_distance(a: str, b: str) -> int:
    return abs(PERIODS.index(a) - PERIODS.index(b))


def build_toy_instance(seed: int = 7) -> STBInstance:
    """Build a reproducible medium-small STB instance with 288 columns.

    The instance has 8 heterogeneous demand groups, 3 departure periods,
    2 destinations, car/transit mode choice, and five car paths per
    destination. Car paths load link-period rows; transit alternatives carry
    fixed generalized cost but no road-link load.
    """

    rng = np.random.default_rng(seed)

    link_specs = [
        ("access_north", 3.0, 520.0),
        ("central_freeway", 5.0, 245.0),
        ("cbd_approach", 4.0, 205.0),
        ("east_connector", 4.0, 300.0),
        ("ring_road", 6.0, 470.0),
        ("access_south", 5.0, 390.0),
        ("cbd_local", 8.0, 285.0),
        ("suburb_approach", 4.0, 270.0),
        ("south_arterial", 6.0, 335.0),
        ("suburb_local", 7.0, 300.0),
    ]
    link_index = {name: i for i, (name, _, _) in enumerate(link_specs)}

    path_specs: dict[str, list[tuple[str, tuple[str, ...], float]]] = {
        "cbd": [
            ("direct", ("access_north", "central_freeway", "cbd_approach"), 2.0),
            ("ring", ("access_north", "east_connector", "ring_road", "cbd_approach"), 0.5),
            ("local", ("access_south", "cbd_local", "cbd_approach"), 0.0),
            ("hybrid", ("access_north", "east_connector", "cbd_local", "cbd_approach"), 1.0),
            ("bypass", ("south_arterial", "ring_road", "cbd_approach"), 0.0),
        ],
        "suburb": [
            ("direct", ("access_north", "east_connector", "suburb_approach"), 1.0),
            ("ring", ("access_south", "ring_road", "suburb_approach"), 0.2),
            ("local", ("south_arterial", "suburb_local", "suburb_approach"), 0.0),
            ("central", ("access_north", "central_freeway", "ring_road", "suburb_approach"), 1.4),
            ("bypass", ("south_arterial", "ring_road", "suburb_approach"), 0.0),
        ],
    }

    groups = pd.DataFrame(
        [
            ("cbd_commuter_car", 420.0, "cbd", "AM", 1, 0, -2.5, 0.0),
            ("cbd_commuter_transit", 300.0, "cbd", "AM", 0, 1, 4.0, -3.5),
            ("flex_worker", 280.0, "cbd", "MD", 1, 1, -0.5, -0.5),
            ("suburb_shopper_car", 260.0, "suburb", "MD", 1, 0, -2.0, 1.0),
            ("suburb_shopper_no_car", 220.0, "suburb", "MD", 0, 1, 4.5, -3.0),
            ("cbd_evening_worker", 260.0, "cbd", "PM", 1, 1, -1.0, -1.0),
            ("suburb_commuter", 360.0, "suburb", "AM", 1, 0, -2.3, 0.8),
            ("mixed_flexible", 300.0, "suburb", "PM", 1, 1, -0.4, -0.8),
        ],
        columns=[
            "group_id",
            "demand",
            "preferred_destination",
            "target_period",
            "car_available",
            "transit_pass",
            "car_bias",
            "transit_bias",
        ],
    )
    groups.insert(0, "group_index", np.arange(len(groups), dtype=int))

    period_capacity_factor = {"AM": 0.88, "MD": 1.18, "PM": 0.94}
    period_time_factor = {"AM": 1.03, "MD": 0.96, "PM": 1.01}

    resource_labels: list[str] = []
    free_flow: list[float] = []
    capacity: list[float] = []
    resource_index: dict[tuple[str, str], int] = {}
    for period in PERIODS:
        for link_name, t0, cap in link_specs:
            resource_index[(period, link_name)] = len(resource_labels)
            resource_labels.append(f"{period}:{link_name}")
            free_flow.append(t0 * period_time_factor[period])
            capacity.append(cap * period_capacity_factor[period])

    rows: list[dict[str, object]] = []
    incidence_columns: list[np.ndarray] = []
    behavior_costs: list[float] = []

    transit_time = {
        ("cbd", "AM"): 24.0,
        ("cbd", "MD"): 28.0,
        ("cbd", "PM"): 26.0,
        ("suburb", "AM"): 30.0,
        ("suburb", "MD"): 27.0,
        ("suburb", "PM"): 31.0,
    }

    column_id = 0
    for _, group in groups.iterrows():
        g = int(group["group_index"])
        for destination in DESTINATIONS:
            destination_penalty = 0.0 if destination == group["preferred_destination"] else 8.0
            for period in PERIODS:
                schedule_penalty = 4.5 * _period_distance(period, str(group["target_period"]))

                if int(group["car_available"]) == 1:
                    parking = 7.5 if destination == "cbd" else 2.5
                    for route_rank, (route_name, links, toll) in enumerate(path_specs[destination]):
                        a = np.zeros(len(resource_labels), dtype=float)
                        for link in links:
                            a[resource_index[(period, link)]] = 1.0
                        jitter = float(rng.normal(0.0, 0.08))
                        linear_cost = (
                            destination_penalty
                            + schedule_penalty
                            + parking
                            + toll
                            + float(group["car_bias"])
                            + jitter
                        )
                        rows.append(
                            {
                                "column_index": column_id,
                                "column_id": f"g{g}:{destination}:car:{period}:{route_name}",
                                "group_index": g,
                                "group_id": group["group_id"],
                                "destination": destination,
                                "mode": "car",
                                "period": period,
                                "route": route_name,
                                "route_rank": route_rank,
                                "preferred_destination_match": int(destination == group["preferred_destination"]),
                                "target_period_match": int(period == group["target_period"]),
                                "car_available": int(group["car_available"]),
                                "transit_pass": int(group["transit_pass"]),
                                "path_links": "|".join(links),
                            }
                        )
                        incidence_columns.append(a)
                        behavior_costs.append(linear_cost)
                        column_id += 1

                a = np.zeros(len(resource_labels), dtype=float)
                transfer_penalty = 1.5 if destination == "suburb" else 0.5
                linear_cost = (
                    destination_penalty
                    + schedule_penalty
                    + transit_time[(destination, period)]
                    + transfer_penalty
                    + float(group["transit_bias"])
                )
                rows.append(
                    {
                        "column_index": column_id,
                        "column_id": f"g{g}:{destination}:transit:{period}:transit",
                        "group_index": g,
                        "group_id": group["group_id"],
                        "destination": destination,
                        "mode": "transit",
                        "period": period,
                        "route": "transit",
                        "route_rank": -1,
                        "preferred_destination_match": int(destination == group["preferred_destination"]),
                        "target_period_match": int(period == group["target_period"]),
                        "car_available": int(group["car_available"]),
                        "transit_pass": int(group["transit_pass"]),
                        "path_links": "",
                    }
                )
                incidence_columns.append(a)
                behavior_costs.append(linear_cost)
                column_id += 1

    columns = pd.DataFrame(rows)
    A = np.stack(incidence_columns, axis=1)
    columns["free_flow_path_time"] = A.T @ np.asarray(free_flow)
    columns["linear_behavior_cost"] = np.asarray(behavior_costs)
    columns["total_free_flow_cost"] = columns["free_flow_path_time"] + columns["linear_behavior_cost"]

    return STBInstance(
        name="stb_toy_base",
        columns=columns,
        groups=groups,
        A=A,
        resource_labels=tuple(resource_labels),
        free_flow_time=np.asarray(free_flow, dtype=float),
        capacity=np.asarray(capacity, dtype=float),
        behavior_cost=np.asarray(behavior_costs, dtype=float),
    )


def capacity_shock_multiplier(instance: STBInstance) -> np.ndarray:
    """A targeted shock that makes previously minor ring/bypass paths valuable."""

    mult = np.ones(instance.n_resources, dtype=float)
    for i, label in enumerate(instance.resource_labels):
        if "central_freeway" in label or "cbd_approach" in label:
            mult[i] *= 0.48
        if label.startswith("AM:") and "east_connector" in label:
            mult[i] *= 0.72
        if "ring_road" in label:
            mult[i] *= 1.08
    return mult
