from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tensormobility.dta.algorithms import AlgorithmResult, solve_exact_fw, solve_logit_sue_msa
from tensormobility.core.flow_through import FlowThroughResult, build_flow_through_tensors
from tensormobility.dynamics.fluid_queue import FluidQueueResult, build_departure_tensor, run_fluid_point_queue, smooth_departure_profile
from tensormobility.dta.sioux_falls import SiouxFallsPathSet, build_sioux_falls_path_set


class BehaviorOperator(str, Enum):
    LOGIT = "logit"
    DETERMINISTIC_UE = "deterministic_ue"
    FIXED = "fixed"


class TimeOperator(str, Enum):
    STATIC_IDENTITY = "static_identity"
    FLUID_POINT_QUEUE = "fluid_point_queue"


@dataclass(frozen=True)
class STBSpecialCase:
    name: str
    behavior: BehaviorOperator
    time: TimeOperator
    description: str


CASE_1_LOGIT_UE = STBSpecialCase(
    name="case_1_logit_sue_to_ue",
    behavior=BehaviorOperator.LOGIT,
    time=TimeOperator.STATIC_IDENTITY,
    description="Static path/link incidence + BPR costs + logit fixed point; temperature->0 gives deterministic UE.",
)
CASE_2_FLUID_QUEUE = STBSpecialCase(
    name="case_2_fluid_queue",
    behavior=BehaviorOperator.FIXED,
    time=TimeOperator.FLUID_POINT_QUEUE,
    description="Fixed path/departure flows propagated through path-resolved fluid point queues.",
)
CASE_3_FLOW_THROUGH = STBSpecialCase(
    name="case_3_flow_through_tensor",
    behavior=BehaviorOperator.FIXED,
    time=TimeOperator.STATIC_IDENTITY,
    description="Zone->OD->path->link typed probability/flow tensors with exact mass reconstruction.",
)


def solve_case_1(path_set: SiouxFallsPathSet, temperature: float = 1.0) -> tuple[AlgorithmResult, AlgorithmResult]:
    logit = solve_logit_sue_msa(path_set.instance, temperature=temperature, max_iterations=1000, tolerance=1e-7)
    ue = solve_exact_fw(path_set.instance, max_iterations=900, tolerance=1e-7)
    return logit, ue


def solve_case_2(
    path_set: SiouxFallsPathSet,
    path_flow: np.ndarray,
    capacity_multiplier: np.ndarray | float = 1.0,
) -> FluidQueueResult:
    profile = smooth_departure_profile()
    departures = build_departure_tensor(path_flow, profile)
    return run_fluid_point_queue(path_set, departures, capacity_multiplier=capacity_multiplier)


def solve_case_3(path_set: SiouxFallsPathSet, path_flow: np.ndarray) -> FlowThroughResult:
    return build_flow_through_tensors(path_set.instance, path_flow)


def build_default_special_case_network() -> SiouxFallsPathSet:
    return build_sioux_falls_path_set(k_paths=4, demand_scale=1.0)
