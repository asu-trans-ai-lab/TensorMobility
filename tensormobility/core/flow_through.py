from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensormobility.core.instance import STBInstance


@dataclass(frozen=True)
class FlowThroughResult:
    zone_generation: np.ndarray
    od_flow: np.ndarray
    path_flow: np.ndarray
    link_flow: np.ndarray
    zone_to_od: np.ndarray
    od_to_path: np.ndarray
    path_to_link: np.ndarray
    mass_residual_zone_od: float
    mass_residual_od_path: float
    link_reconstruction_residual: float


def build_flow_through_tensors(instance: STBInstance, path_flow: np.ndarray) -> FlowThroughResult:
    """Build the typed zone->OD->path->link flow-through operators.

    This is a modern, explicit version of the uploaded BTCG connection chain.
    Probabilities are row-normalized within their feasible incidence masks;
    real demand mass is propagated by matrix multiplication.
    """
    f = np.asarray(path_flow, dtype=float)
    if f.shape != (instance.n_columns,):
        raise ValueError("path_flow has wrong shape")
    if np.min(f) < -1e-9:
        raise ValueError("path_flow must be nonnegative")

    if "network_n_zones" in instance.groups:
        n_declared = int(instance.groups["network_n_zones"].iloc[0])
        origins = list(range(1, n_declared + 1))
    elif "origin" in instance.groups and "destination" in instance.groups:
        zone_values = set(instance.groups["origin"].tolist()) | set(instance.groups["destination"].tolist())
        if zone_values and all(isinstance(z, (int, np.integer)) for z in zone_values):
            origins = list(range(1, int(max(zone_values)) + 1))
        else:
            origins = sorted(zone_values)
    elif "origin" in instance.groups:
        origins = sorted(instance.groups["origin"].unique().tolist())
    else:
        origins = list(range(instance.n_groups))
    origin_index = {origin: i for i, origin in enumerate(origins)}
    n_zones = len(origins)
    n_od = instance.n_groups
    n_paths = instance.n_columns

    zone_to_od = np.zeros((n_zones, n_od), dtype=float)
    od_flow = np.zeros(n_od, dtype=float)
    for g, idx in enumerate(instance.group_columns):
        od_flow[g] = f[idx].sum()
        origin = instance.groups.iloc[g]["origin"] if "origin" in instance.groups else g
        zone_to_od[origin_index[origin], g] = od_flow[g]
    zone_generation = zone_to_od.sum(axis=1)
    for z in range(n_zones):
        if zone_generation[z] > 0:
            zone_to_od[z] /= zone_generation[z]

    od_to_path = np.zeros((n_od, n_paths), dtype=float)
    for g, idx in enumerate(instance.group_columns):
        if od_flow[g] > 0:
            od_to_path[g, idx] = f[idx] / od_flow[g]

    recovered_od = zone_to_od.T @ zone_generation
    recovered_path = od_to_path.T @ recovered_od
    link_flow = instance.A @ recovered_path
    direct_link = instance.A @ f

    return FlowThroughResult(
        zone_generation=zone_generation,
        od_flow=recovered_od,
        path_flow=recovered_path,
        link_flow=link_flow,
        zone_to_od=zone_to_od,
        od_to_path=od_to_path,
        path_to_link=instance.A.copy(),
        mass_residual_zone_od=float(np.max(np.abs(recovered_od - od_flow))),
        mass_residual_od_path=float(np.max(np.abs(recovered_path - f))),
        link_reconstruction_residual=float(np.max(np.abs(link_flow - direct_link))),
    )
