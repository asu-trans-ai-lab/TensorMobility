from __future__ import annotations

"""Unified GMNS/TAPLite-style adapters for STB path, OD, link, and time flows.

The adapter accepts a small set of field aliases because existing GMNS-derived
repositories are not perfectly uniform.  Internally, every table is converted
to one canonical schema.  Exports are plain CSV and can serve as the exchange
boundary to TAPLite, DTALite/DLSim, or an agency-specific backend.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from tensormobility.dta.column_generation import ColumnGenerationResult
from tensormobility.core.network_core import StaticNetwork, build_static_network
from tensormobility.dynamics.path_queue import SparsePathQueueResult


@dataclass(frozen=True)
class GMNSBundle:
    network: StaticNetwork
    demand: pd.DataFrame
    paths: pd.DataFrame | None
    departure_profiles: pd.DataFrame | None
    source_folder: Path


def _read_first(folder: Path, names: tuple[str, ...], required: bool = True) -> pd.DataFrame | None:
    for name in names:
        path = folder / name
        if path.exists():
            return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"none of {names} found in {folder}")
    return None


def _first_column(df: pd.DataFrame, aliases: tuple[str, ...], *, required: bool = True, default: float | str | None = None) -> pd.Series:
    lookup = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return df[lookup[alias.lower()]]
    if required:
        raise ValueError(f"missing required field; expected one of {aliases}")
    return pd.Series([default] * len(df), index=df.index)


def read_gmns_bundle(folder: str | Path) -> GMNSBundle:
    root = Path(folder)
    node_raw = _read_first(root, ("node.csv", "nodes.csv"))
    link_raw = _read_first(root, ("link.csv", "road_link.csv", "links.csv"))
    demand_raw = _read_first(root, ("demand.csv", "od.csv", "input_demand.csv"))
    assert node_raw is not None and link_raw is not None and demand_raw is not None

    nodes = pd.DataFrame({
        "node_id": _first_column(node_raw, ("node_id", "id")).astype(int),
        "x_coord": _first_column(node_raw, ("x_coord", "x", "longitude", "lon"), required=False, default=np.nan).astype(float),
        "y_coord": _first_column(node_raw, ("y_coord", "y", "latitude", "lat"), required=False, default=np.nan).astype(float),
    })
    if "zone_id" in node_raw.columns:
        nodes["zone_id"] = node_raw["zone_id"]

    from_node = _first_column(link_raw, ("from_node_id", "from_node", "o_node_id", "tail", "from"))
    to_node = _first_column(link_raw, ("to_node_id", "to_node", "d_node_id", "head", "to"))
    link_id = _first_column(link_raw, ("link_id", "id"), required=False, default=None)
    if link_id.isna().all() or (link_id.astype(str) == "None").all():
        link_id = pd.Series([f"link_{i}" for i in range(len(link_raw))])

    ff = None
    for aliases in (
        ("free_flow_time", "free_flow_travel_time", "fftt", "travel_time"),
        ("time_peroid",),
    ):
        try:
            ff = _first_column(link_raw, aliases)
            break
        except ValueError:
            pass
    if ff is None:
        length = _first_column(link_raw, ("length", "length_mile", "length_km")).astype(float)
        speed = _first_column(link_raw, ("free_speed", "speed_limit", "ffspeed")).astype(float)
        ff = 60.0 * length / np.maximum(speed, 1e-6)

    capacity = _first_column(
        link_raw,
        ("capacity", "lane_capacity", "capacity_vph", "hourly_capacity"),
        required=False,
        default=1800.0,
    ).astype(float)
    if "lanes" in link_raw.columns and np.nanmedian(capacity.to_numpy(float)) <= 3000:
        # Do not automatically multiply if the supplied field already appears
        # to be total capacity.  A manifest records the unmodified values.
        pass

    links = pd.DataFrame({
        "link_id": link_id.astype(str),
        "from_node_id": from_node.astype(int),
        "to_node_id": to_node.astype(int),
        "free_flow_time": pd.Series(ff).astype(float),
        "capacity": capacity,
    })
    for field in ("length", "lanes", "facility_type", "geometry"):
        if field in link_raw.columns:
            links[field] = link_raw[field]

    demand = pd.DataFrame({
        "od_id": _first_column(demand_raw, ("od_id", "demand_id"), required=False, default=None).astype(str),
        "origin_node_id": _first_column(demand_raw, ("origin_node_id", "o_zone_id", "origin", "from_zone_id")).astype(int),
        "destination_node_id": _first_column(demand_raw, ("destination_node_id", "d_zone_id", "destination", "to_zone_id")).astype(int),
        "volume": _first_column(demand_raw, ("volume", "demand", "od_flow", "person_volume", "vehicle_volume")).astype(float),
    })
    missing_od = demand["od_id"].isin(("None", "nan", ""))
    demand.loc[missing_od, "od_id"] = [f"od_{i}" for i in demand.index[missing_od]]

    network = build_static_network(root.name or "gmns_network", nodes, links)
    paths = _read_first(root, ("path.csv", "route.csv", "output_path.csv"), required=False)
    profiles = _read_first(root, ("departure_profile.csv", "departure_time_profile.csv", "demand_period.csv"), required=False)
    return GMNSBundle(network=network, demand=demand, paths=paths, departure_profiles=profiles, source_folder=root)


def write_gmns_case(network: StaticNetwork, demand: pd.DataFrame, folder: str | Path) -> Path:
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    network.nodes.to_csv(root / "node.csv", index=False)
    network.links.to_csv(root / "link.csv", index=False)
    demand.to_csv(root / "demand.csv", index=False)
    manifest = {
        "network_name": network.name,
        "node_count": network.n_nodes,
        "link_count": network.n_links,
        "od_count": len(demand),
        "orientation": "link_by_path for incidence; target <- source for all typed operators",
    }
    (root / "gmns_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def export_unified_assignment(
    network: StaticNetwork,
    demand: pd.DataFrame,
    result: ColumnGenerationResult,
    folder: str | Path,
    *,
    departure_profile: np.ndarray | None = None,
    queue: SparsePathQueueResult | None = None,
) -> Path:
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)

    od_out = demand[["od_id", "origin_node_id", "destination_node_id", "volume"]].copy()
    od_out = od_out.rename(columns={"volume": "od_flow"})
    od_out.to_csv(root / "od_flow.csv", index=False)

    result.path_table.to_csv(root / "path_flow.csv", index=False)
    route_cols = [
        "route_id", "path_id", "od_id", "origin_node_id", "destination_node_id",
        "path_flow", "path_cost", "node_sequence", "link_sequence",
    ]
    result.path_table[route_cols].to_csv(root / "route_flow.csv", index=False)

    link = network.links.copy()
    link["link_flow"] = result.link_flow
    link["link_travel_time"] = result.link_cost
    link["volume_capacity_ratio"] = result.link_flow / np.maximum(network.capacity, 1e-12)
    link.to_csv(root / "link_flow.csv", index=False)

    if departure_profile is not None:
        profile = np.asarray(departure_profile, dtype=float)
        if profile.ndim != 1 or np.any(profile < 0) or not np.isclose(profile.sum(), 1.0):
            raise ValueError("departure_profile must be nonnegative and sum to one")
        pd.DataFrame({
            "departure_interval": np.arange(profile.size),
            "share": profile,
        }).to_csv(root / "departure_profile.csv", index=False)
        path_time_rows: list[dict[str, object]] = []
        for path_row in result.path_table.itertuples(index=False):
            for t, share in enumerate(profile):
                path_time_rows.append({
                    "path_id": path_row.path_id,
                    "od_id": path_row.od_id,
                    "departure_interval": t,
                    "path_departure_flow": float(path_row.path_flow) * float(share),
                })
        pd.DataFrame(path_time_rows).to_csv(root / "path_departure_flow.csv", index=False)

    if queue is not None:
        link_time_rows: list[dict[str, object]] = []
        for a, link_row in network.links.iterrows():
            for t in range(queue.queue_by_link_time.shape[1]):
                link_time_rows.append({
                    "link_id": link_row["link_id"],
                    "time_step": t,
                    "minutes": t * queue.dt_minutes,
                    "inflow": queue.inflow_by_link_time[a, t],
                    "outflow": queue.outflow_by_link_time[a, t],
                    "queue": queue.queue_by_link_time[a, t],
                })
        pd.DataFrame(link_time_rows).to_csv(root / "link_time_flow.csv", index=False)
        queue.history.to_csv(root / "network_queue_summary.csv", index=False)

    result.history.to_csv(root / "solver_history.csv", index=False)
    manifest = {
        "schema": "STB unified GMNS/TAPLite exchange v0.4",
        "network_name": network.name,
        "certificate": result.metadata.get("certificate"),
        "relative_gap": result.relative_gap,
        "demand_residual": result.demand_residual,
        "path_count": len(result.path_table),
        "files": sorted(p.name for p in root.glob("*.csv")),
    }
    (root / "result_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def create_external_dnl_exchange(
    network: StaticNetwork,
    demand: pd.DataFrame,
    path_table: pd.DataFrame,
    departure_profile: np.ndarray,
    folder: str | Path,
    *,
    backend_name: str = "DLSim_or_DTALite",
) -> Path:
    """Create a backend-neutral CSV exchange folder for an external DNL.

    The function does not claim that every DLSim/DTALite build uses identical
    filenames.  It provides a complete, auditable exchange boundary that can be
    adapted by a thin backend-specific script once the local executable and its
    exact schema are supplied.
    """
    root = Path(folder)
    input_dir = root / "input"
    output_dir = root / "expected_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_gmns_case(network, demand, input_dir)
    path_table.to_csv(input_dir / "path.csv", index=False)
    profile = np.asarray(departure_profile, dtype=float)
    pd.DataFrame({"departure_interval": np.arange(profile.size), "share": profile}).to_csv(
        input_dir / "departure_profile.csv", index=False
    )
    pd.DataFrame(columns=["link_id", "time_step", "inflow", "outflow", "queue", "travel_time"]).to_csv(
        output_dir / "link_time_flow.csv", index=False
    )
    pd.DataFrame(columns=["path_id", "departure_interval", "experienced_travel_time"]).to_csv(
        output_dir / "path_departure_time.csv", index=False
    )
    contract = {
        "backend": backend_name,
        "status": "adapter_contract_only_external_executable_not_invoked",
        "inputs": ["node.csv", "link.csv", "demand.csv", "path.csv", "departure_profile.csv"],
        "expected_outputs": ["link_time_flow.csv", "path_departure_time.csv"],
        "required_invariants": [
            "sum inflow minus outflow equals queue change per link and time step",
            "total completed plus queued plus in-transit mass equals loaded demand",
            "path/departure identifiers are preserved",
        ],
    }
    (root / "external_dnl_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root
