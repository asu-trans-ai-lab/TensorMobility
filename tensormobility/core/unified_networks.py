"""One canonical network interface for grid / Sioux Falls / Chicago
Sketch, all validated through v0.4's StaticNetwork contract.

    case = load_case('grid', rows=10, columns=10, n_od=20)
    case = load_case('sioux_falls')
    case = load_case('chicago_sketch', top_od=500)

Every case returns UnifiedCase(network: StaticNetwork, demand, extras)
where demand has the v0.4 canonical columns
(od_id, origin_node_id, destination_node_id, volume) so every v0.4
solver (column generation, harness, bounded learning) runs unchanged on
all three networks.

Canonicalization rules (stated, not silent):
  - zero free-flow-time connector links (Chicago has 774) are clamped
    to CONNECTOR_FFTT minutes to satisfy the positive-time contract;
  - Sioux Falls free-flow time = 60 * length / free_speed (minutes);
  - the TCGlite 126-path behavioral pool and the Chicago ref_volume
    column ride along in `extras` for certificate reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tensormobility.core.network_core import StaticNetwork, build_static_network
from tensormobility.core.grid_network import build_grid_case

import os

_DATA_ROOT = Path(__file__).resolve().parents[2] / 'data'
SF_DIR = Path(os.environ.get('TENSORMOBILITY_SF_DATA',
                             _DATA_ROOT / 'sioux_falls_tcglite'))
CHICAGO_DIR = Path(os.environ.get('TENSORMOBILITY_CHICAGO_DATA',
                                  _DATA_ROOT / 'chicago_sketch'))
CONNECTOR_FFTT = 1e-3   # minutes; clamp for zero-fftt connectors


@dataclass
class UnifiedCase:
    name: str
    network: StaticNetwork
    demand: pd.DataFrame
    extras: dict = field(default_factory=dict)


def _demand_frame(o, d, volume):
    return pd.DataFrame(dict(
        od_id=np.arange(1, len(volume) + 1),
        origin_node_id=np.asarray(o, dtype=int),
        destination_node_id=np.asarray(d, dtype=int),
        volume=np.asarray(volume, dtype=float)))


def load_grid(rows=10, columns=10, n_od=20, **kw) -> UnifiedCase:
    case = build_grid_case(rows, columns, n_od=n_od, **kw)
    return UnifiedCase('grid', case.network, case.demand,
                       extras=dict(rows=rows, columns=columns))


def load_sioux_falls(demand_scale: float = 1.0) -> UnifiedCase:
    node_df = pd.read_csv(SF_DIR / 'node.csv', encoding='gbk')
    link_df = pd.read_csv(SF_DIR / 'road_link.csv', encoding='gbk')
    agent_df = pd.read_csv(SF_DIR / 'agent.csv', encoding='gbk')

    nodes = pd.DataFrame(dict(node_id=node_df.node_id.astype(int),
                              zone_id=node_df.zone_id,
                              x_coord=node_df.x_coord,
                              y_coord=node_df.y_coord))
    link_df = link_df.dropna(subset=['roadlink_id']).reset_index(drop=True)
    fftt = 60.0 * link_df.length / link_df.free_speed
    links = pd.DataFrame(dict(
        link_id=link_df.roadlink_id.astype(int),
        from_node_id=link_df.from_node_id.astype(int),
        to_node_id=link_df.to_node_id.astype(int),
        free_flow_time=np.maximum(fftt.to_numpy(float), CONNECTOR_FFTT),
        capacity=link_df.capacity.astype(float),
        length=link_df.length.astype(float)))
    network = build_static_network('sioux_falls', nodes, links)

    od = agent_df[agent_df.agent_type == 2].reset_index(drop=True)
    demand = _demand_frame(od.o_zone_id, od.d_zone_id,
                           od.od_flow.to_numpy(float) * demand_scale)

    paths = agent_df[agent_df.agent_type == 3].reset_index(drop=True)
    sensors = agent_df[agent_df.agent_type == 4].reset_index(drop=True)
    return UnifiedCase('sioux_falls', network, demand,
                       extras=dict(tcglite_paths=paths,
                                   sensor_counts=sensors,
                                   demand_scale=demand_scale))


def load_chicago_sketch(top_od: int | None = None) -> UnifiedCase:
    link_df = pd.read_csv(CHICAGO_DIR / 'link.csv')
    node_df = pd.read_csv(CHICAGO_DIR / 'node.csv')
    dem = pd.read_csv(CHICAGO_DIR / 'demand.csv')
    dem = dem[(dem.volume > 0) & (dem.o_zone_id != dem.d_zone_id)]
    if top_od is not None:
        dem = dem.nlargest(top_od, 'volume')
    dem = dem.reset_index(drop=True)

    nodes = pd.DataFrame(dict(node_id=node_df.node_id.astype(int),
                              zone_id=node_df.zone_id,
                              x_coord=node_df.x_coord,
                              y_coord=node_df.y_coord))
    fallback = (60.0 * link_df.vdf_length_mi
                / link_df.vdf_free_speed_mph).fillna(0.0)
    fftt = np.where(link_df.vdf_fftt.to_numpy(float) > 0,
                    link_df.vdf_fftt.to_numpy(float),
                    fallback.to_numpy(float))
    links = pd.DataFrame(dict(
        link_id=link_df.link_id.astype(int),
        from_node_id=link_df.from_node_id.astype(int),
        to_node_id=link_df.to_node_id.astype(int),
        free_flow_time=np.maximum(fftt, CONNECTOR_FFTT),
        capacity=link_df.capacity.astype(float),
        vdf_alpha=link_df.vdf_alpha.astype(float),
        vdf_beta=link_df.vdf_beta.astype(float),
        ref_volume=link_df.ref_volume.astype(float)))
    network = build_static_network('chicago_sketch', nodes, links)
    demand = _demand_frame(dem.o_zone_id, dem.d_zone_id, dem.volume)
    return UnifiedCase('chicago_sketch', network, demand,
                       extras=dict(ref_volume=links.ref_volume.to_numpy(),
                                   top_od=top_od))


_LOADERS = {'grid': load_grid, 'sioux_falls': load_sioux_falls,
            'chicago_sketch': load_chicago_sketch}


def load_case(name: str, **kw) -> UnifiedCase:
    if name not in _LOADERS:
        raise ValueError(f'unknown case {name!r}; '
                         f'options: {sorted(_LOADERS)}')
    return _LOADERS[name](**kw)
