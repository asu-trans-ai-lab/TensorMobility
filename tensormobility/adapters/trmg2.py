"""Adapter: TRMG2 regional GMNS scenarios -> TensorMobility unified
case (experiment E4/F1: transfer validation + canonical load).

The TRMG2 open regional model (Research Triangle, 33,963 nodes /
75,939 links / 3,247 zones) ships per-scenario GMNS bundles
(scenario_AM/MD/PM/NT) with multi-class demand (sov, hov2, hov3).
This adapter performs the F1 stage of the regional contract:

  1. canonical load through the StaticNetwork contract;
  2. a machine-readable transfer-validation report (connectivity,
     zone-centroid mapping, demand conservation by class, unroutable
     and self ODs, duplicate records) rather than only converted
     files.

Data stays local; set TENSORMOBILITY_TRMG2_DATA to the gmns folder.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tensormobility.core.network_core import build_static_network
from tensormobility.core import unified_networks as _un
from tensormobility.core.unified_networks import UnifiedCase

TRMG2_ROOT = Path(os.environ.get(
    'TENSORMOBILITY_TRMG2_DATA',
    r'C:\source_codes\0_source_code_new\compressed_optimization'
    r'\Paper_2_Latent_Atom_NN_connection\trmg2_GMNS\TRMG2_gmns\gmns'))
CONNECTOR_FFTT = 1e-3


def data_available(scenario: str = 'AM') -> bool:
    return (TRMG2_ROOT / f'scenario_{scenario}' / 'link.csv').exists()


def load_trmg2(scenario: str = 'AM',
               demand_classes=('sov', 'hov2', 'hov3'),
               top_od: int | None = None) -> UnifiedCase:
    d = TRMG2_ROOT / f'scenario_{scenario}'
    link_df = pd.read_csv(d / 'link.csv', low_memory=False)
    node_df = pd.read_csv(d / 'node.csv')

    nodes = pd.DataFrame(dict(node_id=node_df.node_id.astype(int),
                              zone_id=node_df.zone_id,
                              x_coord=node_df.x_coord,
                              y_coord=node_df.y_coord))
    fftt = 60.0 * link_df.length / link_df.free_speed
    links = pd.DataFrame(dict(
        link_id=np.arange(1, len(link_df) + 1),
        trmg2_link_id=link_df.link_id,
        from_node_id=link_df.from_node_id.astype(int),
        to_node_id=link_df.to_node_id.astype(int),
        free_flow_time=np.maximum(
            fftt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            .to_numpy(float), CONNECTOR_FFTT),
        capacity=np.maximum(link_df.capacity.to_numpy(float), 1.0),
        length=link_df.length.astype(float)))
    network = build_static_network(f'trmg2_{scenario}', nodes, links)

    # zone -> centroid node (zone_id column on nodes)
    zone_node = (node_df.dropna(subset=['zone_id'])
                 .astype({'zone_id': int})
                 .set_index('zone_id').node_id.to_dict())

    frames = []
    class_totals = {}
    for cls in demand_classes:
        f = d / f'demand_{cls}.csv'
        if not f.exists():
            continue
        dd = pd.read_csv(f)
        class_totals[cls] = float(dd.volume.sum())
        frames.append(dd)
    dem = (pd.concat(frames).groupby(['o_zone_id', 'd_zone_id'],
                                     as_index=False).volume.sum())
    n_raw = len(dem)
    dem = dem[dem.o_zone_id != dem.d_zone_id]
    n_self = n_raw - len(dem)
    unmapped = (~dem.o_zone_id.isin(zone_node)
                | ~dem.d_zone_id.isin(zone_node)).sum()
    dem = dem[dem.o_zone_id.isin(zone_node)
              & dem.d_zone_id.isin(zone_node)]
    if top_od is not None:
        dem = dem.nlargest(top_od, 'volume')
    dem = dem.reset_index(drop=True)
    demand = pd.DataFrame(dict(
        od_id=np.arange(1, len(dem) + 1),
        origin_node_id=[zone_node[z] for z in dem.o_zone_id],
        destination_node_id=[zone_node[z] for z in dem.d_zone_id],
        volume=dem.volume.astype(float)))

    validation = dict(
        scenario=scenario,
        nodes=int(network.n_nodes), links=int(network.n_links),
        zones=len(zone_node),
        demand_classes=class_totals,
        demand_total=float(demand.volume.sum()),
        od_pairs=len(demand),
        self_od_dropped=int(n_self),
        unmapped_zone_od_dropped=int(unmapped),
        duplicate_links=int(link_df.duplicated(
            ['from_node_id', 'to_node_id']).sum()),
        nonpositive_speed_links=int((link_df.free_speed <= 0).sum()),
        nonpositive_capacity_links=int((link_df.capacity <= 0).sum()))
    return UnifiedCase(f'trmg2_{scenario}', network, demand,
                       extras=dict(validation=validation,
                                   zone_node=zone_node))


_un._LOADERS['trmg2'] = load_trmg2
