"""Adapter: TrafficFlowBench (IEEE five-corridor benchmark, PeMS
D7/D12) -> TensorMobility unified case.

The benchmark ships, per directional panel (e.g. I405N): a GMNS
corridor network, a PRE-BUILT path set with link sequences, a base OD
table, 5-minute detector states (speed/flow/occupancy/density),
Newell-style queue episode objects (T0/T2/T3, extent, discharge),
per-detector fundamental-diagram parameters, and historical
time-of-day profiles.

Mapping into the axis calculus:
  links/paths      -> space axes (the DTA core runs unchanged)
  5-min timestamps -> simulation-time axis (queue core)
  queue episodes   -> regime axis material (v0.7 state promotion)
  historical profiles -> the SIMPLE DEPARTURE-TIME PROFILE (first
                      behavioral step, per the author's sequencing)
  FD parameters    -> the physics library for PINN-style residual
                      certificates (docs/PINN_INTEGRATION.md)

Data stays local (competition data): set TENSORMOBILITY_TFB_DATA to the
benchmark root; tests skip gracefully when absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tensormobility.core.network_core import build_static_network
from tensormobility.core import unified_networks as _un
from tensormobility.core.unified_networks import UnifiedCase

TFB_ROOT = Path(os.environ.get(
    'TENSORMOBILITY_TFB_DATA',
    r'C:\source_codes\0_source_code_new\IEEE_Simulate_Players'
    r'\trafficflowbench-d12-0711_2026'))


def _panel_dir(panel: str) -> Path:
    d = TFB_ROOT / 'data_sample' / panel / 'train'
    if not d.exists():
        raise FileNotFoundError(
            f'TrafficFlowBench panel {panel!r} not found under '
            f'{TFB_ROOT} (set TENSORMOBILITY_TFB_DATA)')
    return d


def data_available(panel: str = 'I405N') -> bool:
    try:
        _panel_dir(panel)
        return True
    except FileNotFoundError:
        return False


def load_tfb_corridor(panel: str = 'I405N') -> UnifiedCase:
    d = _panel_dir(panel)
    link_df = pd.read_csv(d / 'train_gmns_link.csv')
    node_df = pd.read_csv(d / 'train_gmns_node.csv')
    paths = pd.read_csv(d / 'path_set.csv')
    base_od = pd.read_csv(d / 'base_od.csv')

    nodes = pd.DataFrame(dict(node_id=node_df.node_id.astype(int),
                              zone_id=node_df.zone_id,
                              x_coord=node_df.x_coord,
                              y_coord=node_df.y_coord))
    fftt = 60.0 * link_df.length_km / link_df.free_speed_kmh
    links = pd.DataFrame(dict(
        link_id=np.arange(1, len(link_df) + 1),
        tfb_link_id=link_df.link_id,
        from_node_id=link_df.from_node_id.astype(int),
        to_node_id=link_df.to_node_id.astype(int),
        free_flow_time=np.maximum(fftt.to_numpy(float), 1e-3),
        capacity=link_df.capacity_vph.astype(float),
        length=link_df.length_km.astype(float),
        lanes=link_df.lanes,
        is_ramp=link_df.is_ramp))
    network = build_static_network(f'tfb_{panel}', nodes, links)
    lid = {t: i + 1 for i, t in enumerate(link_df.link_id)}
    l_from = dict(zip(link_df.link_id, link_df.from_node_id))
    l_to = dict(zip(link_df.link_id, link_df.to_node_id))

    # paths -> link-index sequences; OD via path endpoints (robust to
    # the Zk zone-label indirection)
    seqs = {r.path_id: [s for s in str(r.link_seq).split(';') if s]
            for r in paths.itertuples(index=False)}
    flow = base_od.groupby('path_id').base_flow.sum()
    rows = []
    for pid, seq in seqs.items():
        rows.append(dict(path_id=pid,
                         origin_node_id=int(l_from[seq[0]]),
                         destination_node_id=int(l_to[seq[-1]]),
                         links=tuple(lid[s] for s in seq),
                         volume=float(flow.get(pid, 0.0))))
    path_table = pd.DataFrame(rows)
    demand = (path_table.groupby(['origin_node_id',
                                  'destination_node_id'],
                                 as_index=False).volume.sum())
    # circular corridor paths (ramp zones) can give o == d; the static
    # OD core cannot represent self-loops -- drop and record
    self_od = demand.origin_node_id == demand.destination_node_id
    n_self = int(self_od.sum())
    demand = demand[~self_od].reset_index(drop=True)

    # DATA CHARACTERISTIC (measured): most released link_seq chains are
    # detector-incidence sequences, NOT contiguous walks (2068/2145 on
    # I405N). The incidence structure is kept exactly (ODME face); the
    # DTA core additionally requires graph-routable ODs, so unroutable
    # endpoint pairs are dropped WITH COUNT, never silently.
    from scipy import sparse as _sp
    from scipy.sparse.csgraph import dijkstra as _dij
    pos = network.node_id_to_pos
    G = _sp.csr_matrix(
        (network.free_flow_time,
         (network.from_pos, network.to_pos)),
        shape=(network.n_nodes, network.n_nodes))
    origins = sorted({pos[int(o)] for o in demand.origin_node_id})
    dist = _dij(G, indices=origins)
    orow = {o: i for i, o in enumerate(origins)}
    reachable = np.array([
        np.isfinite(dist[orow[pos[int(r.origin_node_id)]],
                         pos[int(r.destination_node_id)]])
        for r in demand.itertuples(index=False)])
    n_unroutable = int((~reachable).sum())
    demand = demand[reachable].reset_index(drop=True)
    demand.insert(0, 'od_id', np.arange(1, len(demand) + 1))

    extras = dict(
        n_self_od_dropped=n_self,
        n_unroutable_od_dropped=n_unroutable,
        panel=panel,
        path_table=path_table,
        base_od=base_od,
        tfb_link_ids=link_df.link_id.tolist(),
        detector_states_file=d / 'train_detector_states.csv',
        queue_objects=pd.read_csv(d / 'queue_objects_train.csv'),
        detector_fd=pd.read_csv(d / 'detector_chain_fd.csv'),
        historical_profiles=pd.read_csv(
            d / 'train_historical_profiles.csv'),
        path_link_incidence_file=d / 'path_link_incidence.csv')
    return UnifiedCase(f'tfb_{panel}', network, demand, extras)


def validate_path_incidence(case: UnifiedCase) -> int:
    """Certificate: the link sequences reconstruct the released
    path_link_incidence table exactly. Returns number of (path, link)
    pairs checked."""
    inc = pd.read_csv(case.extras['path_link_incidence_file'])
    released = set(zip(inc.path_id, inc.link_id))
    tfb_ids = case.extras['tfb_link_ids']
    rebuilt = set()
    for r in case.extras['path_table'].itertuples(index=False):
        for li in r.links:
            rebuilt.add((r.path_id, tfb_ids[li - 1]))
    if rebuilt != released:
        missing = len(released - rebuilt)
        extra = len(rebuilt - released)
        raise AssertionError(f'incidence mismatch: missing={missing} '
                             f'extra={extra}')
    return len(released)


def rebuild_contiguous_paths(case: UnifiedCase) -> dict:
    """REBUILD the path-link incidence as contiguous walks.

    Measured on I405N: 2068/2145 released link_seq chains are
    non-contiguous and 1547 are not even ordered along the corridor --
    they are detector-incidence records for ODME scoring, and some
    anchor sets (mainline + branch links) cannot lie on one directed
    walk at all. Rebuild policy, reported not hidden:

      1. sort each path's anchor links by free-flow network distance
         from the path origin (graph order, not milepost order);
      2. greedily chain anchors, inserting the shortest free-flow
         connector sequence between consecutive reachable anchors;
      3. anchors unreachable from the running walk head are DROPPED
         and counted (anchors_dropped);
      4. every rebuilt walk is verified contiguous link-by-link.

    Returns dict(path_table, incidence, stats) where stats reports
    n_paths, n_contiguous, anchors_total, anchors_kept,
    links_inserted, coverage = anchors_kept / anchors_total.
    """
    net = case.network
    n = net.n_nodes
    from scipy import sparse as _sp
    from scipy.sparse.csgraph import dijkstra as _dij
    fp, tp = net.from_pos, net.to_pos
    w = net.free_flow_time
    G = _sp.csr_matrix((w, (fp, tp)), shape=(n, n))
    link_of: dict = {}
    for i in range(net.n_links):
        key = (int(fp[i]), int(tp[i]))
        if key not in link_of or w[i] < w[link_of[key]]:
            link_of[key] = i

    _cache: dict = {}

    def sp_from(s):
        if s not in _cache:
            d, pr = _dij(G, indices=s, return_predecessors=True)
            _cache[s] = (d, pr)
        return _cache[s]

    def connector(s, t):
        d, pr = sp_from(s)
        if not np.isfinite(d[t]):
            return None
        links = []
        node = t
        while node != s:
            pnode = int(pr[node])
            if pnode < 0:
                return None
            links.append(link_of[(pnode, node)] + 1)
            node = pnode
        return list(reversed(links))

    pos = net.node_id_to_pos
    pt = case.extras['path_table']
    tfb_ids = case.extras['tfb_link_ids']
    rows, inc_rows = [], []
    anchors_total = anchors_kept = links_inserted = n_contig = 0
    for r in pt.itertuples(index=False):
        seq = list(dict.fromkeys(r.links))       # dedupe, keep order
        anchors_total += len(seq)
        o_pos = pos[int(r.origin_node_id)]
        d0, _ = sp_from(o_pos)
        # graph order from the origin; unreachable-from-origin anchors
        # sort to the end and get dropped by the chaining step
        seq.sort(key=lambda li: (not np.isfinite(d0[int(fp[li - 1])]),
                                 d0[int(fp[li - 1])]))
        full = None
        kept = []
        for li in seq:
            if full is None:
                if np.isfinite(d0[int(fp[li - 1])]):
                    lead = connector(o_pos, int(fp[li - 1]))
                    if lead is None:
                        continue
                    full = lead + [li]
                    kept.append(li)
                continue
            head = int(tp[full[-1] - 1])
            if head == int(fp[li - 1]):
                full.append(li)
                kept.append(li)
                continue
            c = connector(head, int(fp[li - 1]))
            if c is None:
                continue                          # branch anchor: drop
            links_inserted += len(c)
            full.extend(c)
            full.append(li)
            kept.append(li)
        if full is None or not kept:
            continue
        anchors_kept += len(kept)
        contiguous = all(int(tp[full[i] - 1]) == int(fp[full[i + 1] - 1])
                         for i in range(len(full) - 1))
        if contiguous:
            n_contig += 1
        rows.append(dict(
            path_id=r.path_id,
            origin_node_id=r.origin_node_id,
            destination_node_id=int(
                net.links.iloc[full[-1] - 1].to_node_id)
            if hasattr(net.links.iloc[full[-1] - 1], 'to_node_id')
            else r.destination_node_id,
            anchors=tuple(kept), links=tuple(full), volume=r.volume))
        for li in full:
            inc_rows.append((r.path_id, tfb_ids[li - 1]))
    rebuilt = pd.DataFrame(rows)
    incidence = pd.DataFrame(inc_rows, columns=['path_id', 'link_id'])
    return dict(path_table=rebuilt, incidence=incidence,
                stats=dict(n_paths=len(rebuilt), n_contiguous=n_contig,
                           anchors_total=anchors_total,
                           anchors_kept=anchors_kept,
                           coverage=anchors_kept / max(anchors_total, 1),
                           links_inserted=links_inserted))


def departure_profile(case: UnifiedCase, day_type: str | None = None
                      ) -> pd.DataFrame:
    """The simple departure-time profile: normalized mean mainline flow
    by time of day from the released historical profiles (the first
    behavioral axis step; richer activity axes come later)."""
    h = case.extras['historical_profiles']
    if day_type is not None and 'day_type' in h.columns:
        h = h[h.day_type == day_type]
    prof = (h.groupby('time_of_day', as_index=False).mean_flow.mean()
            .sort_values('time_of_day').reset_index(drop=True))
    prof['share'] = prof.mean_flow / prof.mean_flow.sum()
    return prof[['time_of_day', 'mean_flow', 'share']]


def bottleneck_inflow(case: UnifiedCase, episode_rank: int = 0):
    """Real 5-minute inflow series at the bottleneck link of the
    episode with the largest queue extent; returns (inflow veh/5min,
    capacity veh/5min, episode row)."""
    q = case.extras['queue_objects']
    det = pd.read_csv(case.extras['detector_states_file'])
    ranked = q.sort_values('max_queue_extent_km', ascending=False)
    taken = 0
    for _, ep in ranked.iterrows():
        for lk in (ep.link_id, ep.get('bottleneck_link')):
            if not isinstance(lk, str):
                continue
            day = det[(det.date == ep.date) & (det.link_id == lk)]
            if len(day) >= 100:
                if taken < episode_rank:
                    taken += 1
                    break
                day = day.sort_values('timestamp')
                inflow = np.nan_to_num(
                    day.flow.to_numpy(float)) / 12.0   # vph->veh/5min
                fd = case.extras['detector_fd']
                caps = fd[fd.link_id == lk].capacity_vph
                cap = (float(caps.iloc[0]) / 12.0 if len(caps)
                       else float(np.nanmax(day.flow)) / 12.0)
                return inflow, cap, ep
    raise LookupError('no queue episode with detector coverage found')


# register with the unified front door
_un._LOADERS['tfb_corridor'] = load_tfb_corridor
