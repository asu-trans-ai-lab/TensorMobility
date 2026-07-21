"""Export the authoritative explainer scenario (design review §6.1,
step 1 of the zero-install sequencing approved 2026-07-22).

One Python-certified scenario JSON becomes the shared source of truth
for all explainer pages: the 5x5 GridCity with certified equilibrium
flows and certificates, the canonical 4x2x2x2 tensor fixture, and the
H017 household. Pages currently embed copies of these data; they will
migrate to fetch this file, eliminating data drift while keeping the
double-click, no-build teaching property.

Run: python cases/export_scenario.py
Output: explainer/scenarios/gridcity_small.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.schema import Certificate, certificates_json
from tensormobility.core.unified_networks import load_case
from tensormobility.dta.sparse_assignment import (network_from_case,
                                                  solve_fw)
from tensormobility.network import incidence_matrix, node_balance

OUT = Path(__file__).resolve().parents[1] / 'explainer' / 'scenarios'
OUT.mkdir(parents=True, exist_ok=True)

FIBERS = {
    'worker|auto|AM': [10, 14, 18, 8],
    'worker|auto|PM': [8, 12, 14, 6],
    'worker|transit|AM': [4, 6, 5, 3],
    'worker|transit|PM': [3, 5, 4, 2],
    'student|auto|AM': [5, 7, 6, 4],
    'student|auto|PM': [4, 6, 5, 3],
    'student|transit|AM': [6, 8, 7, 5],
    'student|transit|PM': [5, 7, 6, 4],
}
B_MATRIX = [[1.0, 0.7, 0.4, 0.0], [0.0, 0.3, 0.6, 1.0]]


def main():
    case = load_case('grid', rows=5, columns=5, n_od=20,
                     demand_per_od=400.0, base_capacity=900.0)
    net = network_from_case(case)
    res = solve_fw(net, max_rounds=60, tolerance=1e-4)

    nodes = case.network.nodes
    idx = {int(n): i for i, n in enumerate(nodes.node_id)}
    links = case.network.links
    f = np.array([idx[int(v)] for v in links.from_node_id])
    t = np.array([idx[int(v)] for v in links.to_node_id])
    A = incidence_matrix(case.network.n_nodes, f, t)
    supply = np.zeros(case.network.n_nodes)
    for row in case.demand.itertuples():
        supply[idx[int(row.origin_node_id)]] += row.volume
        supply[idx[int(row.destination_node_id)]] -= row.volume
    _, cons = node_balance(A, res.link_flow, supply)

    certs = [Certificate('full_space_gap', res.relative_gap < 1e-3,
                         float(res.relative_gap), 1e-3),
             Certificate('node_conservation', cons['passed'],
                         cons['residual'], 1e-6),
             Certificate('feasibility',
                         res.feasibility_residual < 1e-9,
                         float(res.feasibility_residual), 1e-9)]

    scenario = {
        'name': 'gridcity_small',
        'generator': 'cases/export_scenario.py',
        'mode': 'certified_replay_source',
        'network': {
            'n': 5,
            'nodes': [{'id': int(r.node_id),
                       'x': float(r.x_coord),
                       'y': float(r.y_coord)}
                      for r in nodes.itertuples()],
            'links': [{'from': int(fr), 'to': int(to),
                       't0': float(t0), 'cap': float(cap),
                       'flow': float(v)}
                      for fr, to, t0, cap, v in zip(
                          links.from_node_id, links.to_node_id,
                          links.free_flow_time, links.capacity,
                          res.link_flow)],
        },
        'demand': [{'o': int(r.origin_node_id),
                    'd': int(r.destination_node_id),
                    'q': float(r.volume)}
                   for r in case.demand.itertuples()],
        'tensor_fixture': {
            'axes': ['s', 'g', 'm', 'tau'],
            'unit': 'person-trips',
            'fibers': FIBERS,
            'B': B_MATRIX,
        },
        'household_h017': {
            'persons': [
                {'id': 'P1', 'name': 'Alex',
                 'type': 'full_time_worker', 'vot': 28},
                {'id': 'P2', 'name': 'Morgan',
                 'type': 'part_time_worker', 'vot': 18},
                {'id': 'P3', 'name': 'Sam', 'type': 'k12_student',
                 'vot': 8}],
            'vehicle': 'V1',
        },
        'certificates': json.loads(certificates_json(
            certs, case='gridcity_small'))
    }
    out = OUT / 'gridcity_small.json'
    out.write_text(json.dumps(scenario, indent=1), encoding='utf-8')
    print(f'wrote {out}')
    print(f"gap {res.relative_gap:.2e} · conservation "
          f"{cons['residual']:.1e} · all_passed "
          f"{scenario['certificates']['all_passed']}")


if __name__ == '__main__':
    main()
