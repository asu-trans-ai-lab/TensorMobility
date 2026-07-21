"""Network-science layer: native incidence/Laplacian certified
against the assignment core; networkx wrappers sane."""
import numpy as np
import pytest

from tensormobility.core.unified_networks import load_case
from tensormobility.dta.sparse_assignment import (network_from_case,
                                                  solve_fw)
from tensormobility.network import (betweenness_centrality,
                                    communities, fiedler_value,
                                    incidence_matrix, laplacian,
                                    node_balance)


def grid_arrays(case):
    """Map GMNS node ids to 0-based indices for the operators."""
    nodes = case.network.nodes.node_id.to_numpy()
    idx = {int(n): i for i, n in enumerate(nodes)}
    links = case.network.links
    f = np.array([idx[int(v)] for v in links.from_node_id])
    t = np.array([idx[int(v)] for v in links.to_node_id])
    return case.network.n_nodes, f, t, idx


@pytest.fixture(scope='module')
def solved_grid():
    case = load_case('grid', rows=6, columns=6, n_od=15,
                     demand_per_od=300.0, base_capacity=900.0)
    net = network_from_case(case)
    res = solve_fw(net, max_rounds=40, tolerance=1e-5)
    return case, res


def test_incidence_expresses_assignment_conservation(solved_grid):
    """A x - b == 0 for certified equilibrium flows: the incidence
    operator reproduces the solver's own conservation audit."""
    case, res = solved_grid
    n, f, t, idx = grid_arrays(case)
    A = incidence_matrix(n, f, t)
    supply = np.zeros(n)
    for row in case.demand.itertuples():
        supply[idx[int(row.origin_node_id)]] += row.volume
        supply[idx[int(row.destination_node_id)]] -= row.volume
    _, cert = node_balance(A, res.link_flow, supply)
    assert cert['passed'], cert


def test_laplacian_rows_zero_and_psd(solved_grid):
    case, _ = solved_grid
    n, f, t, _ = grid_arrays(case)
    L = laplacian(n, f, t)
    assert np.abs(np.asarray(L.sum(axis=1))).max() < 1e-9
    vals = np.linalg.eigvalsh(L.toarray())
    assert vals.min() > -1e-9


def test_fiedler_positive_iff_connected(solved_grid):
    case, _ = solved_grid
    n, f, t, _ = grid_arrays(case)
    assert fiedler_value(laplacian(n, f, t)) > 1e-6
    # disconnect: drop every link touching node 0
    keep = (f != 0) & (t != 0)
    assert fiedler_value(laplacian(n, f[keep], t[keep])) < 1e-8


def test_betweenness_ranks_arterial_cross(solved_grid):
    """The certified engine loads the center cross; betweenness on
    free-flow times must agree that the cross is structurally
    central."""
    case, _ = solved_grid
    n, f, t, _ = grid_arrays(case)
    w = case.network.links.free_flow_time.to_numpy()
    bc = betweenness_centrality(n, f, t, w)
    N = 6
    center = (N // 2) * N + N // 2
    corner = 0
    assert bc[center] > 5 * max(bc[corner], 1e-12)


def test_communities_partition_covers_all_nodes(solved_grid):
    case, _ = solved_grid
    n, f, t, _ = grid_arrays(case)
    parts = communities(n, f, t)
    covered = set().union(*parts)
    assert covered == set(range(n))
    assert 1 < len(parts) < n
