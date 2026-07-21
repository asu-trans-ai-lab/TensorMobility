"""Sparse full-space-certified assignment solvers on the unified
network interface (generic port of the Chicago Sketch module).

Every algorithm's relative gap is priced by all-origin Dijkstra over the
REAL network (never pool-restricted). Solvers: solve_fw (CG Frank-Wolfe),
solve_fw_gp (CG + grouped-simplex GP), solve_atom_gp (major/minor
route-family atoms, static or adaptive promotion).

Use network_from_case(load_case('grid'|'sioux_falls'|'chicago_sketch'))
to run the SAME solver on any canonical network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from tensormobility.core.unified_networks import UnifiedCase


@dataclass
class SparseNetwork:
    n_node: int
    from_node: np.ndarray
    to_node: np.ndarray
    t0: np.ndarray                 # free-flow minutes
    cap: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    ref_volume: np.ndarray
    od_o: np.ndarray               # OD group -> origin node index
    od_d: np.ndarray
    demand: np.ndarray

    @property
    def n_link(self) -> int:
        return len(self.t0)

    @property
    def n_od(self) -> int:
        return len(self.demand)

    def graph(self, link_times: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (link_times, (self.from_node, self.to_node)),
            shape=(self.n_node, self.n_node))


# --------------------------------------------------------------------------
# column pool (sparse incidence, O(1) dedup/index)
# --------------------------------------------------------------------------
class ColumnPool:
    def __init__(self, net: SparseNetwork):
        self.net = net
        self.paths: list[tuple[int, ...]] = []
        self.path_od: list[int] = []
        self.index: dict[tuple[int, tuple[int, ...]], int] = {}
        self.od_cols: list[list[int]] = [[] for _ in range(net.n_od)]
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._A: sparse.csr_matrix | None = None

    @property
    def n_col(self) -> int:
        return len(self.paths)

    def add(self, od: int, links: tuple[int, ...]) -> int | None:
        """Add a column; returns its index if new, None if already pooled."""
        key = (od, links)
        if key in self.index:
            return None
        j = len(self.paths)
        self.index[key] = j
        self.paths.append(links)
        self.path_od.append(od)
        self.od_cols[od].append(j)
        self._rows.extend(links)
        self._cols.extend([j] * len(links))
        self._A = None
        return j

    def find(self, od: int, links: tuple[int, ...]) -> int | None:
        return self.index.get((od, links))

    @property
    def A(self) -> sparse.csr_matrix:
        if self._A is None or self._A.shape[1] != self.n_col:
            self._A = sparse.csr_matrix(
                (np.ones(len(self._rows)), (self._rows, self._cols)),
                shape=(self.net.n_link, self.n_col))
        return self._A

    def grouping(self):
        """(order, ptr, seg): columns sorted by OD; group g occupies
        order[ptr[g]:ptr[g+1]]; seg[i] = group of sorted position i."""
        po = np.asarray(self.path_od)
        order = np.argsort(po, kind='stable')
        counts = np.bincount(po, minlength=self.net.n_od)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        seg = np.repeat(np.arange(self.net.n_od), counts)
        return order, ptr, seg


# --------------------------------------------------------------------------
# objective (per-link alpha/beta BPR Beckmann)
# --------------------------------------------------------------------------
def link_time(net: SparseNetwork, v: np.ndarray) -> np.ndarray:
    return net.t0 * (1.0 + net.alpha
                     * (np.maximum(v, 0.0) / net.cap) ** net.beta)


def beckmann(net: SparseNetwork, v: np.ndarray) -> float:
    v = np.maximum(v, 0.0)
    return float(np.sum(net.t0 * (v + net.alpha * v
                                  * (v / net.cap) ** net.beta
                                  / (net.beta + 1.0))))


# --------------------------------------------------------------------------
# pricing: all-origin Dijkstra + path extraction (the full-space oracle)
# --------------------------------------------------------------------------
def price_all_ods(net: SparseNetwork, times: np.ndarray):
    origins = np.unique(net.od_o)
    dist, pred = dijkstra(net.graph(times), indices=origins,
                          return_predecessors=True)
    o_row = {int(o): i for i, o in enumerate(origins)}
    link_of: dict[tuple[int, int], int] = {}
    for l in range(net.n_link):
        key = (int(net.from_node[l]), int(net.to_node[l]))
        cur = link_of.get(key)
        if cur is None or times[l] < times[cur]:
            link_of[key] = l
    sp_cost = np.empty(net.n_od)
    sp_path: list[tuple[int, ...]] = []
    for od in range(net.n_od):
        r = o_row[int(net.od_o[od])]
        o, d = int(net.od_o[od]), int(net.od_d[od])
        sp_cost[od] = dist[r, d]
        links: list[int] = []
        node = d
        ok = True
        while node != o:
            p = int(pred[r, node])
            if p < 0:
                ok = False
                break
            links.append(link_of[(p, node)])
            node = p
        sp_path.append(tuple(reversed(links)) if ok else ())
    return sp_cost, sp_path


def full_space_gap(net, pool, f, v=None):
    """TRUE relative gap: pricing over the network, not the pool."""
    if v is None:
        v = pool.A @ f
    t = link_time(net, v)
    c = pool.A.T @ t
    sp_cost, sp_path = price_all_ods(net, t)
    total = float(f @ c)
    best = float(net.demand @ sp_cost)
    return (total - best) / max(total, 1e-12), sp_cost, sp_path, c, t


# --------------------------------------------------------------------------
# vectorized grouped kernels
# --------------------------------------------------------------------------
def project_simplex_grouped(values, ptr, masses):
    """Project each segment values[ptr[g]:ptr[g+1]] (already segment-
    ordered) onto {x >= 0, sum x = masses[g]}, all segments at once."""
    n = len(values)
    counts = np.diff(ptr)
    seg = np.repeat(np.arange(len(masses)), counts)
    order = np.lexsort((-values, seg))
    u = values[order]
    cs = np.cumsum(u)
    seg_start = ptr[:-1].astype(int)
    shift = np.where(seg_start > 0, cs[np.maximum(seg_start - 1, 0)], 0.0)
    local_cs = cs - np.repeat(shift, counts)
    k = np.arange(n) - np.repeat(seg_start, counts)
    m_rep = np.repeat(masses, counts)
    cond = u - (local_cs - m_rep) / (k + 1.0) > 0
    rho = np.maximum(np.add.reduceat(cond.astype(np.int64), seg_start), 1)
    theta = (local_cs[seg_start + rho - 1] - masses) / rho
    w = np.maximum(u - np.repeat(theta, counts), 0.0)
    out = np.empty(n)
    out[order] = w
    # exact mass repair for float roundoff
    sums = np.add.reduceat(out, seg_start)
    corr = masses - sums
    for g in np.flatnonzero(np.abs(corr) > 1e-10):
        s = slice(ptr[g], ptr[g + 1])
        out[ptr[g] + int(np.argmax(out[s]))] += corr[g]
    return out


def grouped_argmin(c, order, ptr, seg):
    """Index (into the original column numbering) of the min-cost column
    of every group, vectorized."""
    cs = c[order]
    seg_start = ptr[:-1].astype(int)
    mins = np.minimum.reduceat(cs, seg_start)
    mask = cs <= np.repeat(mins, np.diff(ptr))
    pos = np.flatnonzero(mask)
    first = np.unique(seg[pos], return_index=True)[1]
    return order[pos[first]]


# --------------------------------------------------------------------------
# result container (v0.3 AlgorithmResult discipline at sparse scale)
# --------------------------------------------------------------------------
@dataclass
class SparseResult:
    name: str
    objective: float
    relative_gap: float            # ALWAYS full-space priced
    feasibility_residual: float
    n_columns: int
    active_columns: int
    n_atoms: int | None
    pricing_rounds: int
    wall_time_seconds: float
    history: pd.DataFrame
    link_flow: np.ndarray
    metadata: dict = field(default_factory=dict)


def _feas_residual(pool, f):
    sums = np.zeros(pool.net.n_od)
    np.add.at(sums, np.asarray(pool.path_od), f)
    neg = max(0.0, -float(f.min())) if len(f) else 0.0
    return max(float(np.abs(sums - pool.net.demand).max()), neg)


def _line_search_beckmann(net, vA, dA, iters=48):
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    lo, hi = 0.0, 1.0
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    Fa, Fb = beckmann(net, vA + a * dA), beckmann(net, vA + b * dA)
    for _ in range(iters):
        if Fa < Fb:
            hi, b, Fb = b, a, Fa
            a = hi - phi * (hi - lo)
            Fa = beckmann(net, vA + a * dA)
        else:
            lo, a, Fa = a, b, Fb
            b = lo + phi * (hi - lo)
            Fb = beckmann(net, vA + b * dA)
    return 0.5 * (lo + hi)


def _seed_pool(net):
    """Free-flow shortest paths, one column per OD, all-or-nothing flows."""
    pool = ColumnPool(net)
    _, sp_path = price_all_ods(net, net.t0)
    f = []
    for od in range(net.n_od):
        if not sp_path[od]:
            raise ValueError(f'OD {od} disconnected at free flow')
        pool.add(od, sp_path[od])
        f.append(net.demand[od])
    return pool, np.asarray(f)


# --------------------------------------------------------------------------
# algorithm 1: column-generation Frank-Wolfe (full-space pricing per round)
# --------------------------------------------------------------------------
def solve_fw(net, max_rounds=30, inner_iters=20, tolerance=1e-4,
             keep_pool=False):
    start = time.perf_counter()
    pool, f = _seed_pool(net)
    rows = []
    gap = np.inf
    for rnd in range(1, max_rounds + 1):
        v = pool.A @ f
        gap, sp_cost, sp_path, c, t = full_space_gap(net, pool, f, v)
        rows.append(dict(round=rnd, objective=beckmann(net, v),
                         full_relative_gap=gap, n_columns=pool.n_col,
                         seconds=time.perf_counter() - start))
        if gap <= tolerance:
            break
        for od in range(net.n_od):
            if sp_path[od]:
                pool.add(od, sp_path[od])
        f = np.concatenate([f, np.zeros(pool.n_col - len(f))])
        order, ptr, seg = pool.grouping()
        for _ in range(inner_iters):
            v = pool.A @ f
            c = pool.A.T @ link_time(net, v)
            best = grouped_argmin(c, order, ptr, seg)
            y = np.zeros(pool.n_col)
            y[best] = net.demand
            d = y - f
            s = _line_search_beckmann(net, v, pool.A @ d)
            if s <= 1e-14:
                break
            f = f + s * d
    v = pool.A @ f
    meta = {}
    if keep_pool:
        meta['pool'] = pool
        meta['flow'] = f
    return SparseResult(
        name='fw_cg', objective=beckmann(net, v), relative_gap=gap,
        feasibility_residual=_feas_residual(pool, f),
        n_columns=pool.n_col,
        active_columns=int(np.count_nonzero(f > 1e-9)), n_atoms=None,
        pricing_rounds=len(rows),
        wall_time_seconds=time.perf_counter() - start,
        history=pd.DataFrame(rows), link_flow=v, metadata=meta)


# --------------------------------------------------------------------------
# algorithm 2: column generation + grouped-simplex gradient projection
# --------------------------------------------------------------------------
def solve_fw_gp(net, max_rounds=30, gp_sweeps=8, fw_steps=3, tolerance=1e-4):
    start = time.perf_counter()
    pool, f = _seed_pool(net)
    rows = []
    gap = np.inf
    step = 2.0
    for rnd in range(1, max_rounds + 1):
        v = pool.A @ f
        gap, sp_cost, sp_path, c, t = full_space_gap(net, pool, f, v)
        rows.append(dict(round=rnd, objective=beckmann(net, v),
                         full_relative_gap=gap, n_columns=pool.n_col,
                         seconds=time.perf_counter() - start))
        if gap <= tolerance:
            break
        for od in range(net.n_od):
            if sp_path[od]:
                pool.add(od, sp_path[od])
        f = np.concatenate([f, np.zeros(pool.n_col - len(f))])
        order, ptr, seg = pool.grouping()
        # FW discovery steps first (v0.3 division of labor), then GP
        for _ in range(fw_steps):
            v = pool.A @ f
            c = pool.A.T @ link_time(net, v)
            best = grouped_argmin(c, order, ptr, seg)
            y = np.zeros(pool.n_col)
            y[best] = net.demand
            d = y - f
            s = _line_search_beckmann(net, v, pool.A @ d)
            if s <= 1e-14:
                break
            f = f + s * d
        for _ in range(gp_sweeps):
            v = pool.A @ f
            g = pool.A.T @ link_time(net, v)
            cur = beckmann(net, v)
            ok = False
            trial = step
            for _ in range(20):
                w = project_simplex_grouped((f - trial * g)[order], ptr,
                                            net.demand)
                cand = np.empty_like(f)
                cand[order] = w
                d = cand - f
                if beckmann(net, v + pool.A @ d) <= cur + 1e-4 * float(g @ d):
                    f = cand
                    ok = True
                    break
                trial *= 0.5
            step = min(max(trial * (1.6 if ok else 1.0), 1e-8), 64.0)
            if not ok:
                break
    v = pool.A @ f
    return SparseResult(
        name='fw_gp_cg', objective=beckmann(net, v), relative_gap=gap,
        feasibility_residual=_feas_residual(pool, f),
        n_columns=pool.n_col,
        active_columns=int(np.count_nonzero(f > 1e-9)), n_atoms=None,
        pricing_rounds=len(rows),
        wall_time_seconds=time.perf_counter() - start,
        history=pd.DataFrame(rows), link_flow=v,
        metadata=dict(gp_sweeps=gp_sweeps))


# --------------------------------------------------------------------------
# algorithms 3/4: major/minor route-family atoms (static vs adaptive)
# --------------------------------------------------------------------------
def solve_atom_gp(net, adaptive, warm_rounds=6, max_cycles=8, gp_sweeps=8,
                  tolerance=1e-4):
    """Warm a pool with CG-FW rounds, then compress: per OD one MAJOR
    singleton (free-flow-best column) + one MINOR uniform bundle of the
    rest. GP runs in atom space (f = D alpha). Static keeps the decoder
    fixed; adaptive promotes, per cycle, the priced shortest path of every
    OD with positive reduced-cost saving to a singleton atom (v0.3
    latent.solve_latent_fw_gp logic at sparse scale)."""
    start = time.perf_counter()
    warm = solve_fw(net, max_rounds=warm_rounds, tolerance=0.0,
                    keep_pool=True)
    pool: ColumnPool = warm.metadata['pool']

    ff_cost = pool.A.T @ net.t0
    rowsD: list[int] = []
    colsD: list[int] = []
    valsD: list[float] = []
    atom_od: list[int] = []
    singleton_cols: set[int] = set()

    def add_atom(od, cols, weights, singleton_of=None):
        a = len(atom_od)
        rowsD.extend(cols)
        colsD.extend([a] * len(cols))
        valsD.extend(weights)
        atom_od.append(od)
        if singleton_of is not None:
            singleton_cols.add(singleton_of)
        return a

    major_atoms = []
    for od, cols in enumerate(pool.od_cols):
        a = np.asarray(cols)
        best = int(a[np.argmin(ff_cost[a])])
        major_atoms.append(add_atom(od, [best], [1.0], singleton_of=best))
        minor = [c for c in cols if c != best]
        if minor:
            add_atom(od, minor, [1.0 / len(minor)] * len(minor))

    def build_D():
        return sparse.csr_matrix((valsD, (rowsD, colsD)),
                                 shape=(pool.n_col, len(atom_od)))

    def build_grouping():
        ao = np.asarray(atom_od)
        order = np.argsort(ao, kind='stable')
        counts = np.bincount(ao, minlength=net.n_od)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        return order, ptr

    D = build_D()
    n_atoms0 = D.shape[1]
    order, ptr = build_grouping()
    alpha = np.zeros(D.shape[1])
    alpha[major_atoms] = net.demand

    rows = []
    gap = np.inf
    step = 2.0
    promotions = 0
    prev_obj = np.inf
    promoting = adaptive
    for cycle in range(1, max_cycles + 1):
        step = 2.0   # promotion changes the geometry; restart the step
        for _ in range(gp_sweeps):
            f = D @ alpha
            v = pool.A @ f
            g_atom = D.T @ (pool.A.T @ link_time(net, v))
            cur = beckmann(net, v)
            ok = False
            trial = step
            for _ in range(20):
                w = project_simplex_grouped((alpha - trial * g_atom)[order],
                                            ptr, net.demand)
                cand = np.empty_like(alpha)
                cand[order] = w
                d = cand - alpha
                if (beckmann(net, v + pool.A @ (D @ d))
                        <= cur + 1e-4 * float(g_atom @ d)):
                    alpha = cand
                    ok = True
                    break
                trial *= 0.5
            step = min(max(trial * (1.6 if ok else 1.0), 1e-8), 64.0)
            if not ok:
                break
        f = D @ alpha
        v = pool.A @ f
        gap, sp_cost, sp_path, c, t = full_space_gap(net, pool, f, v)
        rows.append(dict(cycle=cycle, objective=beckmann(net, v),
                         full_relative_gap=gap, atoms=D.shape[1],
                         promotions=promotions,
                         seconds=time.perf_counter() - start))
        obj_now = beckmann(net, v)
        if gap <= tolerance:
            break
        if not promoting:
            # polish mode (static decoder, or promotions saturated):
            # keep sweeping until the restricted space is exhausted
            if prev_obj - obj_now <= 1e-9 * max(abs(obj_now), 1.0):
                break
            prev_obj = obj_now
            continue
        prev_obj = obj_now
        # promotion: per-OD saving = current average cost - priced cost
        paid = np.zeros(net.n_od)
        np.add.at(paid, np.asarray(pool.path_od), f * c)
        saving = paid / np.maximum(net.demand, 1e-12) - sp_cost
        added = 0
        for od in np.flatnonzero(saving > 1e-9):
            links = sp_path[od]
            if not links:
                continue
            j = pool.find(od, links)
            if j is None:
                j = pool.add(od, links)
                ff_cost = None  # invalidated; not reused below
            if j in singleton_cols:
                continue
            add_atom(od, [j], [1.0], singleton_of=j)
            added += 1
        if added == 0:
            promoting = False   # saturated: fall through to polish mode
            continue
        promotions += added
        D = build_D()
        old = alpha
        alpha = np.zeros(D.shape[1])
        alpha[:len(old)] = old
        order, ptr = build_grouping()

    f = D @ alpha
    v = pool.A @ f
    return SparseResult(
        name='atom_gp_adaptive' if adaptive else 'atom_gp_static',
        objective=beckmann(net, v), relative_gap=gap,
        feasibility_residual=_feas_residual(pool, f),
        n_columns=pool.n_col,
        active_columns=int(np.count_nonzero(f > 1e-9)),
        n_atoms=D.shape[1],
        pricing_rounds=len(rows),
        wall_time_seconds=time.perf_counter() - start,
        history=pd.DataFrame(rows), link_flow=v,
        metadata=dict(initial_atoms=n_atoms0, promotions=promotions,
                      warm_rounds=warm_rounds,
                      warm_seconds=warm.wall_time_seconds,
                      compression_ratio=pool.n_col / D.shape[1]))


def network_from_case(case: UnifiedCase) -> SparseNetwork:
    """Adapt a UnifiedCase (grid / sioux_falls / chicago_sketch) to the
    sparse solver's network object. Per-link BPR alpha/beta are taken
    from the links frame when present (Chicago), else 0.15/4."""
    net = case.network
    links = net.links
    n_link = net.n_links
    alpha = (links['vdf_alpha'].to_numpy(float)
             if 'vdf_alpha' in links.columns else np.full(n_link, 0.15))
    beta = (links['vdf_beta'].to_numpy(float)
            if 'vdf_beta' in links.columns else np.full(n_link, 4.0))
    ref = (links['ref_volume'].to_numpy(float)
           if 'ref_volume' in links.columns else np.zeros(n_link))
    pos = net.node_id_to_pos
    od_o = np.asarray([pos[int(v)] for v in case.demand.origin_node_id])
    od_d = np.asarray([pos[int(v)] for v in case.demand.destination_node_id])
    return SparseNetwork(
        n_node=net.n_nodes,
        from_node=net.from_pos, to_node=net.to_pos,
        t0=net.free_flow_time.astype(float),
        cap=net.capacity.astype(float),
        alpha=alpha, beta=beta, ref_volume=ref,
        od_o=od_o, od_d=od_d,
        demand=case.demand.volume.to_numpy(float))
