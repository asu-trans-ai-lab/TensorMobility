"""MAGE operator profile: steady-state mixed-autonomy ride-hailing
equilibrium on any unified network (grid first — TRB-D-26-00426, Hou,
Wang, Li & Pang).

SIMPLER-FIRST SCOPE (stated, per the author's standing directive): the
full MAGE NCP/VI is *approached* by a damped best-response fixed point
(diagonalization) whose solution is then AUDITED against the
equilibrium conditions — logit-choice consistency, patience cap,
rationing complementarity (unserved > 0 => waiting at patience),
fleet-cap feasibility, and a full-space-certified road assignment every
outer iteration (reusing sparse_assignment.solve_fw). A semismooth
NCP solver is the declared v0.6 extension; multi-start spread is
reported because GNE uniqueness is NOT guaranteed.

The two queues are kept distinct throughout:
  matching queue  w_match (customers waiting for vehicles, patience-capped)
  road congestion t(v)    (vehicles waiting for capacity, BPR)

Structure (each element replaceable — the extensibility contract):
  choice_operator     logit over {solo} U {(company, AV/HV)}
  matching_operator   w = min(patience, w0/(z - D)) + pickup access
  fleet_operator      company best response: AV-first allocation under
                      total-fleet and AV-cap constraints
  assignment_operator sparse_assignment.solve_fw (full-space certified)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tensormobility.engines.equilibrium_engines import solve_fixed_point
from tensormobility.dta.sparse_assignment import (SparseNetwork, network_from_case,
                                solve_fw, link_time, price_all_ods)
from tensormobility.core.unified_networks import UnifiedCase


@dataclass
class Company:
    name: str
    fleet: float                 # total vehicles N_k
    av_cap: float                # mu: max AV share of fleet
    fare_equiv_min: dict         # {'AV': .., 'HV': ..} fare as
                                 # value-of-time-equivalent minutes
    op_cost_equiv_min: dict      # operating cost per served trip (min
                                 # equivalent; HV includes driver wage)


@dataclass
class MAGEConfig:
    companies: list[Company]
    patience_min: float = 12.0       # customer patience cap (minutes)
    w0: float = 60.0                 # matching constant: w = w0/(z - D)
    pickup_factor: float = 0.30      # pickup deadhead VMT as a fraction
                                     # of service VMT (stage accounting)
    pickup_access_min: float = 3.0   # customer-side pickup access time
    theta: float = 0.15              # logit scale (1/min)
    headroom: float = 1.2            # fleet target capacity / demand
    solo_cost_equiv_min: float = 8.0 # ownership/parking equivalent
    damp: float = 0.15
    fleet_damp: float = 0.25         # fleet reallocates on a SLOWER
                                     # timescale than traveler choice
                                     # (two-timescale diagonalization)
    max_outer: int = 40
    inner_engine: str = 'newton'     # 'picard'|'msa'|'anderson'|
                                     # 'newton'|'auto' (escalation ladder)
    tol: float = 1e-7
    fw_rounds: int = 12
    fw_tol: float = 1e-4


def _options(cfg: MAGEConfig):
    """Option list: solo + (company, vtype) pairs."""
    opts = [('solo', None, None)]
    for k, comp in enumerate(cfg.companies):
        opts += [('rh', k, 'AV'), ('rh', k, 'HV')]
    return opts


def _opt_col(opts, k, x):
    for j, (kind, kk, xx) in enumerate(opts):
        if kind == 'rh' and kk == k and xx == x:
            return j
    raise KeyError((k, x))


def solve_mage(case: UnifiedCase, cfg: MAGEConfig,
               shares0: np.ndarray | None = None) -> dict:
    net = network_from_case(case)
    base_demand = net.demand.copy()          # persons per period per OD
    n_od = net.n_od
    opts = _options(cfg)
    n_opt = len(opts)

    shares = (np.full((n_od, n_opt), 1.0 / n_opt)
              if shares0 is None else shares0.copy())
    t_od = None
    z_prev: dict | None = None
    hist = []
    res = np.inf
    for it in range(1, cfg.max_outer + 1):
        # --- demand per option -------------------------------------
        D_opt = shares * base_demand[:, None]           # persons
        D_kx = {}
        for j, (kind, k, x) in enumerate(opts):
            if kind == 'rh':
                D_kx[(k, x)] = D_opt[:, j]

        # --- fleet operator (company best response) ----------------
        # cycle time uses last iteration's OD times (free flow first)
        if t_od is None:
            sp_cost, _ = price_all_ods(net, net.t0)
            t_od = sp_cost
        wbar_t = float(np.average(t_od, weights=np.maximum(
            base_demand, 1e-9)))
        cycle = (1.0 + cfg.pickup_factor) * wbar_t      # minutes/trip
        z = {}
        fleet_alloc = {}
        for k, comp in enumerate(cfg.companies):
            need = {}
            for x in ('AV', 'HV'):
                dem = D_kx[(k, x)].sum()                # persons/period
                # fleet needed for target rate (period = 60 min)
                need[x] = cfg.headroom * dem * cycle / 60.0
            # allocate the CHEAPER-to-operate type first (this is where
            # op_cost_equiv_min enters: cost-ordered priority, with the
            # AV-share cap binding only the AV pool)
            order_x = sorted(('AV', 'HV'),
                             key=lambda x: comp.op_cost_equiv_min[x])
            remaining = comp.fleet
            alloc = {}
            for xk in order_x:
                cap_x = (comp.av_cap * comp.fleet if xk == 'AV'
                         else remaining)
                alloc[xk] = min(need[xk], cap_x, remaining)
                remaining -= alloc[xk]
            fleet_alloc[k] = alloc
            for x in ('AV', 'HV'):
                z_new = 60.0 * fleet_alloc[k][x] / cycle      # pers/period
                if z_prev is not None:
                    z_new = ((1.0 - cfg.fleet_damp) * z_prev[(k, x)]
                             + cfg.fleet_damp * z_new)
                z[(k, x)] = z_new
        z_prev = dict(z)

        # --- rationing + matching operator -------------------------
        served, unserved_total = {}, np.zeros(n_od)
        w_match = {}
        for k, comp in enumerate(cfg.companies):
            for x in ('AV', 'HV'):
                dem = D_kx[(k, x)]
                tot = dem.sum()
                frac = 1.0 if tot <= z[(k, x)] else z[(k, x)] / max(tot, 1e-12)
                served[(k, x)] = dem * frac
                unserved_total += dem * (1.0 - frac)
                slack = max(z[(k, x)] - tot, 0.0)
                # smooth patience-capped wait: slack=0 -> patience,
                # slack->inf -> w0/slack (monotone, kink-free -- the
                # hard min() creates best-response limit cycles)
                w_match[(k, x)] = cfg.w0 / (slack
                                            + cfg.w0 / cfg.patience_min)

        # --- road assignment (certified) ---------------------------
        # vehicle OD demand: solo (+ rationed spill) + service + pickup
        solo = D_opt[:, 0] + unserved_total
        service = sum(served.values())
        veh_demand = solo + (1.0 + cfg.pickup_factor) * service
        net.demand = veh_demand
        r = solve_fw(net, max_rounds=cfg.fw_rounds,
                     tolerance=cfg.fw_tol)
        times = link_time(net, r.link_flow)
        sp_cost, _ = price_all_ods(net, times)
        t_od = sp_cost

        # --- FAST subsystem: choice <-> matching wait at frozen
        # (t_od, z). The M/M/1-style wait is far too stiff for damped
        # iteration (dw/dshare ~ 1e4 near saturation), but the
        # subsystem is only |rh options|-dimensional in the wait
        # vector, so it is solved exactly by a Newton root-finder.
        rh_keys = [(k, x) for k in range(len(cfg.companies))
                   for x in ('AV', 'HV')]

        def shares_given_w(w_vec):
            U = np.empty((n_od, n_opt))
            for j, (kind, k, x) in enumerate(opts):
                if kind == 'solo':
                    U[:, j] = -(t_od + cfg.solo_cost_equiv_min)
                else:
                    U[:, j] = -(t_od + w_vec[rh_keys.index((k, x))]
                                + cfg.pickup_access_min
                                + cfg.companies[k].fare_equiv_min[x])
            Zx = np.exp(cfg.theta * (U - U.max(axis=1, keepdims=True)))
            return Zx / Zx.sum(axis=1, keepdims=True)

        def wait_residual(w_vec):
            s = shares_given_w(w_vec)
            out = np.empty(len(rh_keys))
            for i, (k, x) in enumerate(rh_keys):
                tot = float((s[:, _opt_col(opts, k, x)]
                             * base_demand).sum())
                slack = max(z[(k, x)] - tot, 0.0)
                out[i] = w_vec[i] - cfg.w0 / (slack
                                              + cfg.w0 / cfg.patience_min)
            return out

        def wait_map(w_vec):
            return w_vec - wait_residual(w_vec)     # g(w): predicted wait

        w_guess = np.array([w_match[key] for key in rh_keys]) \
            if it > 1 else np.full(len(rh_keys), 0.5 * cfg.patience_min)
        eng = solve_fixed_point(wait_map, w_guess,
                                engine=cfg.inner_engine, tol=1e-11,
                                bounds=(0.0, cfg.patience_min))
        w_vec = np.clip(eng.x, 0.0, cfg.patience_min)
        inner_res = float(np.abs(wait_residual(w_vec)).max())
        inner_engine_used = eng.engine
        inner_escalations = eng.escalations
        inner_cycle = eng.cycle
        shares = shares_given_w(w_vec)
        w_match = {key: float(w_vec[i]) for i, key in enumerate(rh_keys)}

        # outer residual: slow variables (congestion + fleet capacity)
        res = inner_res
        if it > 1:
            res = max(inner_res,
                      float(np.abs(t_od - t_od_prev).max())
                      / max(float(t_od.mean()), 1e-9))
        t_od_prev = t_od.copy()
        hist.append(dict(iteration=it, residual=res,
                         inner_residual=inner_res,
                         fw_gap=r.relative_gap,
                         mean_t_od=float(t_od.mean())))
        if res < 1e-5 and it > 3:
            break

    # ---------------- final-state consistency pass ------------------
    # One COMPLETE evaluation x* -> D* -> z* -> w* -> T* -> x*' after
    # termination, so every reported certificate refers to the same
    # (final) state and the closure residuals are explicit:
    #   r_x (shares), r_z (service capacity), r_T (experienced times).
    D_opt = shares * base_demand[:, None]
    D_kx = {(k, x): D_opt[:, _opt_col(opts, k, x)]
            for k in range(len(cfg.companies)) for x in ('AV', 'HV')}
    wbar_t = float(np.average(t_od, weights=np.maximum(base_demand,
                                                       1e-9)))
    cyc_f = (1.0 + cfg.pickup_factor) * wbar_t
    z_f, fleet_alloc = {}, {}
    for k, comp in enumerate(cfg.companies):
        need = {x: cfg.headroom * float(D_kx[(k, x)].sum()) * cyc_f / 60.0
                for x in ('AV', 'HV')}
        order_x = sorted(('AV', 'HV'),
                         key=lambda x: comp.op_cost_equiv_min[x])
        remaining = comp.fleet
        alloc = {}
        for xk in order_x:
            cap_x = (comp.av_cap * comp.fleet if xk == 'AV'
                     else remaining)
            alloc[xk] = min(need[xk], cap_x, remaining)
            remaining -= alloc[xk]
        fleet_alloc[k] = alloc
        for x in ('AV', 'HV'):
            z_f[(k, x)] = 60.0 * alloc[x] / cyc_f
    r_z = max(abs(z_f[key] - z[key]) / max(z[key], 1e-9) for key in z_f)
    served, unserved_total = {}, np.zeros(n_od)
    for k, comp in enumerate(cfg.companies):
        for x in ('AV', 'HV'):
            dem = D_kx[(k, x)]
            tot = float(dem.sum())
            frac = 1.0 if tot <= z_f[(k, x)] \
                else z_f[(k, x)] / max(tot, 1e-12)
            served[(k, x)] = dem * frac
            unserved_total += dem * (1.0 - frac)
    solo = D_opt[:, 0] + unserved_total
    service = sum(served.values())
    net.demand = solo + (1.0 + cfg.pickup_factor) * service
    r = solve_fw(net, max_rounds=cfg.fw_rounds, tolerance=cfg.fw_tol)
    t_f, _ = price_all_ods(net, link_time(net, r.link_flow))
    r_T = float(np.abs(t_f - t_od).max()) / max(float(t_od.mean()),
                                                1e-9)
    t_od = t_f          # closures (shares_given_w / wait_residual)
    z = z_f             # now see the FINAL state
    eng_f = solve_fixed_point(wait_map, w_vec, engine=cfg.inner_engine,
                              tol=1e-11,
                              bounds=(0.0, cfg.patience_min))
    w_vec = np.clip(eng_f.x, 0.0, cfg.patience_min)
    shares_f = shares_given_w(w_vec)
    r_x = float(np.abs(shares_f - shares).max())
    shares = shares_f
    w_match = {key: float(w_vec[i]) for i, key in enumerate(rh_keys)}
    D_opt = shares * base_demand[:, None]
    final_consistency = dict(r_x=r_x, r_z=float(r_z), r_T=r_T)

    # ---------------- certificates at the fixed point --------------
    conservation = float(np.abs(D_opt.sum(axis=1) - base_demand).max())
    patience_viol = max((w - cfg.patience_min
                         for w in w_match.values()), default=0.0)
    # rationing complementarity: unserved mass > 0 => waiting == patience
    comp_viol = 0.0
    for k, comp in enumerate(cfg.companies):
        for x in ('AV', 'HV'):
            un = float((D_kx[(k, x)] - served[(k, x)]).sum())
            if un > 1e-6 * max(base_demand.sum(), 1.0):
                comp_viol = max(comp_viol,
                                cfg.patience_min - w_match[(k, x)])
    fleet_feas = max(
        max(0.0, sum(fleet_alloc[k].values()) - c.fleet) +
        max(0.0, fleet_alloc[k]['AV'] - c.av_cap * c.fleet)
        for k, c in enumerate(cfg.companies))

    mode_share = {}
    for j, (kind, k, x) in enumerate(opts):
        key = 'solo' if kind == 'solo' else \
            f'{cfg.companies[k].name}:{x}'
        mode_share[key] = float((shares[:, j] * base_demand).sum()
                                / base_demand.sum())

    return dict(
        shares=shares, mode_share=mode_share,
        w_match={f'{cfg.companies[k].name}:{x}': w
                 for (k, x), w in w_match.items()},
        fleet_alloc={cfg.companies[k].name: v
                     for k, v in fleet_alloc.items()},
        served_fraction=float(sum(s.sum() for s in served.values())
                              / max(sum(d.sum() for d in D_kx.values()),
                                    1e-12)),
        t_od=t_od, outer_iterations=it,
        final_consistency=final_consistency,
        inner_engine=inner_engine_used,
        inner_escalations=inner_escalations,
        inner_cycle=inner_cycle,
        fixed_point_residual=res,
        assignment_gap=float(r.relative_gap),
        conservation_residual=conservation,
        patience_violation=float(max(patience_viol, 0.0)),
        rationing_complementarity_violation=float(comp_viol),
        fleet_feasibility_violation=float(fleet_feas),
        history=pd.DataFrame(hist))


def multi_start_spread(case: UnifiedCase, cfg: MAGEConfig,
                       seeds=(0, 1)) -> dict:
    """GNE uniqueness is not guaranteed — report the spread across
    randomized starts instead of assuming it away."""
    finals = []
    for s in seeds:
        rng = np.random.default_rng(s)
        n_opt = len(_options(cfg))
        raw = rng.uniform(0.5, 1.5, (len(case.demand), n_opt))
        shares0 = raw / raw.sum(axis=1, keepdims=True)
        finals.append(solve_mage(case, cfg, shares0=shares0))
    spread = max(float(np.abs(a['shares'] - b['shares']).max())
                 for i, a in enumerate(finals)
                 for b in finals[i + 1:])
    return dict(spread=spread, runs=finals)


def default_two_company_config(**overrides) -> MAGEConfig:
    cfg = MAGEConfig(companies=[
        Company('alpha', fleet=400.0, av_cap=0.5,
                fare_equiv_min={'AV': 6.0, 'HV': 9.0},
                op_cost_equiv_min={'AV': 3.0, 'HV': 7.0}),
        Company('beta', fleet=250.0, av_cap=0.3,
                fare_equiv_min={'AV': 5.0, 'HV': 8.5},
                op_cost_equiv_min={'AV': 3.5, 'HV': 7.5}),
    ])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
