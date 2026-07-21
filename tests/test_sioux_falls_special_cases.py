from __future__ import annotations

import numpy as np

from tensormobility.dta.algorithms import solve_exact_fw, solve_logit_sue_msa
from tensormobility.core.flow_through import build_flow_through_tensors
from tensormobility.dynamics.fluid_queue import build_departure_tensor, run_fluid_point_queue, smooth_departure_profile
from tensormobility.dta.sioux_falls import build_sioux_falls_path_set


def test_sioux_falls_dimensions() -> None:
    sf = build_sioux_falls_path_set(k_paths=4)
    assert len(sf.nodes) == 24
    assert len(sf.links) == 76
    assert sf.instance.n_groups == 33
    assert sf.instance.n_columns == 132
    assert sf.instance.A.shape == (76, 132)


def test_logit_and_ue_are_feasible() -> None:
    sf = build_sioux_falls_path_set(k_paths=4, demand_scale=0.25)
    logit = solve_logit_sue_msa(sf.instance, temperature=1.0, max_iterations=220, tolerance=1e-5)
    ue = solve_exact_fw(sf.instance, max_iterations=220, tolerance=1e-5)
    assert logit.feasibility_residual < 1e-8
    assert ue.feasibility_residual < 1e-8
    assert np.isfinite(logit.objective)
    assert np.isfinite(ue.objective)


def test_flow_through_exact_reconstruction() -> None:
    sf = build_sioux_falls_path_set(k_paths=4, demand_scale=0.2)
    f = np.zeros(sf.instance.n_columns)
    for demand, idx in zip(sf.instance.demands, sf.instance.group_columns, strict=True):
        f[idx] = demand / len(idx)
    result = build_flow_through_tensors(sf.instance, f)
    assert result.mass_residual_zone_od < 1e-10
    assert result.mass_residual_od_path < 1e-10
    assert result.link_reconstruction_residual < 1e-10


def test_fluid_queue_conservation_and_shock() -> None:
    sf = build_sioux_falls_path_set(k_paths=4, demand_scale=0.15)
    f = np.zeros(sf.instance.n_columns)
    for demand, idx in zip(sf.instance.demands, sf.instance.group_columns, strict=True):
        f[idx] = demand / len(idx)
    dep = build_departure_tensor(f, smooth_departure_profile())
    base = run_fluid_point_queue(sf, dep, horizon_steps=72)
    mult = np.ones(sf.instance.n_resources)
    mult[list(sf.instance.resource_labels).index("10->16")] = 0.35
    shock = run_fluid_point_queue(sf, dep, horizon_steps=72, capacity_multiplier=mult)
    assert np.max(np.abs(base.conservation_residual)) < 1e-7
    assert np.max(np.abs(shock.conservation_residual)) < 1e-7
    assert shock.max_total_queue >= base.max_total_queue
