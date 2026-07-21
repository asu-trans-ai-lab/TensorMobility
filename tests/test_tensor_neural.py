"""The tensor nature and the neural identity, asserted:
STBTensor named-axis contraction == Kronecker-lifted matrix algebra;
softmax router == logit choice; the TCG chain is a neural network with
FD-certified hand-backprop."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.stb_tensor import STBTensor, kronecker_lift
from tensormobility.neural import (softmax_router, logit_choice,
                                   router_is_logit, routed_reduced_cost,
                                   hard_routing_limit,
                                   masked_softmax_rows, forward,
                                   loss_and_grad, fd_check)


def test_stb_tensor_contraction_is_kronecker_lifted_matvec():
    """The guide's spectator proposition: contracting the od axis with
    B while (group, departure) ride along equals (B kron I_S) applied to
    the unfolded tensor."""
    rng = np.random.default_rng(0)
    n_od, n_g, n_tau, n_p = 6, 3, 4, 9
    F = STBTensor(rng.uniform(1, 10, (n_od, n_g, n_tau)),
                  axes=('od', 'group', 'departure'), measure='persons')
    B = rng.uniform(0, 1, (n_p, n_od))          # path <- od operator
    Fp = F.contract('od', B, new_axis='path')
    assert Fp.axes == ('path', 'group', 'departure')
    # Kronecker-lift equivalence on the unfolded representation
    lifted = kronecker_lift(B, n_g * n_tau)
    ref = lifted @ F.unfold('od').reshape(-1)
    assert np.allclose(Fp.unfold('path').reshape(-1), ref)
    # spectator axes untouched: marginals over path equal B-mapped totals
    assert np.isclose(Fp.total(), (B @ F.unfold('od')).sum())


def test_stb_tensor_contract_conserves_mass_for_stochastic_operator():
    rng = np.random.default_rng(1)
    F = STBTensor(rng.uniform(0, 5, (4, 2)), axes=('od', 'group'),
                  measure='persons')
    M = rng.uniform(0, 1, (7, 4))
    M = M / M.sum(axis=0, keepdims=True)        # column-stochastic
    Fp = F.contract('od', M, new_axis='path')
    assert np.isclose(Fp.total(), F.total())    # conserved operator


def test_stb_tensor_rejects_unknown_axis():
    with pytest.raises(ValueError):
        STBTensor(np.zeros((2, 2)), axes=('od', 'not_an_axis'),
                  measure='persons')


def test_router_is_logit_identity():
    rng = np.random.default_rng(2)
    for theta in (0.1, 1.0, 7.5):
        costs = rng.uniform(5, 40, 8)
        assert router_is_logit(costs, theta)
    # zero-temperature limit: top-1 routing == all-or-nothing
    shares = hard_routing_limit(np.array([12.0, 10.0, 11.0]))
    assert np.all(np.diff(shares) >= -1e-12) and shares[-1] > 0.999


def test_reduced_cost_is_dual_conditioned_router_score():
    rng = np.random.default_rng(3)
    n_link, n_path = 5, 4
    A = (rng.uniform(0, 1, (n_link, n_path)) > 0.5).astype(float)
    c = rng.uniform(1, 10, n_path)
    lam = rng.uniform(0, 2, n_link)
    rc = routed_reduced_cost(c, A, lam)
    # admission by router score == admission by negative reduced cost
    scores = -rc
    assert np.array_equal(scores > 0, rc < 0)


def test_tcg_chain_is_a_certified_neural_network():
    """Forward = conserved flow propagation; backward = FD-certified
    hand-backprop through the masked-softmax stack."""
    rng = np.random.default_rng(4)
    n_z, n_od, n_p, n_l = 5, 8, 14, 10
    # every zone owns >=1 OD and every OD owns >=1 path, else mass
    # legitimately vanishes through empty softmax rows
    owner_od = np.concatenate([np.arange(n_z),
                               rng.integers(0, n_z, n_od - n_z)])
    M1 = np.zeros((n_z, n_od))
    M1[owner_od, np.arange(n_od)] = 1.0
    owner_p = np.concatenate([np.arange(n_od),
                              rng.integers(0, n_od, n_p - n_od)])
    M2 = np.zeros((n_od, n_p))
    M2[owner_p, np.arange(n_p)] = 1.0
    Delta = (rng.uniform(0, 1, (n_p, n_l)) > 0.6).astype(float)
    survey = rng.uniform(50, 150, n_z)
    split_t = masked_softmax_rows(rng.normal(0, 1, (n_z, n_od)), M1)
    count = rng.uniform(20, 80, n_l)
    observed = rng.uniform(0, 1, n_l) > 0.3
    theta = np.concatenate([survey + rng.normal(0, 5, n_z),
                            rng.normal(0, .3, n_z * n_od),
                            rng.normal(0, .3, n_od * n_p)])
    # conservation through the stack
    G = masked_softmax_rows(theta[n_z:n_z + n_z * n_od].reshape(n_z,
                                                                n_od), M1)
    R = masked_softmax_rows(theta[n_z + n_z * n_od:].reshape(n_od,
                                                             n_p), M2)
    q, f, v = forward(theta[:n_z], G, R, Delta)
    assert np.isclose(f.sum(), theta[:n_z].sum())   # mass conserved
    # FD-certified backprop (meaningful coordinates only)
    err = fd_check(theta, (n_z, n_od, n_p), M1, M2, Delta, survey,
                   split_t, count, observed, n_checks=12)
    assert err < 1e-5
