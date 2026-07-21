"""Land-use core + typed contracts (design review P0)."""
import json

import numpy as np
import pytest

from tensormobility.core.schema import (AxisUse, Certificate,
                                        OperatorContract,
                                        TensorSchema,
                                        certificates_json)
from tensormobility.landuse import (location_choice,
                                    logsum_accessibility,
                                    step_household_cohorts,
                                    step_stock)


# ---------------- land use ----------------

def test_stock_flow_conservation_exact():
    L = np.array([[100.0, 50.0], [80.0, 20.0]])
    D = np.array([[10.0, 0.0], [5.0, 2.0]])
    R = np.array([[3.0, 1.0], [0.0, 0.5]])
    L2, cert = step_stock(L, D, R)
    assert cert['passed'] and cert['residual'] < 1e-12
    assert L2.sum() == pytest.approx(L.sum() + (D - R).sum())


def test_stock_flow_rejects_negative():
    with pytest.raises(ValueError):
        step_stock([[1.0]], [[0.0]], [[5.0]])


def test_cohort_update():
    H, cert = step_household_cohorts([100.0], [5.0], [3.0], [2.0],
                                     [1.0])
    assert H[0] == pytest.approx(103.0) and cert['passed']


def test_accessibility_monotone_in_opportunities():
    c = np.array([[0.0, 1.0], [1.0, 0.0]])
    A1, _ = logsum_accessibility([10.0, 10.0], c, beta=0.5)
    A2, _ = logsum_accessibility([10.0, 20.0], c, beta=0.5)
    assert (A2 >= A1 - 1e-12).all() and A2[1] > A1[1]


def test_accessibility_monotone_in_cost():
    E = [10.0, 10.0]
    A1, _ = logsum_accessibility(E, [[0.0, 2.0], [2.0, 0.0]])
    A2, _ = logsum_accessibility(E, [[0.0, 1.0], [2.0, 0.0]])
    assert A2[0] > A1[0] and A2[1] == pytest.approx(A1[1])


def test_location_choice_rows_sum_to_one_and_mask():
    V = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    feas = np.array([[True, True, False], [True, True, True]])
    P, info = location_choice(V, feasible=feas)
    assert all(c['passed'] for c in info['certificates'])
    assert P[0, 2] == 0.0
    assert P.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_location_choice_capacity_rationing():
    V = np.zeros((1, 2))                       # 50/50 preference
    P, info = location_choice(V, demand=[100.0],
                              capacity=[30.0, 100.0])
    alloc = info['allocation']
    assert alloc[0, 0] == pytest.approx(30.0)  # rationed to capacity
    assert info['unplaced'][0] == pytest.approx(20.0)
    assert all(c['passed'] for c in info['certificates'])


# ---------------- schema / contracts ----------------

def test_axis_use_validation():
    with pytest.raises(ValueError):
        AxisUse(input_role='bogus', output_role='preserved')
    AxisUse(input_role='contracted', output_role='absent',
            synchronization='consensus')


def test_operator_contract_catches_leaked_axis():
    demand = TensorSchema(axes=('od', 'tau'), measure='trips',
                          units='veh/hr')
    flow = TensorSchema(axes=('od', 'tau', 'path'), measure='flow',
                        units='veh/hr')
    bad = OperatorContract(
        name='loading', inputs={'F': flow}, outputs={'D': demand},
        axis_use={'path': AxisUse('contracted', 'absent')})
    # 'path' is contracted yet demand omits it -> valid; now leak it:
    leaky = OperatorContract(
        name='leaky', inputs={'F': flow}, outputs={'F2': flow},
        axis_use={'path': AxisUse('contracted', 'absent')})
    assert bad.validate()
    with pytest.raises(ValueError):
        leaky.validate()


def test_contract_generated_axis_must_appear():
    demand = TensorSchema(axes=('od',), measure='trips', units='veh')
    flow = TensorSchema(axes=('od', 'path'), measure='flow',
                        units='veh')
    good = OperatorContract(
        name='colgen', inputs={'D': demand}, outputs={'F': flow},
        axis_use={'path': AxisUse('fixed', 'generated'),
                  'od': AxisUse('preserved', 'preserved')})
    assert good.validate()


def test_certificates_json_roundtrip():
    s = certificates_json(
        [Certificate('gap', True, 9.8e-5, 1e-4),
         {'name': 'conservation', 'passed': True,
          'residual': 1e-13}],
        case='grid_10x10')
    d = json.loads(s)
    assert d['all_passed'] and d['case'] == 'grid_10x10'
    assert len(d['certificates']) == 2
