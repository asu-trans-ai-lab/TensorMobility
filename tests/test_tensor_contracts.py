from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from tensormobility.core.tensor_contracts import Axis, TypedOperator, TypedVector, compose


def test_typed_operator_orientation_and_mass() -> None:
    source = Axis("source", ("s0", "s1"))
    middle = Axis("middle", ("m0", "m1", "m2"))
    target = Axis("target", ("t0", "t1"))
    right = TypedOperator(
        "right",
        source,
        middle,
        sparse.csr_matrix([[1.0, 0.0], [0.0, 0.25], [0.0, 0.75]]),
        "person_flow",
        "person_flow",
        "mass_preserving",
    )
    left = TypedOperator(
        "left",
        middle,
        target,
        sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        "person_flow",
        "person_flow",
        "mass_preserving",
    )
    vector = TypedVector(source, np.asarray([4.0, 8.0]), "person_flow", "persons", "d")
    result = left.apply(right.apply(vector))
    assert np.allclose(result.values, [4.0, 8.0])
    combined = compose(left, right, name="combined")
    assert np.allclose(combined.matrix @ vector.values, result.values)


def test_axis_mismatch_is_rejected() -> None:
    a = Axis("a", ("0",))
    b = Axis("b", ("0",))
    op = TypedOperator("op", a, b, sparse.eye(1, format="csr"), "person_flow", "person_flow")
    wrong = TypedVector(b, np.asarray([1.0]), "person_flow", "persons", "wrong")
    with pytest.raises(ValueError):
        op.apply(wrong)
