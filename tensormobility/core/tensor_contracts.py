from __future__ import annotations

"""Typed sparse operator contracts for STB-FTT.

The project intentionally avoids anonymous high-dimensional arrays.  Every
mass-carrying vector has one named axis and every transformation has an
explicit source axis, target axis, orientation, unit semantics, and optional
conservation contract.  Higher-dimensional transportation objects are stored
as a table of feasible columns plus a chain of sparse two-axis operators.
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy import sparse

MeasureKind = Literal["probability", "person_flow", "vehicle_flow", "state", "cost", "incidence"]
ConservationKind = Literal["none", "column_stochastic", "row_stochastic", "mass_preserving"]


@dataclass(frozen=True)
class Axis:
    name: str
    labels: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Axis name cannot be empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"Axis {self.name!r} contains duplicate labels")

    @property
    def size(self) -> int:
        return len(self.labels)

    @classmethod
    def indexed(cls, name: str, size: int, prefix: str | None = None, description: str = "") -> "Axis":
        if size < 0:
            raise ValueError("Axis size must be nonnegative")
        p = prefix or name
        return cls(name=name, labels=tuple(f"{p}:{i}" for i in range(size)), description=description)


@dataclass(frozen=True)
class TypedVector:
    axis: Axis
    values: np.ndarray
    measure: MeasureKind
    unit: str
    name: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (self.axis.size,):
            raise ValueError(
                f"Vector {self.name!r} has shape {values.shape}; expected {(self.axis.size,)} on axis {self.axis.name!r}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Vector {self.name!r} contains non-finite values")
        object.__setattr__(self, "values", values)

    def total(self) -> float:
        return float(self.values.sum())


@dataclass(frozen=True)
class TypedOperator:
    """Sparse linear operator with explicit orientation target <- source."""

    name: str
    source: Axis
    target: Axis
    matrix: sparse.csr_matrix
    input_measure: MeasureKind
    output_measure: MeasureKind
    conservation: ConservationKind = "none"
    description: str = ""

    def __post_init__(self) -> None:
        mat = sparse.csr_matrix(self.matrix, dtype=float)
        expected = (self.target.size, self.source.size)
        if mat.shape != expected:
            raise ValueError(
                f"Operator {self.name!r} has shape {mat.shape}; expected {expected} for {self.target.name} <- {self.source.name}"
            )
        if mat.nnz and not np.all(np.isfinite(mat.data)):
            raise ValueError(f"Operator {self.name!r} contains non-finite values")
        object.__setattr__(self, "matrix", mat)
        self.validate_contract()

    def validate_contract(self, tolerance: float = 1e-10) -> None:
        if self.conservation == "none":
            return
        if self.matrix.nnz and np.min(self.matrix.data) < -tolerance:
            raise ValueError(f"Conserving operator {self.name!r} has negative coefficients")
        if self.conservation in {"column_stochastic", "mass_preserving"}:
            sums = np.asarray(self.matrix.sum(axis=0)).ravel()
            active = np.asarray(self.matrix.getnnz(axis=0)).ravel() > 0
            if np.any(np.abs(sums[active] - 1.0) > tolerance):
                raise ValueError(f"Columns of operator {self.name!r} do not sum to one")
        if self.conservation == "row_stochastic":
            sums = np.asarray(self.matrix.sum(axis=1)).ravel()
            active = np.asarray(self.matrix.getnnz(axis=1)).ravel() > 0
            if np.any(np.abs(sums[active] - 1.0) > tolerance):
                raise ValueError(f"Rows of operator {self.name!r} do not sum to one")

    def apply(self, vector: TypedVector, *, output_name: str | None = None, unit: str | None = None) -> TypedVector:
        if vector.axis != self.source:
            raise ValueError(
                f"Operator {self.name!r} expects source axis {self.source.name!r}, received {vector.axis.name!r}"
            )
        if vector.measure != self.input_measure:
            raise ValueError(
                f"Operator {self.name!r} expects {self.input_measure}, received {vector.measure}"
            )
        values = np.asarray(self.matrix @ vector.values).ravel()
        result = TypedVector(
            axis=self.target,
            values=values,
            measure=self.output_measure,
            unit=unit or vector.unit,
            name=output_name or f"{self.name}({vector.name})",
        )
        if self.conservation == "mass_preserving":
            residual = abs(result.total() - vector.total())
            scale = max(abs(vector.total()), 1.0)
            if residual / scale > 1e-9:
                raise ValueError(f"Mass-preserving operator {self.name!r} produced relative residual {residual / scale:.3e}")
        return result

    @property
    def density(self) -> float:
        denom = self.target.size * self.source.size
        return float(self.matrix.nnz / denom) if denom else 0.0

    @property
    def estimated_bytes(self) -> int:
        # CSR data + indices + indptr.  This estimate is deterministic and
        # avoids relying on implementation-specific Python object overhead.
        return int(self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes)


def compose(left: TypedOperator, right: TypedOperator, *, name: str, description: str = "") -> TypedOperator:
    """Return left o right, preserving explicit axis order.

    right: middle <- source
    left:  target <- middle
    result: target <- source
    """
    if right.target != left.source:
        raise ValueError(
            f"Cannot compose {left.name} after {right.name}: axis {right.target.name!r} != {left.source.name!r}"
        )
    return TypedOperator(
        name=name,
        source=right.source,
        target=left.target,
        matrix=(left.matrix @ right.matrix).tocsr(),
        input_measure=right.input_measure,
        output_measure=left.output_measure,
        conservation="none",
        description=description,
    )


def one_hot_operator(
    *,
    name: str,
    source: Axis,
    target: Axis,
    target_index_for_source: Sequence[int],
    weights: Sequence[float] | None = None,
    input_measure: MeasureKind,
    output_measure: MeasureKind,
    conservation: ConservationKind = "mass_preserving",
    description: str = "",
) -> TypedOperator:
    index = np.asarray(target_index_for_source, dtype=int)
    if index.shape != (source.size,):
        raise ValueError("target_index_for_source has wrong shape")
    if np.any(index < 0) or np.any(index >= target.size):
        raise ValueError("target index out of range")
    data = np.ones(source.size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if data.shape != (source.size,):
        raise ValueError("weights have wrong shape")
    matrix = sparse.coo_matrix((data, (index, np.arange(source.size))), shape=(target.size, source.size)).tocsr()
    return TypedOperator(
        name=name,
        source=source,
        target=target,
        matrix=matrix,
        input_measure=input_measure,
        output_measure=output_measure,
        conservation=conservation,
        description=description,
    )


def sparse_operator_from_triplets(
    *,
    name: str,
    source: Axis,
    target: Axis,
    rows: Iterable[int],
    cols: Iterable[int],
    values: Iterable[float],
    input_measure: MeasureKind,
    output_measure: MeasureKind,
    conservation: ConservationKind = "none",
    description: str = "",
) -> TypedOperator:
    matrix = sparse.coo_matrix(
        (np.fromiter(values, dtype=float), (np.fromiter(rows, dtype=int), np.fromiter(cols, dtype=int))),
        shape=(target.size, source.size),
    ).tocsr()
    return TypedOperator(
        name=name,
        source=source,
        target=target,
        matrix=matrix,
        input_measure=input_measure,
        output_measure=output_measure,
        conservation=conservation,
        description=description,
    )
