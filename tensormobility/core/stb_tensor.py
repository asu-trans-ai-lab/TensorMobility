"""STBTensor: the named-axis space-time-behavior tensor as a real
object, not metadata.

A tensor carries data, an ordered tuple of axis KEYS (resolved against
the axis registry), and a measure. Operators contract one named axis at
a time (mode product); every other axis is a spectator and rides along
-- which is exactly the Kronecker-lift proposition of the tensor guide,
and is tested as such.

    F = STBTensor(data, axes=('od', 'group', 'departure'),
                  measure='persons')
    P = F.contract('od', B_path_od, new_axis='path')   # mode product
    F.marginal('group')                                # sum out
    F.unfold('od')                                     # matricization
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensormobility.core.axes import CANONICAL_AXES, AxisSpec


@dataclass(frozen=True)
class STBTensor:
    data: np.ndarray
    axes: tuple[str, ...]
    measure: str
    registry: dict | None = None       # axis key -> AxisSpec (default canon)

    def __post_init__(self):
        if self.data.ndim != len(self.axes):
            raise ValueError(f'{self.data.ndim}-d data vs '
                             f'{len(self.axes)} axis names')
        if len(set(self.axes)) != len(self.axes):
            raise ValueError('duplicate axis names')
        reg = self.registry or CANONICAL_AXES
        unknown = [a for a in self.axes if a not in reg]
        if unknown:
            raise ValueError(f'axes not in registry: {unknown} '
                             f'(register extensions via extend_axes)')

    def spec(self, axis: str) -> AxisSpec:
        return (self.registry or CANONICAL_AXES)[axis]

    def _pos(self, axis: str) -> int:
        try:
            return self.axes.index(axis)
        except ValueError:
            raise KeyError(f'tensor has no axis {axis!r}; '
                           f'axes = {self.axes}') from None

    # ---- contraction (mode product): sum over ONE named axis --------
    def contract(self, axis: str, operator: np.ndarray,
                 new_axis: str, measure: str | None = None
                 ) -> 'STBTensor':
        """Mode product along `axis` with operator[new, old]; all other
        axes are spectators."""
        k = self._pos(axis)
        if operator.shape[1] != self.data.shape[k]:
            raise ValueError(f'operator columns {operator.shape[1]} != '
                             f'axis {axis!r} size {self.data.shape[k]}')
        out = np.tensordot(operator, self.data, axes=([1], [k]))
        # tensordot puts the new axis first; move it back to position k
        out = np.moveaxis(out, 0, k)
        axes = tuple(new_axis if i == k else a
                     for i, a in enumerate(self.axes))
        return STBTensor(out, axes, measure or self.measure,
                         self.registry)

    def marginal(self, axis: str) -> 'STBTensor':
        k = self._pos(axis)
        axes = tuple(a for a in self.axes if a != axis)
        return STBTensor(self.data.sum(axis=k), axes, self.measure,
                         self.registry)

    def unfold(self, axis: str) -> np.ndarray:
        """Matricization: `axis` fibers as rows (size_axis x rest)."""
        k = self._pos(axis)
        moved = np.moveaxis(self.data, k, 0)
        return moved.reshape(self.data.shape[k], -1)

    def total(self) -> float:
        return float(self.data.sum())

    def __repr__(self):
        dims = ' x '.join(f'{a}[{n}]'
                          for a, n in zip(self.axes, self.data.shape))
        return f'STBTensor({dims}; {self.measure})'


def kronecker_lift(operator: np.ndarray, spectator_size: int
                   ) -> np.ndarray:
    """The guide's spectator proposition, literally: contracting one
    axis while S spectator cells ride along equals applying
    operator (x) I_S to the unfolded tensor. Provided for tests and
    teaching; production code never materializes it."""
    return np.kron(operator, np.eye(spectator_size))
