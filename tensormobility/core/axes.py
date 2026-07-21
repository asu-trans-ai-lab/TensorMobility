"""Streamlined multi-axis registry for STB-FTT v0.5 (executable core of
DESIGN_v0_5_REVIEW.md).

The streamline: instead of an open-ended list of tensor dimensions, every
axis carries exactly three declarations —

  status    SPECTATOR    rides along (Kronecker lift; coordination)
            CONTRACTED   summed out by a typed operator
            SYNCHRONIZED pinned by a fixed point / dual variable
  semiring  how it is contracted: (+,x) flow, (min,+) pricing,
            (max,x) state decode
  anchor    which literature layer owns it (S_repr / S_stat / S_mech /
            S_ctrl, from the structure-is-all-you-need mapping)

A solver is a choice of which axes are promoted from SPECTATOR to
SYNCHRONIZED; a compression is a low-rank statement about SPECTATOR
axes; a learned term is admissible only if it is flow-independent along
every SYNCHRONIZED axis (clean-VI boundary).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    SPECTATOR = 'spectator'
    CONTRACTED = 'contracted'
    SYNCHRONIZED = 'synchronized'


class Semiring(Enum):
    SUM_PRODUCT = '(+,x)'
    MIN_PLUS = '(min,+)'
    MAX_PRODUCT = '(max,x)'


@dataclass(frozen=True)
class AxisSpec:
    key: str
    description: str
    status: Status
    semiring: Semiring
    anchor: str          # S_repr | S_stat | S_mech | S_ctrl | Observation
    pinned_by: str = ''  # required iff SYNCHRONIZED

    def __post_init__(self):
        if self.status is Status.SYNCHRONIZED and not self.pinned_by:
            raise ValueError(f'axis {self.key}: SYNCHRONIZED requires '
                             f'pinned_by (which fixed point/dual pins it)')
        if self.status is not Status.SYNCHRONIZED and self.pinned_by:
            raise ValueError(f'axis {self.key}: pinned_by only for '
                             f'SYNCHRONIZED axes')


# The eight canonical axes of the STB grand tensor + supply layers.
# Statuses are for the STATIC assignment slice; promote() derives the
# other model slices.
CANONICAL_AXES = {
    'od': AxisSpec('od', 'origin-destination pair', Status.CONTRACTED,
                   Semiring.SUM_PRODUCT, 'S_mech'),
    'path': AxisSpec('path', 'route / behavioral column',
                     Status.CONTRACTED, Semiring.SUM_PRODUCT, 'S_mech'),
    'link': AxisSpec('link', 'link-period resource cell',
                     Status.SYNCHRONIZED, Semiring.SUM_PRODUCT, 'S_mech',
                     pinned_by='congestion fixed point (UE/SUE)'),
    'group': AxisSpec('group', 'traveler class / demographic',
                      Status.SPECTATOR, Semiring.SUM_PRODUCT, 'S_stat'),
    'mode': AxisSpec('mode', 'travel mode', Status.SPECTATOR,
                     Semiring.SUM_PRODUCT, 'S_stat'),
    'layer': AxisSpec('layer', 'commodity block (passenger / freight / '
                      'transit operator)', Status.SPECTATOR,
                      Semiring.SUM_PRODUCT, 'S_ctrl'),
    'departure': AxisSpec('departure', 'departure period',
                          Status.SPECTATOR, Semiring.SUM_PRODUCT,
                          'S_mech'),
    'state': AxisSpec('state', 'traffic regime / FD phase',
                      Status.SPECTATOR, Semiring.MAX_PRODUCT, 'S_repr'),
}


def promote(axes: dict[str, AxisSpec], key: str, pinned_by: str
            ) -> dict[str, AxisSpec]:
    """Return a new axis map with one axis promoted to SYNCHRONIZED —
    the formal act of choosing a richer model slice."""
    a = axes[key]
    out = dict(axes)
    out[key] = AxisSpec(a.key, a.description, Status.SYNCHRONIZED,
                        a.semiring, a.anchor, pinned_by)
    return out


# model slices as promotion chains (executable documentation)
def slice_static_ue():
    return dict(CANONICAL_AXES)


def slice_dta():
    return promote(CANONICAL_AXES, 'departure',
                   'queue conservation recursion (point queue)')


def slice_multilayer():
    return promote(CANONICAL_AXES, 'layer',
                   'shared-resource price consensus (lambda / t(v))')


def slice_state_estimation():
    return promote(CANONICAL_AXES, 'state',
                   'Viterbi (max,x) decode consistency')


def extend_axes(base: dict[str, AxisSpec],
                extra: dict[str, AxisSpec]) -> dict[str, AxisSpec]:
    """Register EXTENSION axes on top of a base map without mutating the
    canonical registry — the mechanism by which an operator profile
    (e.g. mixed-autonomy ride-hailing) adds problem-specific axes."""
    dup = set(base) & set(extra)
    if dup:
        raise ValueError(f'extension axes clash with base: {sorted(dup)}')
    out = dict(base)
    out.update(extra)
    return out


# Mixed-autonomy ride-hailing (MAGE profile) extension axes
MAGE_EXTENSION = {
    'company': AxisSpec('company', 'competing TNC', Status.SPECTATOR,
                        Semiring.SUM_PRODUCT, 'S_ctrl'),
    'vtype': AxisSpec('vtype', 'vehicle type AV / HV / SV',
                      Status.SPECTATOR, Semiring.SUM_PRODUCT, 'S_mech'),
    'stage': AxisSpec('stage', 'operational stage pickup / service',
                      Status.CONTRACTED, Semiring.SUM_PRODUCT, 'S_mech'),
    'match': AxisSpec('match', 'customer matching queue (distinct from '
                      'the road queue)', Status.SYNCHRONIZED,
                      Semiring.SUM_PRODUCT, 'S_mech',
                      pinned_by='patience-capped waiting fixed point'),
}


def slice_mage():
    """MAGE steady-state profile: canonical axes + ride-hailing
    extension; road congestion and matching queue both synchronized."""
    return extend_axes(dict(CANONICAL_AXES), MAGE_EXTENSION)


def clean_vi_ok(axes: dict[str, AxisSpec],
                learned_depends_on: set[str]) -> bool:
    """Clean-VI admissibility: a learned term may not depend on any
    SYNCHRONIZED axis' equilibrium quantity."""
    sync = {k for k, a in axes.items()
            if a.status is Status.SYNCHRONIZED}
    return not (learned_depends_on & sync)
