"""Typed contracts for tensor states and operators (design review
2026-07-22, gaps 2.3-2.4; additive — STBTensor is unchanged and can
adopt TensorSchema incrementally).

Three objects:

- TensorSchema: what a tensor state IS (axes, measure, units, clock,
  spatial support, role, sparse support).
- AxisUse: what one OPERATOR does to one axis — roles are
  operator-relative, not global (an OD axis may be contracted by
  loading, preserved by choice, synchronized in ODME).
- OperatorContract: the full declaration an operator publishes:
  input/output schemas, per-axis use, and the certificates it emits.
- Certificate: one machine-checkable claim, serializable to the
  lab-wide certificates.json convention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

AXIS_INPUT_ROLES = ('preserved', 'contracted', 'observed', 'fixed')
AXIS_OUTPUT_ROLES = ('preserved', 'generated', 'absent')
SYNCHRONIZATION = ('none', 'fixed_point', 'dual', 'consensus')
STATE_ROLES = ('state', 'decision', 'observation', 'parameter',
               'derived')


@dataclass(frozen=True)
class TensorSchema:
    axes: tuple
    measure: str
    units: str
    role: str = 'state'
    clock: str = 'static'
    spatial_support: str = 'zone'
    support: str = 'dense'

    def __post_init__(self):
        if self.role not in STATE_ROLES:
            raise ValueError(f'unknown role {self.role!r}; expected '
                             f'one of {STATE_ROLES}')
        if len(set(self.axes)) != len(self.axes):
            raise ValueError('duplicate axis names')


@dataclass(frozen=True)
class AxisUse:
    input_role: str
    output_role: str
    synchronization: str = 'none'

    def __post_init__(self):
        if self.input_role not in AXIS_INPUT_ROLES:
            raise ValueError(f'input_role {self.input_role!r} not in '
                             f'{AXIS_INPUT_ROLES}')
        if self.output_role not in AXIS_OUTPUT_ROLES:
            raise ValueError(f'output_role {self.output_role!r} not '
                             f'in {AXIS_OUTPUT_ROLES}')
        if self.synchronization not in SYNCHRONIZATION:
            raise ValueError(f'synchronization '
                             f'{self.synchronization!r} not in '
                             f'{SYNCHRONIZATION}')


@dataclass
class OperatorContract:
    name: str
    inputs: dict
    outputs: dict
    axis_use: dict = field(default_factory=dict)
    certificates: tuple = ()

    def validate(self):
        """Contract consistency: every contracted axis appears in an
        input; every generated axis appears in an output; a preserved
        axis appears in both."""
        in_axes = set()
        for s in self.inputs.values():
            in_axes.update(s.axes)
        out_axes = set()
        for s in self.outputs.values():
            out_axes.update(s.axes)
        problems = []
        for ax, use in self.axis_use.items():
            if use.input_role != 'fixed' and ax not in in_axes:
                problems.append(f'axis {ax!r} declared '
                                f'{use.input_role} but not in any '
                                f'input schema')
            if use.output_role == 'generated' and ax not in out_axes:
                problems.append(f'axis {ax!r} declared generated but '
                                f'not in any output schema')
            if (use.input_role == 'preserved'
                    and use.output_role == 'preserved'
                    and ax not in out_axes):
                problems.append(f'axis {ax!r} preserved but absent '
                                f'from outputs')
            if use.input_role == 'contracted' and ax in out_axes:
                problems.append(f'axis {ax!r} contracted but still '
                                f'present in an output schema')
        if problems:
            raise ValueError('; '.join(problems))
        return True


@dataclass
class Certificate:
    name: str
    passed: bool
    value: float = float('nan')
    tolerance: float = float('nan')
    detail: str = ''

    def to_dict(self):
        return asdict(self)


def certificates_json(certs, **meta):
    """Serialize certificates in the lab-wide sidecar convention."""
    payload = dict(meta)
    payload['certificates'] = [
        c.to_dict() if isinstance(c, Certificate) else dict(c)
        for c in certs]
    payload['all_passed'] = all(
        (c.passed if isinstance(c, Certificate) else c.get('passed'))
        for c in certs)
    return json.dumps(payload, indent=2, sort_keys=True)
