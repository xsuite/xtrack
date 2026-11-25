# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Apply host-side Xdeps knob commands at stopped tracking boundaries."""

from inspect import getattr_static

import numpy as np
from xobjects import Array
from xobjects.hybrid_class import _FieldOfDressed
from xobjects.scalar import is_scalar
from xdeps.refs import AttrRef, ItemRef
from xdeps.tasks import ExprTask

from ..base_element import FloatOrTpsa, Replica, _FieldOfBeamElement
from ..beam_elements import Bend, Cavity


# These setters have a known write set, including the dipole curvature link.
_SETTER_FIELDS = {
    Bend.k0: ('k0', 'k0_from_h'),
    Cavity.frequency: ('frequency',),
    Cavity.harmonic: ('harmonic',),
}

_REFERENCE_FIELDS = {
    'length', 'length_straight', 'angle', 'h', 'p0c', 'delta_p0c',
    'energy_ref_increment',
}


def _check_time_dependent_geometry(line):
    """Protect the cached full-turn transport geometry from Xdeps updates."""
    if not line.enable_time_dependent_vars:
        return
    env = line.env
    names = set(line.element_names)
    for name in line.element_names:
        element = line.get(name)
        if isinstance(element, Replica):
            names.add(element.resolve(line, get_name=True))
    time_ref = env._xdeps_vref['t_turn_s']
    for task in time_ref._manager.find_tasks([time_ref]):
        target = task.taskid
        if isinstance(target, ItemRef) and isinstance(target._owner, AttrRef):
            target = target._owner
        if (isinstance(target, AttrRef)
                and isinstance(target._owner, ItemRef)
                and target._owner._owner is env._xdeps_eref
                and target._owner._key in names
                and target._key.lstrip('_').lower() in _REFERENCE_FIELDS):
            raise NotImplementedError(
                'Feedback cannot use a time-dependent reference geometry or '
                f'element-driven energy change ({target})')


def _host_array(values):
    """Explicit transfer for the coherent host/Xdeps output path."""
    if hasattr(values, 'to_nparray'):
        values = values.to_nparray()
    elif hasattr(values, 'get'):
        values = values.get()
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.number) or np.iscomplexobj(result):
        raise TypeError('Feedback commands must be real numerical values')
    return result.astype(np.float64, copy=True)


def _limits(lower, upper, size):
    def expand(value, default):
        if value is None:
            return np.full(size, default)
        array = _host_array(value)
        if array.shape == ():
            return np.full(size, float(array))
        if array.shape != (size,):
            raise ValueError(f'Expected scalar or ({size},) output limits')
        return array

    lower, upper = expand(lower, -np.inf), expand(upper, np.inf)
    if (
        np.any(np.isnan(lower)) or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise ValueError('Output limits must be ordered and cannot contain NaN')
    return lower, upper


class KnobOutput:
    """Replace independent host knobs through ordinary Xdeps assignments.

    Parameters
    ----------
    env_or_line : Environment or Line
        Owner of existing independent numerical variables.
    targets : str or sequence of str
        Names of independent knobs.
    lower, upper : scalar or array, optional
        Bounds in the units of the corresponding knob.

    Writes use ordinary Xdeps assignment in the order given. Intermediate
    commands must be admissible. The dependency graph must use pure
    expressions and remain fixed with tracking stopped during each write.
    Dependent variables must be finite real scalars or flags. Element
    strengths use Xtrack's ordinary scalar storage.
    """

    def __init__(
        self, env_or_line, targets, lower=None, upper=None, *, _read_only=False,
    ):
        self.owner = env_or_line
        self.targets = (
            (targets,) if isinstance(targets, str) else tuple(targets)
        )
        if not self.targets or any(
            not isinstance(name, str) for name in self.targets
        ):
            raise ValueError('KnobOutput needs one or more existing knob names')
        if len(set(self.targets)) != len(self.targets):
            raise ValueError('Output knob targets must be unique')
        self._vars = env_or_line.vars
        self._read_only = _read_only
        self.lower, self.upper = _limits(lower, upper, len(self.targets))
        self._field_objects = None
        self.snapshot()

    @property
    def ownership_keys(self):
        """Identify the input knobs within their Xdeps owner."""
        return tuple(('knob', id(self._vars), name) for name in self.targets)

    @property
    def dependent_field_keys(self):
        """Identify the physical storage locations written by the graph."""
        self._check_ownership()
        return tuple(sorted(self._field_objects))

    @property
    def _dependent_variable_keys(self):
        """Return the computed variables from the last ownership check."""
        return tuple(
            ('knob', id(self._vars), ref._key) for ref in self._dependent_vars
        )

    def _find_dependencies(self, sources):
        """Resolve the Xdeps write set for a knob assignment."""
        env = self._vars.env
        root_elements = env._xdeps_eref
        root_vars = env._xdeps_vref
        variables = set()
        objects = {}
        for task in root_vars._manager.find_tasks(sources):
            if type(task) is not ExprTask:
                raise NotImplementedError(
                    'Feedback output requires ordinary Xdeps ExprTasks')
            ref = task.taskid
            if (
                isinstance(ref, ItemRef) and ref._owner is root_vars
                and isinstance(ref._key, str)
            ):
                variables.add(ref)
                continue
            indexed = isinstance(ref, ItemRef)
            index = ref._key if indexed else None
            attr = ref._owner if indexed else ref
            if (
                not isinstance(attr, AttrRef)
                or not isinstance(attr._owner, ItemRef)
                or attr._owner._owner is not root_elements
                or not isinstance(attr._owner._key, str)
                or not isinstance(attr._key, str)
            ):
                raise NotImplementedError(
                    'Feedback requires scalar variables or direct element fields')
            element = env._element_dict[attr._owner._key]
            if isinstance(element, Replica):
                raise NotImplementedError(
                    'Bind the physical field on the Replica parent')
            for key, field, normalized_index in self._element_fields(
                element, attr._key, index, indexed,
            ):
                if key in objects:
                    raise NotImplementedError(
                        'Multiple expressions write the same physical field')
                objects[key] = (element, field, normalized_index)
        return variables, objects

    @staticmethod
    def _element_fields(element, name, index, indexed):
        """Resolve the storage written by a supported element attribute."""
        descriptor = getattr_static(type(element), name, None)
        if isinstance(descriptor, (_FieldOfDressed, _FieldOfBeamElement)):
            fields = (descriptor.name,)
        elif not indexed and isinstance(descriptor, property) and (
            descriptor in _SETTER_FIELDS
        ):
            fields = _SETTER_FIELDS[descriptor]
        else:
            raise NotImplementedError(
                f'Cannot resolve feedback writes through '
                f'{type(element).__name__}.{name}')

        # Use native offsets without reading a scalar or kick value.
        xobject = element._xobject
        for field in fields:
            native_field = getattr(type(xobject), field)
            if native_field.readonly:
                raise NotImplementedError(f'Feedback field {field} is read-only')
            field_type, offset = native_field.get_offset(xobject)
            if indexed:
                if isinstance(index, bool) or not isinstance(
                    index, (int, np.integer),
                ):
                    raise NotImplementedError(
                        'Dependent field indices must be fixed integers')
                storage = getattr(xobject, field)
                if not isinstance(storage, Array) or len(storage._shape) != 1:
                    raise NotImplementedError(
                        'Indexed dependent fields must be one-dimensional')
                index = int(index)
                if index < 0:
                    index += len(storage)
                offset = storage._get_offset(index)
                field_type = storage._itemtype
            curvature_link = len(fields) == 2 and field == 'k0_from_h'
            if not (curvature_link or field_type is FloatOrTpsa or (
                is_scalar(field_type)
                and np.issubdtype(field_type._dtype, np.floating)
            )):
                raise NotImplementedError(
                    'Feedback element targets must be floating-point fields')
            yield ('element-field', id(element._buffer), int(offset)), field, index

    def _check_ownership(self):
        if self.owner.vars is not self._vars:
            raise RuntimeError('Output environment changed; bind a new output')
        if self._vars.vars_to_update:
            raise NotImplementedError(
                'Feedback output cannot use linked variable setters'
            )
        if not self._read_only and 't_turn_s' in self.targets:
            raise ValueError('Feedback cannot write the native t_turn_s clock')
        sources = []
        for name in self.targets:
            if name not in self._vars:
                raise KeyError(f'Feedback output knob {name!r} does not exist')
            source = self._vars[name]
            if not self._read_only and source._tasks:
                raise ValueError(
                    f'Output knob {name!r} is expression-defined '
                    'or has an Xdeps writer')
            sources.append(source)
        variables, objects = self._find_dependencies(sources)
        if not self._read_only and any(
            ref._key == 't_turn_s' for ref in variables
        ):
            raise ValueError('Feedback cannot write the native t_turn_s clock')
        if self._field_objects is not None:
            old_targets = {
                (id(el), field, index)
                for el, field, index in self._field_objects.values()
            }
            new_targets = {
                (id(el), field, index) for el, field, index in objects.values()
            }
            if old_targets != new_targets or variables != self._dependent_vars:
                raise RuntimeError(
                    'Output dependency graph changed; recreate the binding'
                )
        # Refresh storage keys after an element moves to a tracker buffer.
        self._field_objects = objects
        self._dependent_vars = variables
        self._sources = tuple(sources)

    def _check_reference_fields(self):
        """Reject outputs that change the reference geometry or momentum."""
        if any(field.lstrip('_').lower() in _REFERENCE_FIELDS
               for _, field, _ in self._field_objects.values()):
            raise NotImplementedError(
                'Full-turn feedback cannot drive reference geometry or energy')

    def read(self):
        """Return an owned host vector of the current knob values."""
        self._check_ownership()
        return self._read_values()

    def _read_values(self):
        return _host_array([self._vars.val[name] for name in self.targets])

    def _dependent_state(self):
        variables = tuple((ref, ref._get_value()) for ref in self._dependent_vars)
        for ref, value in variables:
            scalar = np.asarray(value)
            if (
                scalar.ndim != 0 or scalar.dtype.kind not in 'biuf'
                or not np.isfinite(scalar)
            ):
                raise ValueError(
                    f'Feedback propagation requires a finite real scalar at {ref}')
        fields = []
        for element, field, index in self._field_objects.values():
            xobject = element._xobject
            strength = getattr(type(xobject), field).ftype is FloatOrTpsa
            if strength and xobject._tpsa_enabled:
                raise NotImplementedError(
                    'Feedback requires scalar mode for TPSA fields')
            value = getattr(xobject, field)
            if index is not None:
                value = value[index]
            if not bool(np.isfinite(value)):
                raise ValueError(
                    f'Feedback propagation produced a nonfinite field {field}')
            fields.append((element, field, index, value, strength))
        return variables, fields

    def snapshot(self):
        """Return the current host knob vector after validating its write set."""
        self._check_ownership()
        values = self._validate_values(self._read_values())
        self._dependent_state()
        return values

    def validate(self, values):
        """Validate the command and return its owned host vector."""
        values = self._validate_values(values)
        self._check_ownership()
        return values

    def _validate_values(self, values):
        values = _host_array(values)
        if values.shape == () and len(self.targets) == 1:
            values = values.reshape(1)
        if values.shape != (len(self.targets),):
            raise ValueError(f'Expected ({len(self.targets)},) output commands')
        if not np.all(np.isfinite(values)):
            raise ValueError('Feedback output commands must be finite')
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError('Feedback command exceeds contribution limits')
        return values

    def apply(self, values):
        """Apply the command and restore the write set if propagation fails."""
        if self._read_only:
            raise RuntimeError('Cannot write an externally owned nominal setpoint')
        values = self._validate_values(values)
        checkpoint = self._checkpoint()
        try:
            for name, value in zip(self.targets, values):
                self._vars[name] = float(value)
            self._dependent_state()
        except BaseException as original:
            try:
                self._restore_checkpoint(checkpoint)
            except BaseException as rollback:
                error = RuntimeError('Xdeps output rollback failed')
                error.add_note(str(rollback))
                raise error from original
            raise
        return values

    def _checkpoint(self):
        """Capture the exact write set within the stopped tracking boundary."""
        self._check_ownership()
        knobs = tuple((ref, ref._get_value()) for ref in self._sources)
        self._validate_values([value for _, value in knobs])
        variables, fields = self._dependent_state()
        return knobs + variables, fields

    @staticmethod
    def _restore_checkpoint(checkpoint):
        # Restore the certified destinations without re-running expressions.
        variables, fields = checkpoint
        for ref, value in variables:
            ref._set_value(value)
        for element, field, index, value, strength in fields:
            xobject = element._xobject
            if strength and xobject._tpsa_enabled:
                element.disable_tpsa()
            if index is None:
                setattr(xobject, field, value)
            else:
                getattr(xobject, field)[index] = value
