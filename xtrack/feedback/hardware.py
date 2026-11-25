# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""A single physical response owner after nominal and feedback summation."""

from dataclasses import dataclass

import numpy as np

from .output import KnobOutput, _host_array, _limits
from .timing import ActuatorCandidate, FirstOrderActuator


@dataclass(frozen=True)
class _PhysicalCandidate:
    """Previewed response and exact components of the held request."""

    response: ActuatorCandidate
    contribution_target: np.ndarray
    nominal_target: np.ndarray

    @property
    def applied(self):
        return self.response.applied

    @property
    def target(self):
        return self.response.target


class _PhysicalResponse:
    """Track a total field together with its sampled request components."""

    def __init__(self, hardware, *, xp, initial_time_seconds):
        self.hardware = hardware
        self.xp = xp
        installed = hardware.snapshot()
        self._contribution_target = np.zeros_like(installed)
        self._nominal_target = installed
        self._response = FirstOrderActuator(
            installed,
            xp=xp,
            initial_time_seconds=initial_time_seconds,
            response_time_constant_seconds=hardware.response_time_constant_seconds,
            maximum_slew_per_second=hardware.maximum_slew_per_second,
            slew_mode=hardware.slew_mode,
        )

    @property
    def applied(self):
        return self._response.applied

    @property
    def target(self):
        return self._response.target

    @property
    def contribution_target(self):
        return self.xp.asarray(self._contribution_target).copy()

    @property
    def nominal_target(self):
        return self.xp.asarray(self._nominal_target).copy()

    @property
    def time_seconds(self):
        return self._response.time_seconds

    def preview(self, time_seconds, *, target=None):
        # Read the current nominal setpoint even if no controller is due.
        contribution, nominal, total = self.hardware._request(
            self._contribution_target if target is None else target)
        response = self._response.preview(time_seconds, target=total)
        return _PhysicalCandidate(response, contribution, nominal)

    def commit(self, candidate):
        contribution = candidate.contribution_target.copy()
        nominal = candidate.nominal_target.copy()
        self._response.commit(candidate.response)
        self._contribution_target = contribution
        self._nominal_target = nominal


class PhysicalActuator:
    """Sum nominal and feedback requests before one physical response.

    Parameters
    ----------
    owner : Environment or Line, optional
        Owner of the knobs. Omit to configure before binding.
    target : str or sequence of str
        Independent applied-field knobs already wired to elements.
    baseline : str or sequence of str, optional
        Read-only nominal setpoints, one per channel. These can use Xdeps
        expressions, including a machine-time programme. The nominal defaults
        to zero. Setpoints may drive fields outside the system's controlled
        write set.
    lower, upper : float or array, optional
        Bounds in channel units on the applied field and on the sum of the
        nominal and contribution. The total is checked when a contribution
        is requested and when it is delivered.
    response_time_constant_seconds : float or array, optional
        First-order time constant in seconds, scalar or per channel.
    maximum_slew_per_second : float or array, optional
        Slew rate in channel units per second. Each field increment is
        bounded by the rate times the elapsed interval.
    slew_mode : {'independent', 'coordinated'}, optional
        Limit each channel separately (default) or scale the whole increment.

    Notes
    -----
    Hold the installed field over the first interval. At each boundary,
    advance toward the previously held total, then latch the new request
    together with the nominal setpoint held during the completed traversal.
    Zero-time-constant channels with unrestricted slew can apply the new
    request at that boundary. A nominal setpoint sampled at turn entrance
    therefore first affects transport on the following turn. A field wired
    directly to ``t_turn_s`` changes at the same entrance. The response
    continues during a pause.
    """

    def __init__(
        self,
        owner=None,
        *,
        target,
        baseline=None,
        lower=None,
        upper=None,
        response_time_constant_seconds=0.0,
        maximum_slew_per_second=None,
        slew_mode='independent',
    ):
        self._owner = owner
        self._targets = (target,) if isinstance(target, str) else tuple(target)
        if (
            not self.targets
            or any(not isinstance(name, str) or not name for name in self.targets)
            or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError('Physical actuator needs distinct named target knobs')
        self._lower, self._upper = _limits(lower, upper, len(self.targets))
        self._output = (
            None
            if owner is None
            else KnobOutput(owner, self.targets, lower=self._lower, upper=self._upper)
        )
        self._baseline = (
            ()
            if baseline is None
            else (baseline,)
            if isinstance(baseline, str)
            else tuple(baseline)
        )
        if self.baseline and (
            len(self.baseline) != len(self.targets)
            or any(not isinstance(name, str) or not name for name in self.baseline)
            or len(set(self.baseline)) != len(self.baseline)
            or set(self.baseline) & set(self.targets)
        ):
            raise ValueError('Baseline needs one distinct knob per actuator channel')
        self._baseline_output = (
            KnobOutput(owner, self.baseline, _read_only=True)
            if self.baseline and owner is not None
            else None
        )
        self._response_time_constant_seconds = np.asarray(
            response_time_constant_seconds
        ).copy()
        self._maximum_slew_per_second = (
            None
            if maximum_slew_per_second is None
            else np.asarray(maximum_slew_per_second).copy()
        )
        self._slew_mode = slew_mode
        # Check the response settings against the configured channel count.
        FirstOrderActuator(
            np.zeros(len(self.targets)),
            response_time_constant_seconds=self.response_time_constant_seconds,
            maximum_slew_per_second=self.maximum_slew_per_second,
            slew_mode=slew_mode,
        )
        if owner is not None:
            self._request(np.zeros(len(self.targets)))

    @property
    def owner(self):
        return self._owner

    @property
    def targets(self):
        return self._targets

    @property
    def baseline(self):
        return self._baseline

    @property
    def response_time_constant_seconds(self):
        return self._response_time_constant_seconds.copy()

    @property
    def maximum_slew_per_second(self):
        value = self._maximum_slew_per_second
        return None if value is None else value.copy()

    @property
    def slew_mode(self):
        return self._slew_mode

    @property
    def lower(self):
        return self._lower.copy()

    @property
    def upper(self):
        return self._upper.copy()

    def _bound_output(self):
        if self._output is None:
            raise RuntimeError('Physical actuator must be bound to a Line/Environment')
        return self._output

    @property
    def ownership_keys(self):
        return self._bound_output().ownership_keys

    @property
    def dependent_field_keys(self):
        return self._bound_output().dependent_field_keys

    @property
    def _dependent_variable_keys(self):
        return self._bound_output()._dependent_variable_keys

    @property
    def external_setpoint_keys(self):
        """Identify nominal knobs and all physical fields they can write."""
        baseline = self._baseline_output
        if baseline is None:
            return ()
        return baseline.ownership_keys + baseline.dependent_field_keys

    def _request(self, contribution):
        """Return the individual requests and their validated total on host."""
        output = self._bound_output()
        contribution = _host_array(contribution)
        if contribution.shape == () and len(self.targets) == 1:
            contribution = contribution.reshape(1)
        if contribution.shape != (len(self.targets),):
            raise ValueError('Expected one feedback contribution per actuator channel')
        nominal = np.zeros(len(self.targets))
        fields = output.dependent_field_keys
        if self._baseline_output is not None:
            if set(self._baseline_output.dependent_field_keys).intersection(fields):
                raise ValueError(
                    'Nominal setpoints must not bypass the physical response'
                )
            if set(self._baseline_output.ownership_keys).intersection(
                output._dependent_variable_keys,
            ):
                raise ValueError('Nominal setpoints require external ownership')
            nominal = self._baseline_output._read_values()
        total = output._validate_values(nominal + contribution)
        return contribution, nominal, total

    def validate(self, contribution):
        """Validate a feedback request against the present total hardware limit."""
        contribution, _, _ = self._request(contribution)
        return contribution

    def validate_applied(self, total):
        """Validate an already summed physical value without adding nominal twice."""
        return self._bound_output().validate(total)

    def snapshot(self):
        return self._bound_output().snapshot()

    def apply(self, total):
        return self._bound_output().apply(total)

    def _checkpoint(self):
        return self._bound_output()._checkpoint()

    def _restore_checkpoint(self, checkpoint):
        self._bound_output()._restore_checkpoint(checkpoint)

    def make_actuator(self, *, xp, initial_time_seconds):
        """Create the response state advanced by a feedback system."""
        return _PhysicalResponse(self, xp=xp, initial_time_seconds=initial_time_seconds)

    def to_dict(self):
        """Serialize configuration only; numerical runtime state is separate."""
        return dict(
            target=list(self.targets),
            baseline=list(self.baseline) or None,
            lower=self.lower.tolist(),
            upper=self.upper.tolist(),
            response_time_constant_seconds=(
                self.response_time_constant_seconds.tolist()
            ),
            maximum_slew_per_second=(
                None
                if self.maximum_slew_per_second is None
                else self.maximum_slew_per_second.tolist()
            ),
            slew_mode=self.slew_mode,
        )

    def copy_for(self, owner):
        """Bind a copy of the settings to existing knobs in a new owner."""
        return type(self)(owner, **self.to_dict())
