# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Synchronize fixed-period full-turn callbacks using native pipeline holds."""

from collections.abc import Mapping
import math
from types import MappingProxyType

import numpy as np
import xobjects as xo
from scipy.constants import c as clight

from ..beam_elements.reference_energy_change import ReferenceEnergyChange
from ..beam_elements.reference_energy_increase import ReferenceEnergyIncrease
from ..line import Line
from ..particles import Particles
from ..pipeline import PipelineMultiTracker, PipelineStatus
from .runtime import FeedbackSystem
from .timing import _check_turn_clock_resolution


def _same_timestamp(value, reference):
    """Compare host clock readings to within two floating-point spacings."""
    return (
        math.isfinite(value) and math.isfinite(reference)
        and abs(value - reference) <= 2 * math.ulp(max(abs(value), abs(reference)))
    )


class _SharedBoundaryElement:
    iscollective = True
    needs_cpu = False

    def __init__(self, coordinator, branch_name):
        self.coordinator = coordinator
        self.branch_name = branch_name

    def track(self, particles):
        return self.coordinator._visit(self.branch_name, particles)


class SharedFeedbackBoundary:
    """Advance feedback after all branches complete a full turn.

    Install before tracker build, then bind to the existing multitracker and
    use this object's :meth:`track`. Before the feedback step, the optional
    hook receives the completed turn and the tracked bunches::

        before_feedback(completed_turn=n, time_seconds=t,
                        particles_by_branch={name: particles})

    The optional post-update hook receives the records returned by the system::

        after_feedback(completed_turn=n, time_seconds=t, updates=records)

    Each branch supplies a distinct particle collection on the same numerical
    context. The clock uses fixed reference energy and a common revolution
    period. The callback receives borrowed bunches whose native clocks still
    show the start of the traversal. Its timestamp is the bound epoch plus
    the elapsed full turns times the revolution period.

    Treat the borrowed bunches as read-only and apply settings for later
    transport. All branches are checked together before the callback.
    If tracking or the callback fails after transport starts, reconstruct
    the run before continuing. Device checks may synchronise scalar results.
    Copy or serialize the lines before installing the boundary.

    Pass a :class:`FeedbackSystem` to :meth:`bind` to check its native pickup
    records and advance it on the elapsed clock between the two hooks.
    """

    def __init__(
        self, branch_names, before_feedback=None, reference_branch=None,
        *, after_feedback=None,
    ):
        if isinstance(branch_names, str):
            raise TypeError('branch_names must be a sequence, not one string')
        self.branch_names = tuple(branch_names)
        if (
            not self.branch_names
            or any(not isinstance(name, str) or not name for name in self.branch_names)
            or len(set(self.branch_names)) != len(self.branch_names)
        ):
            raise ValueError('Expected distinct nonempty branch names')
        if before_feedback is not None and not callable(before_feedback):
            raise TypeError('before_feedback must be callable')
        if after_feedback is not None and not callable(after_feedback):
            raise TypeError('after_feedback must be callable')
        self.reference_branch = (
            self.branch_names[0] if reference_branch is None else reference_branch
        )
        if self.reference_branch not in self.branch_names:
            raise ValueError('reference_branch must belong to branch_names')
        self._before_feedback = before_feedback
        self._after_feedback = after_feedback
        self.elements = MappingProxyType(
            {name: _SharedBoundaryElement(self, name) for name in self.branch_names}
        )
        self._lines = None
        self._trackers = None
        self._particles = None
        self._multitracker = None
        self._system = None
        self._period = None
        self._epoch_turn = None
        self._epoch_time = None
        self._context = None
        self._branch_periods = None
        self._turn_start_times = None
        self._turn_start_reference_turns = None
        self._lengths = None
        self._reference_energy = None
        self._tracking = False
        self._generation = None
        self._arrivals = set()
        self._released = set()
        self._failure = None
        self.completed_turn = None
        self.callback_count = 0
        self.last_time_seconds = None

    def install(self, lines):
        """Append one zero-length collective boundary per unbuilt line."""
        if self._lines is not None:
            raise RuntimeError('Boundary elements have already been installed')
        if not isinstance(lines, Mapping) or set(lines) != set(self.branch_names):
            raise ValueError('lines must map exactly the configured branch names')
        if len({id(line) for line in lines.values()}) != len(lines):
            raise ValueError('Each branch needs a distinct Line')
        for name, line in lines.items():
            if not isinstance(line, Line):
                raise TypeError(f'{name}: expected a Line')
            if line.mode == 'compose':
                raise ValueError(f'{name}: end compose mode before installing the boundary')
            if line.tracker is not None:
                raise ValueError(f'{name}: install the boundary before tracker build')
            element_name = f'_feedback_boundary_{name}'
            if element_name in line.env.elements:
                raise ValueError(f'{name}: boundary element name already exists')
        for name in self.branch_names:
            lines[name].append(f'_feedback_boundary_{name}', self.elements[name])
        self._lines = dict(lines)
        return self

    def bind(self, multitracker, *, system=None):
        """Validate complete branch membership and a common fixed-energy epoch."""
        if self._lines is None:
            raise RuntimeError('Install the boundaries before binding')
        if self._multitracker is not None:
            raise RuntimeError('Coordinator has already been bound')
        if self._after_feedback is not None and system is None:
            raise ValueError('after_feedback requires a bound FeedbackSystem')
        particles = self._match_branches(multitracker)
        trackers = {}
        epochs = {}
        lengths = {}
        reference_energy = {}
        raw_turns = {}
        for name in self.branch_names:
            line = self._lines[name]
            tracker = self._check_line(name)
            if any(
                isinstance(element, (ReferenceEnergyChange, ReferenceEnergyIncrease))
                for element in tracker._tracker_data_base.elements
            ):
                raise NotImplementedError(
                    f'{name}: this boundary requires fixed reference energy'
                )
            p = particles[name]
            if not isinstance(p._context, (xo.ContextCpu, xo.ContextCupy)):
                raise NotImplementedError('Only CPU/CuPy contexts are supported')
            if p._context is not tracker._context:
                raise ValueError(f'{name}: tracker and particles need the same context')
            self._check_start(p, name)
            beta0 = float(p._xobject.beta0[0])
            if not math.isfinite(beta0) or beta0 <= 0:
                raise ValueError(f'{name}: expected positive finite reference speed')
            reference_energy[name] = (float(p._xobject.p0c[0]), beta0)
            lengths[name] = self._check_geometry(line, tracker, name)
            period = lengths[name] / (beta0 * clight)
            if not np.isfinite(period) or period <= 0:
                raise ValueError(
                    f'{name}: expected a positive finite revolution period'
                )
            time = float(p.time_s)
            if not np.isfinite(time) or time < 0:
                raise ValueError(f'{name}: expected a finite nonnegative epoch')
            raw_turns[name] = self._reference_turns(p, tracker)
            turn = self._integer_turn(raw_turns[name], name)
            _check_turn_clock_resolution(time, period)
            self._require_particles_at(p, name, turn=turn, element=0, s=0.0)
            epochs[name] = (turn, time, period)
            trackers[name] = tracker
        reference = epochs[self.reference_branch]
        reference_context = particles[self.reference_branch]._context
        for name, epoch in epochs.items():
            if particles[name]._context is not reference_context:
                raise NotImplementedError(
                    'Shared feedback requires one numerical context'
                )
            if epoch[0] != reference[0]:
                raise ValueError(f'{name}: incompatible reference-turn epoch')
            if not _same_timestamp(epoch[1], reference[1]):
                raise ValueError(f'{name}: incompatible time epoch')
            # Compare periods on their own scale, independent of absolute time.
            if not math.isclose(
                epoch[2], reference[2], rel_tol=128 * np.finfo(float).eps,
            ):
                raise ValueError(f'{name}: incompatible revolution period')
        if system is not None:
            if not isinstance(system, FeedbackSystem):
                raise TypeError('Expected a FeedbackSystem')
            boundary_lines = self._lines.values()
            if any(line not in boundary_lines
                   for line in system.lines.values()):
                raise ValueError('System uses a line outside the boundary')
            if any(c.context is not reference_context
                   for c in system.controllers):
                raise ValueError('Feedback system must use the branch context')
            system._bind_turn_clock(reference)
        self._system = system
        self._multitracker = multitracker
        self._particles = MappingProxyType(particles)
        self._trackers = trackers
        self._period = reference[2]
        self._epoch_turn = reference[0]
        self._epoch_time = reference[1]
        self._context = reference_context
        self._branch_periods = {name: epoch[2] for name, epoch in epochs.items()}
        self._turn_start_times = {name: epoch[1] for name, epoch in epochs.items()}
        self._turn_start_reference_turns = raw_turns
        self._lengths = lengths
        self._reference_energy = reference_energy
        self.completed_turn = reference[0]
        self.last_time_seconds = reference[1]
        return self

    def track(self, *, num_turns):
        """Track complete turns, including the final boundary transaction."""
        if (
            not isinstance(num_turns, (int, np.integer))
            or isinstance(num_turns, bool)
            or num_turns <= 0
        ):
            raise ValueError('num_turns must be a positive integer')
        self._ensure_bound()
        if self._tracking:
            raise RuntimeError('Feedback boundary tracking is not reentrant')
        self._check_configuration()
        self._check_stopped_epoch()
        _check_turn_clock_resolution(
            self.last_time_seconds + num_turns * self._period, self._period
        )
        if self._system is not None:
            self._system._validate_source_windows(
                self.completed_turn, self.completed_turn + num_turns,
            )
        start = self.callback_count
        self._tracking = True
        try:
            self._multitracker.track(num_turns=int(num_turns))
            if self._generation is not None or self.callback_count - start != num_turns:
                raise RuntimeError('Tracking returned with an incomplete boundary')
            self._check_stopped_epoch()
        except BaseException as error:
            self._failure = error
            raise
        finally:
            self._tracking = False

    def _visit(self, name, particles):
        """Wait at the turn barrier and release after the common callback."""
        self._ensure_bound()
        if not self._tracking:
            raise RuntimeError(
                'Use SharedFeedbackBoundary.track for full-turn tracking'
            )
        if particles is not self._particles[name]:
            raise ValueError(f'{name}: unexpected particle collection')
        raw_turn = float(particles.reference_turns)
        if raw_turn != self._turn_start_reference_turns[name]:
            raise ValueError(f'{name}: reference turns changed within a traversal')
        start_turn = round(raw_turn)
        completed_turn = start_turn + 1
        if self._generation is None:
            if completed_turn != self.completed_turn + 1:
                raise ValueError(f'{name}: skipped or repeated boundary turn')
            self._generation = completed_turn
        if completed_turn > self._generation:
            if completed_turn != self._generation + 1 or name not in self._released:
                raise ValueError(
                    f'{name}: branch advanced outside the boundary protocol'
                )
            # A released branch can reach the next turn before its peers leave.
            return PipelineStatus(
                on_hold=True, info='waiting for preceding boundary release'
            )
        if completed_turn < self._generation or name in self._released:
            raise ValueError(f'{name}: stale or duplicated boundary visit')
        self._arrivals.add(name)
        if (len(self._arrivals) == len(self.branch_names)
                and self.completed_turn < self._generation):
            # Other collective elements can run while a branch waits here.
            for branch_name, bunch in self._particles.items():
                self._check_arrival(branch_name, bunch, start_turn)
            reference_time = (
                self._epoch_time + (self._generation - self._epoch_turn) * self._period
            )
            if self._before_feedback is not None:
                self._before_feedback(
                    completed_turn=self._generation,
                    time_seconds=reference_time,
                    particles_by_branch=self._particles,
                )
            if self._system is not None:
                updates = self._system.step(self._generation, reference_time)
                if self._after_feedback is not None:
                    self._after_feedback(
                        completed_turn=self._generation,
                        time_seconds=reference_time,
                        updates=updates,
                    )
            self.callback_count += 1
            self.completed_turn = self._generation
            self.last_time_seconds = reference_time
        if self.completed_turn < self._generation:
            return PipelineStatus(
                on_hold=True, info='waiting for all feedback branches'
            )
        self._released.add(name)
        # Predict the native addition on each branch to account for its rounding.
        self._turn_start_times[name] += self._branch_periods[name]
        self._turn_start_reference_turns[name] += 1
        if len(self._released) == len(self.branch_names):
            self._generation = None
            self._arrivals.clear()
            self._released.clear()
        return None

    def _check_arrival(self, name, particles, start_turn):
        """Certify a branch's clock and particle position at the callback."""
        self._check_line(name)
        if particles._context is not self._context:
            raise ValueError(f'{name}: tracking context changed at the boundary')
        if float(particles.reference_turns) != self._turn_start_reference_turns[name]:
            raise ValueError(f'{name}: reference turns changed within a traversal')
        if not _same_timestamp(float(particles.time_s), self._turn_start_times[name]):
            raise ValueError(f'{name}: clock changed within a traversal')
        if float(particles.t_sim) != self._branch_periods[name]:
            raise ValueError(f'{name}: reference period changed while tracking')
        self._check_reference_energy(name, particles, 'while tracking')
        self._require_particles_at(
            particles, name, turn=start_turn,
            element=len(self._lines[name]) - 1, s=self._lengths[name],
        )

    def _match_branches(self, multitracker):
        """Match each registered line to a separate Xobjects particle allocation."""
        if not isinstance(multitracker, PipelineMultiTracker):
            raise TypeError('Expected a PipelineMultiTracker')
        by_line = {id(line): name for name, line in self._lines.items()}
        result = {}
        for branch in multitracker.branches:
            name = by_line.get(id(branch.line))
            if name is None or name in result:
                raise ValueError(
                    'Multitracker must contain every configured line exactly once'
                )
            result[name] = branch.particles
        if set(result) != set(self.branch_names):
            raise ValueError('Multitracker is missing configured branches')
        if any(not isinstance(p, Particles) for p in result.values()):
            raise TypeError('Each branch needs an Xtrack Particles collection')
        storage = {(id(p._buffer.buffer), p._offset) for p in result.values()}
        if len(storage) != len(result):
            raise ValueError('Each branch needs distinct particle storage')
        return {name: result[name] for name in self.branch_names}

    def _check_configuration(self):
        particles = self._match_branches(self._multitracker)
        if any(
            particles[name] is not self._particles[name] for name in self.branch_names
        ):
            raise ValueError('Bound particle collections changed')
        for name in self.branch_names:
            self._check_line(name)
            self._check_geometry(self._lines[name], self._trackers[name], name)

    def _check_stopped_epoch(self):
        """Check that every branch is at the origin of the same traversal."""
        for name, particles in self._particles.items():
            if particles._context is not self._context:
                raise ValueError(f'{name}: tracking context changed after binding')
            self._check_start(particles, name)
            self._check_reference_energy(name, particles, 'while stopped')
            raw_turn = self._reference_turns(particles, self._trackers[name])
            if raw_turn != self._turn_start_reference_turns[name]:
                raise ValueError(f'{name}: stopped reference-turn epoch changed')
            if not _same_timestamp(
                float(particles.time_s), self._turn_start_times[name]
            ):
                raise ValueError(f'{name}: stopped time epoch changed')
            self._require_particles_at(
                particles, name, turn=self.completed_turn, element=0, s=0.0
            )

    def _check_reference_energy(self, name, particles, phase):
        p0c, beta0 = self._reference_energy[name]
        actual_p0c = float(particles._xobject.p0c[0])
        actual_beta0 = float(particles._xobject.beta0[0])
        if (actual_p0c, actual_beta0) != (p0c, beta0):
            raise NotImplementedError(
                f'{name}: fixed reference energy changed {phase}'
            )

    def _check_line(self, name):
        line = self._lines[name]
        tracker = line.tracker
        if self._trackers is not None and tracker is not self._trackers[name]:
            raise RuntimeError(
                f'{name}: tracker changed; reconstruct and rebind the coordinator'
            )
        if tracker is None:
            raise ValueError(f'{name}: build the tracker before binding')
        if not tracker.enable_pipeline_hold:
            raise ValueError(f'{name}: shared feedback requires native pipeline holds')
        if line.get(line.element_names[-1]) is not self.elements[name]:
            raise ValueError(f'{name}: boundary must be the final element')
        if getattr(tracker, '_sync_time_controller', None) is not None:
            raise NotImplementedError(f'{name}: shared feedback requires full turns')
        if line.enable_time_dependent_vars or line.energy_program is not None:
            raise NotImplementedError(
                f'{name}: shared feedback requires fixed reference energy'
            )
        if line.skip_end_turn_actions or not line.reset_s_at_end_turn:
            raise ValueError('Full end-turn actions and s reset are required')
        return tracker

    def _ensure_bound(self):
        if self._multitracker is None:
            raise RuntimeError('Bind the coordinator before tracking')
        if self._failure is not None:
            raise RuntimeError(
                'Boundary tracking previously failed; coordinator cannot resume'
            ) from self._failure

    @staticmethod
    def _integer_turn(value, name):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name}: expected an integer full-turn reference epoch')
        turn = round(value)
        if abs(value - turn) > max(1e-9, 2 * math.ulp(value)):
            raise ValueError(f'{name}: expected an integer full-turn reference epoch')
        return turn

    @staticmethod
    def _reference_turns(particles, tracker):
        value = float(particles.reference_turns)
        return float(tracker._get_reference_turns(particles)) if math.isnan(value) else value

    @staticmethod
    def _check_start(particles, name):
        if particles.start_tracking_at_element >= 0:
            raise ValueError(f'{name}: shared feedback requires a turn start at the origin')

    @staticmethod
    def _check_geometry(line, tracker, name):
        """Use the cached native geometry after checking live positions."""
        geometry = tracker._tracker_data_base
        if not np.array_equal(
            line._get_s_elements(mode='downstream'), geometry._line_table.s[1:]
        ):
            raise ValueError(f'{name}: line geometry changed after tracker build')
        return float(geometry.line_length)

    @staticmethod
    def _flag(value):
        return bool(value.item() if hasattr(value, 'item') else value)

    def _require_particles_at(self, particles, name, *, turn, element, s):
        xp = particles._context.nplike_lib
        active = particles.state > 0
        has_active = xp.any(active)
        valid = (
            has_active
            & xp.all(particles.at_turn[active] == turn)
            & xp.all(particles.at_element[active] == element)
            & xp.all(
                xp.abs(particles.s[active] - s)
                <= 128 * np.finfo(float).eps * max(abs(s), 1.0)
            )
        )
        if not self._flag(valid):
            if not self._flag(has_active):
                raise ValueError(f'{name}: no active particles at feedback boundary')
            raise ValueError(
                f'{name}: particles do not share the required full-turn boundary'
            )
