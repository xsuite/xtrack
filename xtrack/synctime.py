import numpy as np
from scipy.constants import c as clight
import xobjects as xo

from .base_element import BeamElement

COAST_STATE_RANGE_START= 1000000
DEFAULT_FRAME_RELATIVE_LENGTH = 0.9

class SyncTime(BeamElement):

    _xofields = {
        "id": xo.Int64,
        "frame_relative_length": xo.Float64,
        "circumference": xo.Float64,
        "at_start": xo.Int64,
        "at_end": xo.Int64,
        "frame_clock": xo.Int64,
        "_t_min": xo.Float64,
        "_t_max": xo.Float64,
    }

    iscollective = True
    # Use our Python track() below instead of generating C tracking kernels.
    allow_track = False
    allow_rot_and_shift = False

    def __init__(self, circumference, id, frame_relative_length=None,
                 at_start=False, at_end=False, frame_clock=False, **kwargs):

        if frame_relative_length is None:
            frame_relative_length = DEFAULT_FRAME_RELATIVE_LENGTH
        assert id > COAST_STATE_RANGE_START
        
        super().__init__(
            circumference=circumference,
            id=id,
            frame_relative_length=frame_relative_length,
            at_start=int(at_start),
            at_end=int(at_end),
            frame_clock=int(frame_clock),
            **kwargs,
        )

    def track(self, particles):

        if isinstance(particles._context, xo.ContextPyopencl):
            raise ValueError('SyncTime does not work with ContextPyopencl')

        if self.frame_clock:
            self._track_with_frame_clock(particles)
            return

        beta0 = particles._xobject.beta0[0]
        beta1 = beta0 / self.frame_relative_length
        beta0_beta1 = beta0 / beta1

        mask_alive = particles.state > 0

        zeta_min = -self.circumference/ 2 * beta0_beta1 + particles.s * (
                   1 - beta0_beta1)

        if (self.at_start and particles.at_turn[0] == 0
                and not (particles.state == -COAST_STATE_RANGE_START).any()): # done by the user
            mask_stop = mask_alive * (particles.zeta < zeta_min)
            particles.state[mask_stop] = -COAST_STATE_RANGE_START
            particles.zeta[mask_stop] += self.circumference * beta0 / beta1

        # Resume particles previously stopped
        particles.state[particles.state==-self.id] = 1
        particles.reorganize()
        mask_alive = particles.state > 0

        # Identify particles that need to be stopped
        zeta_min = -self.circumference/ 2 * beta0_beta1 + particles.s * (1 - beta0_beta1)
        mask_stop = mask_alive & (particles.zeta < zeta_min)

        # Check if some particles are too fast
        mask_too_fast = mask_alive & (
            particles.zeta > zeta_min + self.circumference * beta0_beta1)
        if mask_too_fast.any():
            raise ValueError('Some particles move faster than the time window')

        # For debugging (expected to be triggered)
        # mask_out_of_circumference = mask_alive & (
        #       (particles.zeta > self.circumference / 2)
        #     | (particles.zeta < -self.circumference / 2))
        # if mask_out_of_circumference.any():
        #     raise ValueError('Some particles are out of the circumference')

        # Update zeta for particles that are stopped
        particles.zeta[mask_stop] += beta0_beta1 * self.circumference

        # Stop particles
        particles.state[mask_stop] = -self.id

        if self.at_end:
            mask_alive = particles.state > 0
            particles.zeta[mask_alive] -= (
                self.circumference * (1 - beta0_beta1))

        if self.at_end and particles.at_turn[0] == 0:
            particles.state[particles.state==-COAST_STATE_RANGE_START] = 1

    def _track_with_frame_clock(self, particles):
        # Frame boundaries are set by the tracker. A stopped particle keeps its
        # arrival time; the clock and every surviving particle's coordinates
        # advance together at the end of the frame.
        if particles.at_frame < 0:
            raise ValueError('Frame-clock SyncTime needs a configured tracker')
        particles.state[particles.state == -self.id] = 1
        particles.reorganize()
        active = particles.state > 0
        arrival = (particles.t_frame
                   + (particles.s - particles.zeta) / (particles.beta0 * clight))
        # Allow roundoff at a shared boundary (e.g. s=C and next frame s=0).
        tol = 32 * np.finfo(float).eps * max(
            abs(self._t_min), abs(self._t_max), particles.t_sim)
        if (active & (arrival < self._t_min - tol)).any():
            raise ValueError('Some particles move faster than the time window')
        particles.state[active & (arrival >= self._t_max)] = -self.id
        if self.at_end:
            # The tracker subsequently resets s and increments at_turn only
            # for particles that have actually finished a revolution.
            active = particles.state > 0
            particles.zeta[active] -= self.circumference


def install_sync_time_at_collective_elements(
        line, frame_relative_length=None, with_progress=True, *,
        frame_clock=False, at_element_names=()):
    """Install time-window boundaries at the ends of the line and at elements.

    With ``frame_clock=True``, also synchronize before RF cavities and any
    ``at_element_names``. The common clock is stored on Particles as ``t_frame``
    and ``at_frame``; ``num_turns`` then counts frames, while each particle's
    ``at_turn`` counts revolutions. Enable ``line.enable_time_dependent_vars``
    to drive ramp/RF expressions with this clock.

    ``frame_relative_length`` is the fraction of a reference revolution per
    frame (default 0.9). It must make the window faster than all particles.
    During an EnergyProgram, boundaries follow its reference-turn/time map,
    keeping adjacent windows contiguous even as their durations change.
    Frame-clock tracking currently requires complete frames (no partial-line
    tracking). Install after constructing/slicing the lattice.
    """

    line._frozen_check()
    if any(isinstance(ee, SyncTime) for ee in line._elements):
        raise ValueError('SyncTime elements are already installed')
    at_element_names = set(at_element_names)
    missing = at_element_names - set(line.element_names)
    if missing:
        raise ValueError(f'Unknown synchronization elements: {sorted(missing)}')

    circumference = line.get_length()

    ltab = line.get_table()
    selected = ltab.iscollective | np.isin(ltab.name, list(at_element_names))
    if frame_clock:
        selected |= np.isin(ltab.element_type, ['Cavity', 'CrabCavity'])
    tab_collective = ltab.rows[selected]

    env = line.env
    places = []

    env.elements["synctime_start"] = SyncTime(circumference=circumference,
        frame_relative_length=frame_relative_length,
        id=COAST_STATE_RANGE_START + len(tab_collective)+1,
        at_start=True,
        frame_clock=frame_clock,
    )
    places.append(env.place(name="synctime_start", at=0))

    for ii, nn in enumerate(tab_collective.name):
        name = f"synctime_{ii}"
        env.elements[name] = SyncTime(
            circumference=circumference,
            frame_relative_length= frame_relative_length,
            id=COAST_STATE_RANGE_START + ii + 1,
            frame_clock=frame_clock,
        )
        places.append(env.place(name=name, at=nn))

    env.elements["synctime_end"] = SyncTime(circumference=circumference,
        frame_relative_length=frame_relative_length,
        id=COAST_STATE_RANGE_START + len(tab_collective)+2,
        at_end=True,
        frame_clock=frame_clock,
    )
    places.append(env.place(name="synctime_end", at=circumference))

    if frame_clock:
        # Preserve exact ordering, including multiple thin elements at s=0/C.
        # Each occurrence gets its own synchronization point.
        names = ['synctime_start']
        ii = 0
        for nn, use_sync in zip(line.element_names, selected[:-1]):
            if use_sync:
                names.append(f'synctime_{ii}')
                ii += 1
            names.append(nn)
        names.append('synctime_end')
        line.element_names = names
    else:
        line.insert(places, with_progress=with_progress)


def prepare_particles_for_sync_time(particles, line):
    synctime_start = line['synctime_start']
    if synctime_start.frame_clock:
        # The first SyncTime handles initial particles outside the window.
        # No early coordinate shift is needed in the common-clock mode.
        return
    beta0 = particles._xobject.beta0[0]
    beta1 = beta0 / synctime_start.frame_relative_length
    beta0_beta1 = beta0 / beta1
    zeta_min = -synctime_start.circumference/ 2 * beta0_beta1 + particles.s * (
                1 - beta0_beta1)
    mask_alive = particles.state > 0
    mask_stop = mask_alive * (particles.zeta < zeta_min)
    particles.state[mask_stop] = -COAST_STATE_RANGE_START
    particles.zeta[mask_stop] += synctime_start.circumference * beta0 / beta1


class _SyncTimeClock:
    """Frame geometry/controller; persistent clock state lives on Particles."""

    def __init__(self, line):
        self.line = line
        table = line.get_table()
        self.elements = [(ee, ss) for ee, ss in zip(line._elements, table.s)
                         if isinstance(ee, SyncTime)]
        if (not self.elements or not all(ee.frame_clock for ee, _ in self.elements)
                or not isinstance(line._elements[0], SyncTime)
                or not line._elements[0].at_start
                or not isinstance(line._elements[-1], SyncTime)
                or not line._elements[-1].at_end):
            raise ValueError('Frame-clock SyncTime must bracket the complete line')
        self.circumference = line.get_length()
        self.fraction = self.elements[0][0].frame_relative_length
        if not 0 < self.fraction <= 1:
            raise ValueError('frame_relative_length must be in (0, 1]')
        self.ids = [ee.id for ee, _ in self.elements]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('SyncTime ids must be unique')
        if any(ee.frame_relative_length != self.fraction
               or ee.circumference != self.circumference
               for ee, _ in self.elements):
            raise ValueError('SyncTime elements must use the same frame geometry')

    def surviving_mask(self, particles):
        mask = particles.state > 0
        for id_ in self.ids:
            mask |= particles.state == -id_
        return mask

    def prepare_frame(self, particles):
        if isinstance(particles._context, xo.ContextPyopencl):
            raise ValueError('SyncTime does not work with ContextPyopencl')
        if particles.at_frame < 0:
            if (particles.at_turn[particles.state > 0] != 0).any():
                raise ValueError('Initialize the frame clock before tracking turns')
            particles.at_frame = 0

        if particles.lost_particles_are_hidden:
            raise ValueError('SyncTime needs access to waiting particles; unhide them first')

        time = particles.t_frame
        program = self.line.energy_program
        if program is None:
            beta0 = particles._xobject.beta0[0]
            period = self.circumference / (beta0 * clight)
            def time_at_offset(offset):
                return time + offset * period
        else:
            if not self.line.enable_time_dependent_vars:
                raise ValueError('Enable time-dependent variables for the ramp')
            interp = program.t_at_turn_interpolator
            turns, times = np.asarray(interp.x), np.asarray(interp.y)
            if time < times[0] or time > times[-1]:
                raise ValueError('Frame clock outside EnergyProgram range')
            turn = np.interp(time, times, turns)
            def time_at_offset(offset):
                target = turn + offset
                if target > turns[-1]:
                    raise ValueError('SyncTime window outside EnergyProgram range')
                if target < 0:
                    # Extend the injection period for the leading half-window.
                    return times[0] + target * (times[1] - times[0]) / (turns[1] - turns[0])
                return np.interp(target, turns, times)

        # At position s, frame edges are separated by fraction reference turns.
        # Using the SAME time map for both edges ensures exact adjacency in
        # successive frames and across s=C -> s=0 during an accelerating ramp.
        boundaries = [(ee,
                       time_at_offset(self.fraction * (ss / self.circumference - .5)),
                       time_at_offset(self.fraction * (ss / self.circumference + .5)))
                      for ee, ss in self.elements]
        particles.t_sim = time_at_offset(self.fraction) - time
        for ee, lower, upper in boundaries:
            ee._t_min = lower
            ee._t_max = upper

    def advance_frame(self, particles):
        mask = self.surviving_mask(particles)
        particles.zeta[mask] += particles.beta0[mask] * clight * particles.t_sim
        particles.t_frame += particles.t_sim
        particles.at_frame += 1
