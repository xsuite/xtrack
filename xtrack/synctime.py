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
    }

    iscollective = True
    # Use our Python track() below instead of generating C tracking kernels.
    allow_track = False
    allow_rot_and_shift = False

    def __init__(self, circumference, id, frame_relative_length=None,
                 at_start=False, at_end=False, **kwargs):

        if frame_relative_length is None:
            frame_relative_length = DEFAULT_FRAME_RELATIVE_LENGTH
        assert id > COAST_STATE_RANGE_START
        
        super().__init__(
            circumference=circumference,
            id=id,
            frame_relative_length=frame_relative_length,
            at_start=int(at_start),
            at_end=int(at_end),
            **kwargs,
        )

    def track(self, particles, *, time_window=None):
        if time_window is None:
            raise ValueError('SyncTime requires time-window bounds from the tracker')
        lower, upper = time_window
        particles.state[particles.state == -self.id] = 1
        particles.reorganize()
        active = particles.state > 0
        arrival = (particles.time_s
                   + (particles.s - particles.zeta) / (particles.beta0 * clight))
        # Allow roundoff at a shared boundary (e.g. s=C and next frame s=0).
        tol = 32 * np.finfo(float).eps * max(
            abs(lower), abs(upper), particles.t_sim)
        if (active & (arrival < lower - tol)).any():
            raise ValueError('Some particles move faster than the time window')
        particles.state[active & (arrival >= upper)] = -self.id
        if self.at_end:
            # The tracker subsequently resets s and increments at_turn only
            # for particles that have actually finished a revolution.
            active = particles.state > 0
            particles.zeta[active] -= self.circumference


def install_sync_time_at_collective_elements(
        line, frame_relative_length=None, with_progress=True, *,
        at_element_names=()):
    """Install time-window boundaries at the ends of the line and at elements.

    Synchronize before collective elements, RF cavities and any specified
    ``at_element_names``. The common clock is ``Particles.time_s``; ``num_turns``
    counts frames, while each particle's ``at_turn`` counts revolutions.
    Enable ``line.enable_time_dependent_vars`` to drive ramp/RF expressions
    with this clock.

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
    selected |= np.isin(ltab.element_type, ['Cavity', 'CrabCavity'])
    tab_collective = ltab.rows[selected]

    env = line.env

    env.elements["synctime_start"] = SyncTime(circumference=circumference,
        frame_relative_length=frame_relative_length,
        id=COAST_STATE_RANGE_START + len(tab_collective)+1,
        at_start=True,
    )

    for ii, nn in enumerate(tab_collective.name):
        name = f"synctime_{ii}"
        env.elements[name] = SyncTime(
            circumference=circumference,
            frame_relative_length=frame_relative_length,
            id=COAST_STATE_RANGE_START + ii + 1,
        )

    env.elements["synctime_end"] = SyncTime(circumference=circumference,
        frame_relative_length=frame_relative_length,
        id=COAST_STATE_RANGE_START + len(tab_collective)+2,
        at_end=True,
    )

    # Preserve ordering, including multiple thin elements at s=0/C.
    names = ['synctime_start']
    ii = 0
    for nn, use_sync in zip(line.element_names, selected[:-1]):
        if use_sync:
            names.append(f'synctime_{ii}')
            ii += 1
        names.append(nn)
    names.append('synctime_end')
    line.element_names = names


def prepare_particles_for_sync_time(particles, line):
    """Kept for compatibility; SyncTime now handles initial waiting automatically."""
    pass


class _SyncTimeClock:
    """Frame geometry/controller; persistent clock state lives on Particles."""

    def __init__(self, line):
        self.line = line
        table = line.get_table()
        self.elements = [(ee, ss) for ee, ss in zip(line._elements, table.s)
                         if isinstance(ee, SyncTime)]
        if (not self.elements
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
        if particles.lost_particles_are_hidden:
            raise ValueError('SyncTime needs access to waiting particles; unhide them first')

        time = particles.time_s
        program = self.line.energy_program
        if program is None or not self.line.enable_time_dependent_vars:
            beta0 = particles._xobject.beta0[0]
            period = self.circumference / (beta0 * clight)
            def time_at_offset(offset):
                return time + offset * period
        else:
            turn = program.get_turn_at_t_s(time)
            def time_at_offset(offset):
                return program.get_t_s_at_turn(turn + offset)

        # At position s, frame edges are separated by fraction reference turns.
        # Using the SAME time map for both edges ensures exact adjacency in
        # successive frames and across s=C -> s=0 during an accelerating ramp.
        boundaries = [(ee,
                       time_at_offset(self.fraction * (ss / self.circumference - .5)),
                       time_at_offset(self.fraction * (ss / self.circumference + .5)))
                      for ee, ss in self.elements]
        particles.t_sim = time_at_offset(self.fraction) - time
        self.windows = {ee.id: (lower, upper)
                        for ee, lower, upper in boundaries}

    def shift_coordinates(self, particles):
        mask = self.surviving_mask(particles)
        particles.zeta[mask] += particles.beta0[mask] * clight * particles.t_sim
