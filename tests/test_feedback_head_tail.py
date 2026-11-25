# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from scipy.constants import c

import xtrack as xt
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)


class DipoleWake:
    iscollective = True

    def __init__(self, slope):
        self.slope = slope

    def track(self, particles):
        head = particles.zeta > 0
        particles.py[~head] += self.slope * particles.y[head][0]


@pytest.mark.parametrize('active', [False, True], ids=['open_loop', 'feedback'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_slice_feedback_follows_an_independent_head_tail_map(
    test_context, sandbox_cwd, active,
):
    """Match signed pickup coordinates across synchrotron slice exchanges."""
    period, turns = 40, 240
    beta_y, beta_s = 10.0, 120.0
    zeta_max, offset = 0.10, 1e-7
    wake, gain = 0.25, 0.15   # Slopes in normalized vertical phase space.
    k_rf = np.pi / (2 * zeta_max)
    reference = xt.Particles(_context=test_context, p0c=7e12)
    env = xt.Environment(particle_ref=reference)
    frequency = k_rf * float(reference.beta0[0]) * c / (2 * np.pi)
    env['common_rad'] = env['odd_rad'] = 0.0
    env.new('common', xt.Multipole, ksl=['common_rad'])
    env.new(
        'odd', xt.RFMultipole, frequency=frequency, ksl=['odd_rad'],
        phase_s=[np.pi / 2],
    )
    env.elements['wake'] = DipoleWake(wake / beta_y)
    env.elements['pickup'] = xt.BeamStatsMonitor(
        _context=test_context,
        stop_at_turn=turns, num_slices=2, zeta_range=(-0.11, 0.11),
        stats=['mean_y', 'mean_py', 'mean_zeta'],
    )
    env.elements['ring'] = xt.LineSegmentMap(
        length=26000.0, qy=0.25, bety=beta_y, qs=1 / period, bets=beta_s,
    )

    fb = xt.feedback.Feedback(name='damper', update_turns=1, active=active)
    fb.inputs.new(name='y', monitor='pickup', quantity='mean_y', bunch_slot=0)
    fb.inputs.new(
        name='zeta', monitor='pickup', quantity='mean_zeta', bunch_slot=0,
    )
    fb.outputs.new(name='kick', target=['common_rad', 'odd_rad'])
    fb.parameters.new(name='gain', value=gain / beta_y)
    fb.parameters.new(name='k', value=k_rf)
    fb.program = '''
outputs.kick = parameters.gain * stack((
    mean(acquired.y), mean(acquired.y * sin(parameters.k * acquired.zeta))))
'''
    env.elements['damper'] = fb
    line = env.new_line(
        components=['pickup', 'wake', 'common', 'odd', 'ring', 'damper'],
    )
    phase = 2 * np.pi * 0.25 / period
    side = np.array([1.0, -1.0])
    particles = xt.Particles(
        _context=test_context, p0c=7e12,
        y=side * offset, zeta=side * zeta_max * np.cos(phase),
        pzeta=side * zeta_max * np.sin(phase) / beta_s,
    )
    line.build_tracker(_context=test_context)
    line.feedback.enable()
    line.track(particles, num_turns=turns)

    # Iterate (y, beta_y*py) at Qy=1/4. The prior pickup sets the kick
    # on this traversal. Neglect the second-order RF change in zeta.
    y, v = side * offset, np.zeros(2)
    command = np.zeros(2)
    expected = np.empty((turns, 2, 2))
    for turn in range(turns):
        zeta = side * zeta_max * np.cos(2 * np.pi * (turn + 0.25) / period)
        profile = np.sin(k_rf * zeta)
        # The pickup orders the slices by zeta on every turn.
        order = np.argsort(zeta)
        expected[turn] = np.column_stack((y[order], v[order]))
        head = int(zeta[0] < 0)
        v[1 - head] += wake * y[head]
        y, v, command = (
            v + command[0] + command[1] * profile, -y,
            active * gain * np.array([y.mean(), (y * profile).mean()]),
        )

    bpm = line.get('pickup')
    measured = np.stack(
        (bpm.get('mean_y', slot=0), beta_y * bpm.get('mean_py', slot=0)),
        axis=-1,
    )
    np.testing.assert_allclose(measured, expected, rtol=2e-5, atol=1e-12)
    amplitude = np.linalg.norm(measured, axis=(1, 2))
    if active:
        assert amplitude[-period] < 0.03 * amplitude[0]
    else:
        assert amplitude[-period] > 10 * amplitude[0]
