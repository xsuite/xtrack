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


@pytest.mark.parametrize('active', [False, True], ids=['off', 'feedback'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_rf_feedback_cancels_energy_kick_on_next_passage(
    test_context, sandbox_cwd, active,
):
    """Cancel the arrival phase from the completed zeta measurement."""
    momentum, voltage, frequency = 45.6e9, 10e6, 400e6
    zeta = 1.5e-3
    reference = xt.Particles(
        _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=-1, p0c=momentum,
    )
    wave_number = 2 * np.pi * frequency / (float(reference.beta0[0]) * c)
    env = xt.Environment(particle_ref=reference)
    env['rf_phase'] = 0.0
    env.new(
        'rf', xt.Cavity, voltage=voltage, frequency=frequency, phase='rf_phase',
    )
    # Keep the arrival phase fixed between the two cavity passages.
    env.elements['transport'] = xt.LineSegmentMap(
        length=1000.0, longitudinal_mode='frozen',
    )
    env.elements['pickup'] = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=2, stats=['mean_zeta'],
    )
    env.elements['witness'] = xt.ParticlesMonitor(
        _context=test_context, start_at_turn=0, stop_at_turn=2, num_particles=1,
    )
    fb = xt.feedback.Feedback(
        name='rf_trim', active=active,
        update_turns=10, first_calculation_turn=1,
    )
    fb.inputs.new(name='zeta', monitor='pickup', quantity='mean_zeta')
    fb.outputs.new(name='phase', target='rf_phase')
    fb.parameters.new(name='wave_number', value=wave_number)
    fb.program = 'outputs.phase = parameters.wave_number * acquired.zeta'
    env.elements['feedback'] = fb
    line = env.new_line(
        components=['rf', 'witness', 'transport', 'pickup', 'feedback'],
    )
    particles = reference.copy()
    particles.zeta[:] = zeta

    line.build_tracker(_context=test_context)
    line.feedback.enable()
    line.track(particles, num_turns=2)

    # Positive zeta arrives early. Xtrack uses the absolute reference charge.
    first_ptau = voltage * np.sin(-wave_number * zeta) / momentum
    second_ptau = first_ptau if active else 2 * first_ptau
    np.testing.assert_allclose(
        line.get('witness').ptau[0], [first_ptau, second_ptau],
        atol=1e-14, rtol=0,
    )
    expected_phase = wave_number * zeta if active else 0.0
    assert line['rf_phase'] == pytest.approx(expected_phase, abs=1e-14)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_chicane_field_trim_and_momentum_have_opposite_path_signs(
    test_context, sandbox_cwd,
):
    """Check the path derivative of the closed four-bend chicane."""
    angle, magnet_length = 0.05, 1.0
    field = 2 * np.sin(angle / 2) / magnet_length
    env = xt.Environment()
    env['trim'] = 0.0
    for i, sign in enumerate((1, -1, -1, 1)):
        name = f'bend_{i}'
        env.new(
            name, xt.RBend, length_straight=magnet_length, angle=sign * angle,
            k0=sign * field, rbend_model='straight-body',
        )
        env.ref[name].k0 = sign * field * (1 + env.ref['trim'])
    for name, length in [('outer', 3.0), ('centre', 5.0)]:
        env.new(name, xt.Drift, length=length)
    line = env.new_line(components=[
        'bend_0', 'outer', 'bend_1', 'centre', 'bend_2', 'outer', 'bend_3',
    ])
    line.build_tracker(_context=test_context)
    output = xt.feedback.KnobOutput(line, 'trim')

    def zeta(trim, delta=0):
        output.apply(trim)
        beam = xt.Particles(
            _context=test_context, mass0=xt.ELECTRON_MASS_EV,
            q0=-1, p0c=45.6e9, delta=delta,
        )
        line.track(beam)
        np.testing.assert_allclose(
            [float(beam.x[0]), float(beam.px[0])], 0, atol=1e-12,
        )
        return float(beam.zeta[0])

    trim_step, delta_step = 1e-3, 1e-4
    trim_slope = (zeta(trim_step) - zeta(-trim_step)) / (2 * trim_step)
    momentum_slope = (
        zeta(0, delta_step) - zeta(0, -delta_step)
    ) / (2 * delta_step)
    # A longer path makes zeta more negative. Use the small-angle path excess
    # of four rectangular bends as the independent dimensional scale. At this
    # electron energy the speed contribution to d(zeta)/d(delta) is negligible.
    path_slope = 2 * angle**2 * (3 + 2 * magnet_length / 3)
    assert -trim_slope == pytest.approx(path_slope, rel=1e-3)
    assert momentum_slope == pytest.approx(-trim_slope, rel=5e-7)
