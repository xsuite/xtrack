# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from scipy.constants import c
from scipy.integrate import quad

import xtrack as xt
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts

from xtrack.feedback import (
    FeedbackController, FeedbackSystem, KnobOutput, SharedFeedbackBoundary,
)
from xtrack.feedback.readout import bind_monitor

pytestmark = pytest.mark.usefixtures('sandbox_cwd')

# Native EnergyProgram momentum updates and feedback arrays require
# CPU or CuPy (PyOpenCL does not support the masked px/py update).


def test_elapsed_time_across_energy_program_segments():
    """Recover an interval spanning acceleration, flat top and deceleration."""
    length = 120.
    times = np.array([0., 10e-6, 18e-6, 30e-6])
    momenta = np.array([0.6e9, 1.8e9, 1.8e9, 0.8e9])
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.set_particle_ref('proton', p0c=momenta[0])
    line.energy_program = xt.EnergyProgram(t_s=times, p0c=momenta)

    def turns_per_second(time):
        p0c = np.interp(time, times, momenta)
        return c / length * p0c / np.hypot(p0c, xt.PROTON_MASS_EV)

    start_time, stop_time = 4e-6, 27e-6
    start = quad(turns_per_second, 0., start_time)[0]
    stop = quad(turns_per_second, 0., stop_time, points=times[1:-1])[0]
    elapsed = line.energy_program._elapsed_time_between_turns(start, stop)
    assert elapsed == pytest.approx(stop_time - start_time, rel=2e-15, abs=0.)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_ramped_feedback_delivers_by_time_and_scales_kick(test_context):
    """Track the variable delay and the field-to-kick conversion on the ramp."""
    length, p0c, slope = 120., 0.6e9, 6e13
    turns, delay, gain = 20, 0.58e-6, 0.02
    # For linear p0c(t) at fixed circumference, dE/dturn = (L/c) dp0c/dt.
    delta_energy = length / c * slope
    mass = xt.PROTON_MASS_EV
    energy = np.hypot(p0c, mass) + np.arange(turns + 1) * delta_energy
    momentum = np.sqrt(energy**2 - mass**2)
    time = (momentum - p0c) / slope
    deliveries = np.searchsorted(time, time[1:-1] + delay)
    assert deliveries[0] - 1 == 1
    assert deliveries[-4] - (turns - 4) == 2

    def run(chunks):
        env = xt.Environment(particle_ref=xt.Particles(p0c=p0c))
        env['p0c_injection'] = p0c
        env['p0c_slope'] = slope
        env['c_light'] = c
        env['field_tm'] = 0.
        env.new('aperture', xt.LimitRect, min_x=-1e-3, max_x=1e-3)
        env.new('corrector', xt.Multipole, ksl=[
            'c_light * field_tm / (p0c_injection + p0c_slope * t_turn_s)',
        ])
        env.new('accelerator', xt.Cavity, voltage=delta_energy, phase=np.pi / 2)
        env.elements['arc'] = xt.LineSegmentMap(
            length=length, qy=0.25, bety=12., longitudinal_mode='frozen',
        )
        env.elements['bpm'] = xt.BeamStatsMonitor(
            stop_at_turn=turns, stats=['mean_y'],
        )
        for name in ('before', 'after'):
            env.elements[name] = xt.ParticlesMonitor(
                start_at_turn=0, stop_at_turn=turns, particle_id_range=(1, 2),
            )
        fb = xt.feedback.Feedback(name='orbit', command_delay_seconds=delay)
        fb.inputs.new(name='y', monitor='bpm', quantity='mean_y')
        fb.outputs.new(name='field', target='field_tm')
        fb.parameters.new(name='gain', value=gain)
        fb.program = 'outputs.field = parameters.gain * acquired.y'
        env.elements['feedback'] = fb
        line = env.new_line(components=[
            'aperture', 'bpm', 'before', 'corrector', 'after', 'arc',
            'accelerator', 'feedback',
        ])
        line.energy_program = xt.EnergyProgram(
            t_s=[0., 2e-5], p0c=[p0c, p0c + slope * 2e-5],
        )
        line.enable_time_dependent_vars = True
        line.build_tracker(_context=test_context)
        particles = line.build_particles(x=[2e-3, 0.], y=1e-3)
        line.feedback.enable()
        for chunk in chunks:
            line.track(particles, num_turns=chunk)

        to_host = test_context.nparray_from_context_array
        rows = fb.log
        np.testing.assert_allclose([row['time_seconds'] for row in rows],
                                   time[1:], rtol=0., atol=6e-20)
        np.testing.assert_allclose([row['dt_seconds'] for row in rows],
                                   np.r_[0., np.diff(time[1:])], atol=6e-20)
        field = np.array([to_host(row['applied_output'])[0] for row in rows])
        requested = gain * env.get('bpm').mean_y
        expected = np.zeros(turns + 1)
        for index, delivery in enumerate(deliveries):
            expected[delivery:] = requested[index]
        np.testing.assert_allclose(field, expected[1:], atol=1e-18, rtol=0.)
        after = env.get('after')
        before = env.get('before')
        np.testing.assert_allclose(before.delta[0], 0., atol=3e-14)
        kick = (after.py - before.py)[0]
        np.testing.assert_allclose(kick, c * expected[:-1] / momentum[:-1],
                                   atol=2e-16, rtol=0.)
        np.testing.assert_allclose(particles.time_s, time[-1], atol=6e-20)
        assert np.count_nonzero(to_host(particles.state) > 0) == 1
        return field, kick

    np.testing.assert_array_equal(run((turns,)), run((7, turns - 7)))


def _shared_ramp(context, slopes, first_turn=0):
    """Bind two lines with identical injection clocks and selectable slopes."""
    events = []
    boundary = SharedFeedbackBoundary(
        ('a', 'b'), after_feedback=lambda **event: events.append(event),
    )
    lines, particles = {}, {}
    for name, slope in zip(boundary.branch_names, slopes):
        line = xt.Line(elements={
            'ring': xt.Drift(length=120.),
            'aperture': xt.LimitRect(min_x=-1e-3, max_x=1e-3),
            'bpm': xt.BeamStatsMonitor(
                start_at_turn=first_turn, stop_at_turn=first_turn + 12,
                stats=['mean_y'], _context=context,
            ),
        }, element_names=['aperture', 'ring', 'bpm'])
        line.set_particle_ref('proton', p0c=0.6e9)
        line.energy_program = xt.EnergyProgram(
            t_s=[0., 1e-6, 2e-5], p0c=[0.6e9, 0.6e9, 0.6e9 + slope * 19e-6],
        )
        line.enable_time_dependent_vars = True
        line['trim'] = 0.
        lines[name] = line
        epoch = line.energy_program.get_t_s_at_turn(first_turn)
        if first_turn and name == 'b':
            epoch = np.nextafter(epoch, np.inf)
        particles[name] = xt.Particles(
            _context=context, p0c=line.energy_program.get_p0c_at_t_s(epoch),
            x=[2e-3, 0.], y=1e-3, at_turn=first_turn, time_s=epoch,
        )

    boundary.install(lines)
    for line in lines.values():
        line.build_tracker(_context=context)
    controller = FeedbackController(
        'shared', sources={'y': bind_monitor(lines['b'].get('bpm'), 'mean_y')},
        output=KnobOutput(lines['a'], 'trim'),
        program=lambda inputs: {'outputs': {'command': -inputs['y']}},
        log_capacity=12,
    )
    system = FeedbackSystem([controller], lines=list(lines.values()))
    multitracker = xt.PipelineMultiTracker([
        xt.PipelineBranch(line=lines[name], particles=particles[name])
        for name in ('b', 'a')
    ])
    boundary.bind(multitracker, system=system)
    return boundary, lines, particles, events


@pytest.mark.parametrize('first_turn', [0, 3])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_shared_feedback_uses_common_energy_program_clock(test_context, first_turn):
    """Run a cross-line controller at native ramp times across split calls."""
    boundary, lines, particles, events = _shared_ramp(
        test_context, [6e13, 6e13], first_turn,
    )
    boundary.track(num_turns=3)
    boundary.track(num_turns=7)
    turns = np.arange(1, 11) + first_turn
    times = lines['a'].energy_program.get_t_s_at_turn(turns)
    np.testing.assert_array_equal([event['time_seconds'] for event in events], times)
    np.testing.assert_array_equal([event['completed_turn'] for event in events], turns)
    assert lines['a']['trim'] == pytest.approx(-1e-3)
    for beam in particles.values():
        assert beam.time_s == pytest.approx(times[-1], rel=1e-15, abs=0.)
        state = test_context.nparray_from_context_array(beam.state)
        assert np.count_nonzero(state > 0) == 1


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_shared_feedback_rejects_diverging_ramps_before_transport(test_context):
    """Reject a later mismatch even when the first turn calendars agree."""
    boundary, _, particles, events = _shared_ramp(test_context, [6e13, 3e13])
    with pytest.raises(ValueError, match='incompatible ramp boundary times'):
        boundary.track(num_turns=5)
    assert not events
    assert all(p.time_s == 0. and p.at_turn[0] == 0 for p in particles.values())


@pytest.mark.parametrize('reference', ['fixed', 'programmed'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_shared_feedback_accepts_equivalent_fixed_and_programmed_clocks(
    test_context, reference,
):
    """Use the same calendar after a split despite native clock roundoff."""
    names = ('fixed', 'programmed')
    events = []
    boundary = SharedFeedbackBoundary(
        names, lambda **event: events.append(event['time_seconds']),
        reference_branch=reference,
    )
    lines = {name: xt.Line(elements=[xt.Drift(length=120.)]) for name in names}
    for line in lines.values():
        line.set_particle_ref('proton', p0c=0.6e9)
    programmed = lines['programmed']
    programmed.energy_program = xt.EnergyProgram(t_s=[0., .01], p0c=[0.6e9, 0.6e9])
    programmed.enable_time_dependent_vars = True
    boundary.install(lines)
    for line in lines.values():
        line.build_tracker(_context=test_context)
    branches = [xt.PipelineBranch(line, line.build_particles()) for line in lines.values()]
    boundary.bind(xt.PipelineMultiTracker(branches))

    boundary.track(num_turns=800)
    boundary.track(num_turns=200)

    expected = programmed.energy_program.get_t_s_at_turn(np.arange(1, 1001))
    np.testing.assert_allclose(events, expected, rtol=1e-15, atol=0.)
    assert all(branch.particles.at_turn[0] == 1000 for branch in branches)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_feedback_tracks_up_to_last_energy_program_knot(test_context):
    """Use the admitted integer even when the native inverse rounds up."""
    unit = 2.**-23
    period = 5 * unit  # beta = 3/5
    reference = xt.Particles(p0c=3e9, mass0=4e9)
    env = xt.Environment(particle_ref=reference)
    env['trim'] = 0.
    env.new('ring', xt.Drift, length=3 * c * unit)
    env.elements['bpm'] = xt.BeamStatsMonitor(
        stop_at_turn=5, stats=['mean_y', 'num_particles'],
    )
    fb = xt.feedback.Feedback(name='orbit')
    fb.inputs.new(name='y', monitor='bpm', quantity='mean_y')
    fb.outputs.new(name='kick', target='trim')
    fb.program = 'outputs.kick = acquired.y'
    env.elements['feedback'] = fb
    line = env.new_line(components=['ring', 'bpm', 'feedback'])
    line.energy_program = xt.EnergyProgram(t_s=[0., 3 * period], p0c=[3e9, 3e9])
    line.enable_time_dependent_vars = True
    epoch = np.nextafter(period, np.inf)
    epoch = np.nextafter(epoch, np.inf)
    assert line.energy_program.get_turn_at_t_s(epoch) > 1.
    line.build_tracker(_context=test_context)
    particles = xt.Particles(_context=test_context, p0c=3e9, mass0=4e9,
                             time_s=epoch, at_turn=1, y=1e-3)
    line.feedback.enable()

    with pytest.raises(ValueError, match='outside program range'):
        line.track(particles, num_turns=3)
    assert particles.at_turn[0] == 1
    np.testing.assert_array_equal(env.get('bpm').num_particles, 0.)

    line.track(particles, num_turns=2)
    assert particles.reference_turns == 3.
    assert particles.time_s == 3 * period
    assert [row['completed_turn'] for row in fb.log] == [2, 3]


@pytest.mark.parametrize(('offset', 'first_delivery'), [(0., 2), (100e-12, 3)])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_energy_program_command_deadline_at_large_epoch(
    test_context, offset, first_delivery,
):
    """Preserve delay resolution after an absolute clock with coarse spacing."""
    env = xt.Environment(particle_ref=xt.Particles(p0c=0.6e9))
    period = 120. / (float(env.particle_ref.beta0[0]) * c)
    turn = round(1e5 / period)
    env['field'] = 0.
    env.new('ring', xt.Drift, length=120.)
    env.elements['bpm'] = xt.BeamStatsMonitor(
        start_at_turn=turn, stop_at_turn=turn + 4, stats=['mean_y'],
    )
    fb = xt.feedback.Feedback(
        name='orbit', command_delay_seconds=period + offset,
        update_frequency_hz=1. / period,
    )
    fb.inputs.new(name='y', monitor='bpm', quantity='mean_y')
    fb.outputs.new(name='field', target='field')
    fb.program = 'outputs.field = acquired.y'
    env.elements['feedback'] = fb
    line = env.new_line(components=['ring', 'bpm', 'feedback'])
    line.energy_program = xt.EnergyProgram(t_s=[0., 2e5], p0c=[0.6e9, 0.6e9])
    line.enable_time_dependent_vars = True
    epoch = line.energy_program.get_t_s_at_turn(turn)
    line.build_tracker(_context=test_context)
    particles = xt.Particles(_context=test_context, p0c=0.6e9, y=1e-3,
                             time_s=epoch, at_turn=turn, reference_turns=turn)
    line.feedback.enable()
    line.track(particles, num_turns=4)
    first = next(row for row in fb.log if row['applied_output'][0] != 0.)
    assert first['completed_turn'] == turn + first_delivery
