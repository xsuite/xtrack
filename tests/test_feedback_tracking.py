# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Public feedback activation, causal tracking and configuration ownership."""

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)

pytestmark = pytest.mark.usefixtures('sandbox_cwd')


@pytest.fixture(scope='module')
def context():
    result = xo.ContextCpu()
    result.allow_kernel_compilation = True
    return result


def test_line_without_feedback_uses_native_build_and_pipeline(context, mocker):
    """Use the public lazy build and automatic dispatch for an ordinary line."""
    drift = xt.Drift(length=1.0, _context=context)
    drift.iscollective = True
    line = xt.Line(elements=[drift])
    particles = xt.Particles(p0c=7e12, _context=context)
    particles.init_pipeline('beam')
    build_tracker = mocker.spy(line, 'build_tracker')

    line.track(particles, num_turns=2)

    build_tracker.assert_called_once_with()
    np.testing.assert_array_equal(particles.at_turn, [2])


@pytest.mark.parametrize('lose_particle', [False, True], ids=['complete', 'loss'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_feedback_consumes_particle_and_beam_monitor_records(
    test_context, lose_particle,
):
    """Apply the joint correction and hold it if a selected particle is lost."""
    env = xt.Environment()
    env['kick_y'] = 0.0
    env.new('corrector', xt.Multipole, ksl=[0.0])
    env.ref['corrector'].ksl[0] = env.ref['kick_y']
    env.new('arc', xt.Drift, length=1.0)
    env.new('aperture', xt.LimitRect, min_y=0.0 if lose_particle else -1.0)
    env.elements['particle_bpm'] = xt.ParticlesMonitor(
        _context=test_context, start_at_turn=0, stop_at_turn=2, num_particles=2,
    )
    env.elements['stats_bpm'] = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=2, stats=['mean_y'],
        particle_id_range=(1, 2),
    )

    fb = xt.feedback.Feedback(name='orbit', _context=test_context)
    fb.inputs.new(name='per_particle', monitor='particle_bpm', quantity='y')
    fb.inputs.new(name='centroid', monitor='stats_bpm', quantity='mean_y')
    fb.outputs.new(name='kick', target='kick_y')
    fb.program = '''
particle_centroid = sum(acquired.per_particle) / 2
outputs.kick = -(particle_centroid + acquired.centroid) / 2
'''
    env.elements['feedback'] = fb

    line = env.new_line(
        components=[
            'corrector', 'arc', 'aperture', 'particle_bpm', 'stats_bpm',
            'feedback',
        ],
    )
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    particles = xt.Particles(_context=test_context, p0c=7e12, y=[1e-6, 3e-6])
    line.feedback.enable()
    line.track(particles, num_turns=2)

    # The first readings are 2 and 3 µm. Their average drives the next kick.
    to_host = test_context.nparray_from_context_array
    surviving_id = to_host(particles.particle_id) == 1
    np.testing.assert_allclose(to_host(particles.py)[surviving_id], [-2.5e-6],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(to_host(particles.y)[surviving_id], [0.5e-6],
                               rtol=0, atol=1e-15)
    assert np.count_nonzero(to_host(particles.state) > 0) == 2 - lose_particle
    assert [row['source_valid'] for row in fb.log] == [True, not lose_particle]
    assert [row['executed'] for row in fb.log] == [True, not lose_particle]
    expected_kick = -2.5e-6 if lose_particle else 0.0
    assert env['kick_y'] == pytest.approx(expected_kick, abs=1e-15)


@pytest.fixture
def study(context):
    def make(
        *,
        active=True,
        tau=0.0,
        vector=False,
        duplicate=False,
        history_length=1,
        program=None,
        order=None,
        enable=True,
    ):
        env = xt.Environment()
        env['nominal'] = 0.0
        env['applied'] = 0.0
        env.new('corrector', xt.Multipole, ksl=[0.0])
        env.ref['corrector'].ksl[0] = env.ref['applied']
        env.elements['arc'] = xt.LineSegmentMap(
            _context=context,
            length=1000.0,
            qx=0.21,
            qy=0.31,
            betx=10.0,
            bety=10.0,
        )
        env.elements['bpm'] = xt.BeamStatsMonitor(
            _context=context,
            start_at_turn=0,
            stop_at_turn=64,
            stats=['num_particles', 'mean_y'],
        )
        target, baseline = 'applied', 'nominal'
        if vector:
            env['other_applied'] = 0.0
            env['other_nominal'] = 0.0
            env.new('other_corrector', xt.Multipole, ksl=[0.0])
            env.ref['other_corrector'].ksl[0] = env.ref['other_applied']
            target = ['applied', 'other_applied']
            baseline = ['nominal', 'other_nominal']
        hardware = xt.feedback.PhysicalActuator(
            env,
            target=target,
            baseline=baseline,
            lower=-1e-3,
            upper=1e-3,
            response_time_constant_seconds=tau,
        )
        fb = xt.feedback.Feedback(
            name='orbit',
            _context=context,
            active=active,
            update_turns=2,
            history_length=history_length,
        )
        fb.inputs.new(name='y', monitor='bpm', quantity='mean_y', unit='m')
        fb.outputs.new(name='kick', actuator=hardware, unit='rad')
        fb.parameters.new(name='gain', value=0.03, unit='rad/m')
        fb.state.new(name='integral', initial_value=0.0, unit='rad')
        fb.program = (
            program
            if program is not None
            else """
state.integral = state.integral - parameters.gain * acquired.y
outputs.kick = state.integral
"""
        )
        env.elements['feedback'] = fb
        components = ['corrector'] + (['other_corrector'] if vector else [])
        components += ['arc', 'bpm', 'feedback']
        if duplicate:
            copied = fb.copy(_context=context)
            copied.name = 'other'
            env.elements['other_feedback'] = copied
            components.append('other_feedback')
        line = env.new_line(components=order or components)
        line.particle_ref = xt.Particles(_context=context, p0c=7e12)
        kernel = getattr(context, '_feedback_public_kernel', None)
        line.build_tracker(
            _context=context, use_prebuilt_kernels=False, track_kernel=kernel
        )
        context._feedback_public_kernel = line.tracker.track_kernel
        particles = line.build_particles(y=[3e-6, 7e-6], py=[0.0, 0.0])
        if enable:
            line.feedback.enable()
        return env, line, fb, particles

    return make


def test_requested_update_rate_cannot_exceed_execution_opportunities(study):
    """Reject a calculation frequency above the revolution frequency."""
    _, line, fb, particles = study(enable=False)
    fb.update_turns = None
    fb.update_frequency_hz = 1e9
    line.feedback.enable()
    with pytest.raises(NotImplementedError, match='revolution frequency'):
        line.track(particles, num_turns=2)
    assert np.all(particles.at_turn == 0)
    assert not fb.log


@pytest.mark.parametrize('with_progress', [False, 4])
def test_split_tracking_preserves_acquisition_integral_and_applied_fields(
    study, with_progress,
):
    """Match an uninterrupted run with progress enabled and across calls."""
    env, line, fb, particles = study()
    other_env, other_line, other_fb, other_particles = study()
    line.track(particles, num_turns=15, with_progress=with_progress)
    for turns in [3, 1, 7, 4]:
        other_line.track(other_particles, num_turns=turns)
    np.testing.assert_array_equal(particles.y, other_particles.y)
    np.testing.assert_array_equal(particles.py, other_particles.py)
    np.testing.assert_array_equal(env['bpm'].mean_y, other_env['bpm'].mean_y)
    np.testing.assert_array_equal(
        fb.state.read('integral'), other_fb.state.read('integral')
    )
    assert env['applied'] == other_env['applied']
    np.testing.assert_array_equal(
        [row['applied_output'] for row in fb.log],
        [row['applied_output'] for row in other_fb.log],
    )


def test_disable_group_retains_hardware_response_and_resume_continues(study):
    """Pause the controller while the actuator follows the nominal request."""
    env, line, fb, particles = study(tau=1e-5)
    line.track(particles, num_turns=3)
    integral = fb.state.read('integral')
    applied = env['applied']
    line.feedback.enabled = False
    assert not line.feedback.enabled
    env['nominal'] = 1e-5
    line.track(particles, num_turns=4)
    np.testing.assert_array_equal(fb.state.read('integral'), integral)
    assert env['applied'] != applied
    assert not any(row['executed'] for row in fb.log[-4:])
    line.feedback.enabled = True
    assert line.feedback.enabled
    line.track(particles, num_turns=3)
    assert any(row['executed'] for row in fb.log[-3:])


def test_vector_output_preflight_uses_actual_applied_readback_shape(study):
    """Read back one value per actuator channel during program validation."""
    env, line, fb, particles = study(
        vector=True,
        program='outputs.kick = acquired.total_target + acquired.nominal_target',
    )
    line.track(particles, num_turns=3)
    assert fb.log[1]['requested_output'].shape == (2,)
    assert fb.log[-1]['applied_output'].shape == (2,)
    assert env['applied'] == env['other_applied'] == 0


def test_placed_program_uses_real_measurements_and_scheduler_fields(study):
    """Solve on a real BPM passage and read the scheduler at the same boundary."""
    env, line, fb, particles = study(program='''
matrix = stack((stack((acquired.y - 1.0,)),))
rhs = stack((1e-6 + 0.0 * acquired.y,))
solution = linalg.solve(matrix, rhs)
outputs.kick = solution[0] + 1e-6 * event.completed_turn
''')
    assert env['applied'] == 0.0
    line.track(particles, num_turns=2)
    expected = 1e-6 / (env.get('bpm').mean_y[1] - 1.0) + 2e-6
    assert fb.log[-1]['executed']
    assert env['applied'] == pytest.approx(expected, rel=1e-12)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_placed_callable_uses_transformed_monitor_history(test_context):
    """Drop an invalid reading and kick from two new context-native samples."""
    def transform(inputs, parameters, event):
        error = inputs['y'] - parameters['offset']
        return {'error': error}, event['completed_turn'] != 2

    def program(inputs, parameters, previous, event, xp):
        history = inputs['error_history']
        assert isinstance(history, test_context.nplike_array_type)
        assert event['completed_turn'] == 4
        trim = previous['trim'] - parameters['gain'] * xp.mean(history)
        return {'memory': {'trim': trim}, 'outputs': {'kick': trim}}

    env = xt.Environment()
    env['kick_rad'] = 0.0
    env.new('corrector', xt.Multipole, ksl=['kick_rad'])
    env.new('arc', xt.Drift, length=1.0)
    env.elements['bpm'] = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=5, stats=['mean_y'],
    )
    fb = xt.feedback.Feedback(
        name='orbit', program=program, acquisition_transform=transform,
        update_turns=10, first_calculation_turn=1, history_length=2,
    )
    fb.inputs.new(name='y', monitor='bpm', quantity='mean_y', unit='m')
    fb.parameters.new(name='offset', value=1e-6, unit='m')
    fb.parameters.new(name='gain', value=0.5, unit='rad/m')
    fb.state.new(name='trim', initial_value=0.0, unit='rad')
    fb.outputs.new(name='kick', target='kick_rad', unit='rad')
    env.elements['feedback'] = fb
    line = env.new_line(components=['bpm', 'corrector', 'arc', 'feedback'])
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    bunch = xt.Particles(_context=test_context, p0c=7e12, y=[1e-6, 3e-6])

    line.feedback.enable()
    line.track(bunch, num_turns=2)
    assert env['kick_rad'] == 0.0
    line.track(bunch, num_turns=3)

    valid_samples = [True, False, True, True, True]
    assert [row['source_valid'] for row in fb.log] == valid_samples
    assert [row['completed_turn'] for row in fb.log if row['executed']] == [4]
    to_host = test_context.nparray_from_context_array
    np.testing.assert_allclose(
        to_host(bunch.py), [-0.5e-6, -0.5e-6], rtol=0, atol=1e-16,
    )
    assert env['kick_rad'] == pytest.approx(-0.5e-6, abs=1e-16)


def test_placed_callable_invalid_output_preserves_field_and_state(study):
    """Reject a bad Python result before committing the field or memory."""
    def program(previous):
        return {
            'memory': {'integral': previous['integral'] + 1e-6},
            'outputs': {'kick': [1e-6, 2e-6]},
        }

    env, line, fb, particles = study(program=program)
    with pytest.raises(ValueError, match='per actuator channel'):
        line.track(particles, num_turns=2)

    assert env['applied'] == 0.0
    assert fb.state.read('integral') == 0.0
    assert [row['executed'] for row in fb.log] == [False]


def test_prepared_callable_rejects_a_replaced_transform(study):
    """Reject a transformed configuration change before another transport."""
    def program(inputs):
        return {'outputs': {'kick': -inputs['y']}}

    def transform(inputs):
        return inputs, True

    env, line, fb, particles = study(enable=False, program=program)
    fb.acquisition_transform = transform
    line.feedback.enable()
    line.track(particles, num_turns=1)
    fb.acquisition_transform = lambda inputs: (inputs, True)

    with pytest.raises(RuntimeError, match='configuration|changed'):
        line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.at_turn, [1, 1])
    assert env['bpm'].num_particles[1] == 0


def test_placed_text_program_requires_untransformed_sources(study):
    """Report the static input schema requirement before tracking."""
    _, line, fb, particles = study(enable=False)
    fb.acquisition_transform = lambda inputs: (inputs, True)
    with pytest.raises(ValueError, match='requires a callable program'):
        line.feedback.enable()
    assert fb._runtime is None
    np.testing.assert_array_equal(particles.at_turn, [0, 0])


@pytest.mark.parametrize('command, error, message', [
    ('stack((1e-6, 2e-6))', ValueError, 'per actuator channel'),
    ('1e309', ValueError, 'commands must be finite'),
    ('1j', TypeError, 'commands must be real'),
])
def test_placed_program_invalid_command_does_not_commit(study, command, error, message):
    """Preserve the corrector and retained integral after a rejected command."""
    env, line, fb, particles = study(
        program=f'state.integral = state.integral + 1e-6\noutputs.kick = {command}',
    )
    with pytest.raises(error, match=message):
        line.track(particles, num_turns=2)
    assert np.isfinite(env.get('bpm').mean_y[1])
    assert env['applied'] == 0.0
    assert float(fb.state.read('integral')) == 0.0
    assert all(not row['executed'] for row in fb.log)


@pytest.mark.parametrize(
    'options',
    [
        {'ele_start': 'arc'},
        {'ele_stop': 'bpm'},
        {'num_elements': 1},
        {'backtrack': True},
    ],
)
def test_partial_or_backward_requests_fail_before_particles_advance(study, options):
    """Reject partial or backward tracking before changing the beam."""
    _, line, _, particles = study()
    initial = particles.copy()
    with pytest.raises(NotImplementedError, match='complete-turn'):
        line.track(particles, num_turns=2, **options)
    np.testing.assert_array_equal(particles.y, initial.y)
    np.testing.assert_array_equal(particles.at_turn, initial.at_turn)


def test_nonterminal_feedback_is_rejected_before_tracking(study):
    """Require a complete traversal before calculating the correction."""
    with pytest.raises(ValueError, match='terminal'):
        study(order=['corrector', 'feedback', 'arc', 'bpm'])


def test_inspect_and_disable_feedback_on_an_unbuilt_composed_line(study):
    """Expose placed feedback before a tracker or runtime exists."""
    env, _, fb, _ = study(enable=False)
    line = env.new_line(
        components=['corrector', 'arc', 'bpm', xt.Place('feedback')],
        compose=True,
    )
    assert line.feedback['orbit'] is fb
    table = line.feedback.get_table()
    np.testing.assert_array_equal(table.name, ['orbit'])
    np.testing.assert_array_equal(table.element, ['feedback'])
    assert line.tracker is None

    line.feedback.disable()
    assert fb.active
    assert not fb.enabled
    with pytest.raises(RuntimeError, match='frozen after preparation'):
        fb.parameters.new(name='other_gain', value=1.0)


def test_enable_without_feedback_leaves_trackers_untouched(study):
    """Reject activation without allocating or discarding trackers."""
    env, owner, _, _ = study()
    tracker = owner.tracker
    line = env.new_line(components=['arc'], compose=True)

    with pytest.raises(ValueError, match='No feedback elements'):
        line.feedback.enable()

    assert line.tracker is None
    assert owner.tracker is tracker


def test_rewound_clock_and_rebuilt_tracker_are_rejected_before_transport(
    study, context
):
    """Keep each prepared run attached to its clock and tracker."""
    _, line, _, particles = study()
    line.track(particles, num_turns=2)
    old_y = particles.y.copy()
    particles.time_s = 0.0
    particles.reference_turns = 0.0
    particles.at_turn[:] = 0
    with pytest.raises(RuntimeError, match='clock/turn'):
        line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.y, old_y)
    _, line, _, particles = study()
    line.discard_tracker()
    line.build_tracker(
        _context=context,
        use_prebuilt_kernels=False,
        track_kernel=context._feedback_public_kernel,
    )
    old_y = particles.y.copy()
    with pytest.raises(RuntimeError, match='tracker changed'):
        line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.y, old_y)


@pytest.mark.parametrize('at_origin', [True, False])
@pytest.mark.parametrize('slice_mode', [False, True], ids=['bunch', 'slice'])
def test_bunched_source_completion_requires_the_supported_origin(
    study, context, at_origin, slice_mode
):
    """Use complete bunch and slice records only at the cycle origin."""
    order = ['bpm', 'corrector', 'arc', 'feedback'] if at_origin else None
    env, line, fb, particles = study(enable=False, order=order)
    line.discard_tracker()
    del env.elements['bpm']
    slicing = dict(zeta_range=(-100.0, 100.0), num_slices=2) if slice_mode else {}
    env.elements['bpm'] = xt.BeamStatsMonitor(
        _context=context,
        start_at_turn=0,
        stop_at_turn=64,
        bunch_spacing_zeta=250.0,
        filled_slots=[0],
        stats=['mean_y'],
        **slicing,
    )
    particles.zeta[:] = [-62.5, 62.5] if slice_mode else -62.5
    fb.program = """
state.integral = state.integral - parameters.gain * sum(acquired.y)
outputs.kick = state.integral
"""
    line.build_tracker(
        _context=context,
        use_prebuilt_kernels=False,
        track_kernel=context._feedback_public_kernel,
    )
    if not at_origin:
        with pytest.raises(NotImplementedError, match='cycle origin'):
            line.feedback.enable()
        assert np.all(particles.at_turn == 0)
        return
    line.feedback.enable()
    line.track(particles, num_turns=8)
    assert len(fb.log) == 8
    assert all(row['source_valid'] for row in fb.log)


@pytest.mark.parametrize('live_program', [False, True], ids=['text', 'python'])
def test_normal_line_copy_rebinds_configuration_to_the_new_environment(
    study, context, tmp_path, live_program,
):
    """Bind a cold copy's feedback and corrector to an independent environment."""
    _, line, fb, _ = study(enable=False)
    if live_program:
        fb.program = lambda inputs: {'outputs': {'kick': -inputs['y']}}
        fb.acquisition_transform = lambda inputs: (inputs, True)
        filename = tmp_path / 'live-feedback.json'
        filename.write_text('existing contents')
        with pytest.raises(NotImplementedError, match='Python callables'):
            line.to_json(filename)
        assert filename.read_text() == 'existing contents'
    copied = line.copy(_context=context)
    copied.build_tracker(
        _context=context,
        use_prebuilt_kernels=False,
        track_kernel=context._feedback_public_kernel,
    )
    copied.feedback.enable()
    particles = copied.build_particles(y=[3e-6, 7e-6], py=[0.0, 0.0])
    copied.track(particles, num_turns=3)
    copied_fb = copied.feedback['orbit']
    assert copied_fb is not fb
    if live_program:
        assert copied_fb.program is fb.program
        assert copied_fb.acquisition_transform is fb.acquisition_transform
    assert copied_fb.log[1]['executed']
    assert fb._runtime is None


def test_normal_line_json_restores_typed_feedback_without_extra_classes(
    study, context, tmp_path
):
    """Round-trip feedback bindings and the initial real and complex state."""
    env, line, fb, _ = study(enable=False)
    fb.parameters.new(name='phasor', value=1 + 2j, dtype=np.complex64)
    fb.state.new(name='mode', initial_value=np.array([2 - 3j]))
    fb.state.new(name='empty_mode', initial_value=np.zeros((0, 2)))
    exported = fb.to_dict()
    exported['inputs'][0]['monitor'] = 'different_bpm'
    assert fb.inputs['y'].monitor == 'bpm'
    with pytest.raises(ValueError, match='WRITEABLE'):
        fb.parameters['phasor'].value.flags.writeable = True
    # Native transport keeps this test independent of LineSegmentMap RF defaults.
    line.discard_tracker()
    del env.elements['arc']
    env.elements['arc'] = xt.Drift(length=1000.0, _context=context)
    filename = tmp_path / 'configured_feedback.json'
    line.to_json(filename)
    restored = xt.Line.from_json(filename, verbose=False)
    restored_fb = restored.feedback['orbit']
    phasor = restored_fb.parameters['phasor'].value
    assert phasor.shape == ()
    assert phasor.dtype == np.complex64
    assert phasor.item() == 1 + 2j
    np.testing.assert_array_equal(restored_fb.state.read('mode'), [2 - 3j])
    assert restored_fb.state.read('empty_mode').shape == (0, 2)
    restored.build_tracker(
        _context=context,
        use_prebuilt_kernels=False,
        track_kernel=context._feedback_public_kernel,
    )
    restored.feedback.enable()
    particles = restored.build_particles(y=[3e-6, 7e-6], py=[0.0, 0.0])
    restored.track(particles, num_turns=4)
    assert isinstance(restored.feedback['orbit'], xt.feedback.Feedback)
    assert restored.feedback.enabled
    assert restored.feedback['orbit'].enabled
    assert len(restored.feedback['orbit'].log) == 4
    assert fb._runtime is None


@pytest.mark.parametrize('reject', [
    'freeze_longitudinal', 'hidden_start', 'short_pickup_frame',
])
def test_rejected_track_does_not_bind_particles_or_touch_pickup(study, reject):
    """Preserve the cold run when any part of the tracking request is invalid."""
    env, line, fb, particles = study()
    accepted = particles.copy()
    weights = env['bpm'].num_particles.copy()
    options = {}
    if reject == 'hidden_start':
        particles.start_tracking_at_element = line.element_names.index('bpm')
    elif reject == 'freeze_longitudinal':
        options['freeze_longitudinal'] = True
    else:
        options.update(num_turns=65, with_progress=8)
    start = particles.start_tracking_at_element

    with pytest.raises(
        (NotImplementedError, ValueError),
        match='complete-turn|freeze_longitudinal|monitor frame',
    ):
        line.track(particles, **options)

    assert particles.start_tracking_at_element == start
    assert fb._runtime is None
    np.testing.assert_array_equal(particles.at_turn, 0)
    np.testing.assert_array_equal(env['bpm'].num_particles, weights)
    line.track(accepted, num_turns=1)
    np.testing.assert_array_equal(accepted.at_turn, 1)


def test_invalid_native_monitor_leaves_feedback_unbound(study):
    """Run native monitor validation before binding feedback to the beam."""
    env, line, fb, particles = study()
    start = particles.copy()
    with pytest.raises(ValueError, match='valid monitor object'):
        line.track(particles, num_turns=2, turn_by_turn_monitor='invalid')

    assert fb._runtime is None
    np.testing.assert_array_equal(particles.at_turn, 0)
    np.testing.assert_array_equal(env['bpm'].num_particles, 0)
    line.track(start, num_turns=2, with_progress=1)
    assert [entry['completed_turn'] for entry in fb.log] == [1, 2]


def test_changed_output_graph_is_rejected_before_continuation(study):
    """Detect a changed Xdeps graph before the next pickup visit."""
    env, line, fb, particles = study()
    line.track(particles, num_turns=1)
    env.ref['corrector'].knl[0] = env.ref['applied']
    start = particles.copy()
    weights = env['bpm'].num_particles.copy()

    with pytest.raises(RuntimeError, match='dependency graph changed'):
        line.track(particles, num_turns=1)

    assert len(fb.log) == 1
    np.testing.assert_array_equal(particles.at_turn, start.at_turn)
    np.testing.assert_array_equal(particles.x, start.x)
    np.testing.assert_array_equal(env['bpm'].num_particles, weights)


def test_prepared_feedback_preflights_once_with_automatic_pipeline(
    study, monkeypatch
):
    """Admit a prepared run once when tracking through automatic pipeline."""
    _, line, fb, particles = study()
    execution = line._feedback_execution
    original = execution.validate_track
    calls = []

    def count(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, 'validate_track', count)
    particles.init_pipeline('beam')
    line.track(particles, num_turns=3)

    assert len(calls) == 1
    assert [entry['completed_turn'] for entry in fb.log] == [1, 2, 3]


@pytest.mark.parametrize('interruption, error, message', [
    ('reference_turns', RuntimeError, 'reference turns changed within'),
    ('time_s', RuntimeError, 'clock changed within'),
    ('t_sim', RuntimeError, 'turn period changed within'),
    ('hold', NotImplementedError, 'cannot resume a native pipeline hold'),
])
def test_prepared_feedback_rejects_interrupted_traversal(
    study, context, interruption, error, message,
):
    """Refuse a clock edit or native hold after a pickup has recorded."""
    env, original, fb, particles = study(enable=False)

    class Interrupt:
        iscollective = True
        isthick = False

        def track(self, particles):
            if interruption == 'hold':
                return xt.PipelineStatus(on_hold=True)
            clock = particles._xobject
            value = float('nan') if interruption == 'time_s' else (
                getattr(clock, interruption) + 0.25
            )
            setattr(clock, interruption, value)

    env.elements['interrupt'] = Interrupt()
    line = env.new_line(components=['corrector', 'arc', 'bpm', 'interrupt', 'feedback'])
    line.build_tracker(
        _buffer=original.tracker._buffer, enable_pipeline_hold=True,
        use_prebuilt_kernels=False, track_kernel=context._feedback_public_kernel,
    )
    line.feedback.enable()
    with pytest.raises(error, match=message):
        line.track(particles, num_turns=2)
    assert env['bpm'].num_particles[0] > 0
    assert not fb.log
    assert env['applied'] == 0.0
    with pytest.raises(RuntimeError, match='reconstruct a consistent run'):
        line.track(particles, num_turns=1)


@pytest.mark.parametrize('alias_kind', ['shallow', 'built', 'compose'])
def test_inactive_prepared_feedback_cannot_track_through_a_shallow_alias(
    study, context, alias_kind
):
    """Protect a paused owner's pickup storage from shallow aliases."""
    env, line, fb, particles = study(enable=False)
    tracker = line.tracker
    if alias_kind == 'compose':
        alias = env.new_line(components=list(line.element_names), compose=True)
    else:
        alias = line.copy(shallow=True)
    if alias_kind == 'built':
        alias.build_tracker(
            _buffer=tracker._buffer,
            use_prebuilt_kernels=False,
            track_kernel=context._feedback_public_kernel,
        )
    line.feedback.enable()
    line.track(particles, num_turns=1)
    fb.disable()
    weights = env['bpm'].num_particles.copy()
    other = particles.copy()

    with pytest.raises(RuntimeError, match='prepared on another line'):
        alias.track(other, num_turns=1)
    with pytest.raises(RuntimeError, match='prepared on another line'):
        alias.feedback.enable()
    if alias_kind == 'built':
        with pytest.raises(NotImplementedError, match='feedback'):
            alias._get_non_collective_line()
    else:
        with pytest.raises(RuntimeError, match='prepared on another line'):
            alias.build_tracker()
        assert alias.tracker is None

    assert line.tracker is tracker
    np.testing.assert_array_equal(other.at_turn, 1)
    np.testing.assert_array_equal(other.s, 0)
    np.testing.assert_array_equal(env['bpm'].num_particles, weights)
    assert len(fb.log) == 1
    line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.at_turn, 2)


@pytest.mark.parametrize('compose', [False, True])
def test_placed_feedback_replica_is_refused_before_transport(study, compose):
    """Reject a replica while the original line retains its feedback runtime."""
    env, original, fb, particles = study()
    original.track(particles, num_turns=1)
    tracker = original.tracker
    env.elements['feedback_replica'] = xt.Replica(parent_name='feedback')
    line = env.new_line(
        components=['corrector', 'arc', 'bpm', 'feedback_replica'],
        compose=compose,
    )
    weights = env['bpm'].num_particles.copy()
    other = particles.copy()

    with pytest.raises(NotImplementedError, match='Replica of a placed feedback'):
        line.track(other, num_turns=1)
    with pytest.raises(NotImplementedError, match='Replica of a placed feedback'):
        line.feedback.enable()
    assert line.tracker is None

    assert original.tracker is tracker
    assert len(fb.log) == 1
    np.testing.assert_array_equal(other.at_turn, 1)
    np.testing.assert_array_equal(env['bpm'].num_particles, weights)
    original.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.at_turn, 2)


def test_placed_feedback_rejects_replicated_reference_energy_change(study):
    """Find unsupported reference-energy changes behind replicas."""
    env, original, fb, _ = study(enable=False)
    env.elements['ramp'] = xt.ReferenceEnergyIncrease(Delta_p0c=1e9)
    env.elements['ramp_replica'] = xt.Replica(parent_name='ramp')
    line = env.new_line(
        components=['corrector', 'arc', 'ramp_replica', 'bpm', 'feedback']
    )
    line.build_tracker(_buffer=original.tracker._buffer, compile=False)

    with pytest.raises(NotImplementedError, match='reference-energy changes'):
        line.feedback.enable()
    assert fb._execution is None


def test_noncollective_probe_is_refused_without_touching_live_records(study, context):
    """Protect a live pickup and corrector from a shared noncollective probe."""
    env, line, fb, particles = study()
    line.track(particles, num_turns=2)
    old_log = len(fb.log)
    old_integral = fb.state.read('integral')
    old_field = env['applied']
    old_weights = np.asarray(env['bpm'].num_particles).copy()
    probe = xt.Particles(_context=context, p0c=7e12, y=[1e-6], py=[0.0])
    with pytest.raises(NotImplementedError, match='feedback|monitor|probe'):
        line._get_non_collective_line()
    assert len(fb.log) == old_log
    np.testing.assert_array_equal(fb.state.read('integral'), old_integral)
    assert env['applied'] == old_field
    np.testing.assert_array_equal(env['bpm'].num_particles, old_weights)
    np.testing.assert_array_equal(particles.at_turn, [2, 2])
    np.testing.assert_array_equal(probe.at_turn, [0])


@pytest.mark.parametrize('epoch', [1e9, 1e12])
def test_epoch_without_turn_clock_resolution_fails_before_transport(study, epoch):
    """Reject epochs that cannot represent the required clock resolution."""
    _, line, fb, particles = study()
    particles.time_s = epoch
    particles.reference_turns = 0.0
    initial = particles.y.copy()
    with pytest.raises(ValueError, match='clock resolution'):
        line.track(particles, num_turns=1)
    assert fb._runtime is None
    np.testing.assert_array_equal(particles.y, initial)


@pytest.mark.parametrize('chunks', [(4,), (2, 2)], ids=['bulk', 'split'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_feedback_delivers_queued_command_after_beam_loss(test_context, chunks):
    """Advance the clock and queued field after the last particle is lost."""
    env = xt.Environment()
    env['kick_rad'] = 0.0
    env.new('drift', xt.Drift, length=1.0)
    env.new('aperture', xt.LimitRect, min_y=0.0)
    env.new('corrector', xt.Multipole, ksl=['kick_rad'])
    bpm = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=4, stats=['mean_y'],
    )
    bunch = xt.Particles(_context=test_context, p0c=7e12, y=1.5e-6, py=-1e-6)
    period = 1.0 / (float(bunch._xobject.beta0[0]) * 299792458.0)
    fb = xt.feedback.Feedback(
        name='orbit', update_turns=1, command_delay_seconds=2 * period,
    )
    fb.inputs.new(name='y', monitor='bpm', quantity='mean_y')
    fb.outputs.new(name='kick', target='kick_rad')
    fb.program = 'outputs.kick = acquired.y'
    env.elements['bpm'] = bpm
    env.elements['feedback'] = fb
    line = env.new_line(components=[
        'drift', 'aperture', 'bpm', 'corrector', 'feedback',
    ])
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    line.feedback.enable()
    for turns in chunks:
        line.track(bunch, num_turns=turns)

    to_host = test_context.nparray_from_context_array
    np.testing.assert_array_equal(to_host(bunch.at_turn), [1])
    assert to_host(bunch.state)[0] == 0  # Native LimitRect loss code.
    assert bunch.reference_turns == 4
    assert bunch.time_s == pytest.approx(4 * period, rel=1e-15)
    expected_validity = [True, False, False, False]
    assert [row['source_valid'] for row in fb.log] == expected_validity
    assert [row['executed'] for row in fb.log] == expected_validity
    trace = [to_host(row['applied_output'])[0] for row in fb.log]
    np.testing.assert_allclose(trace, [0, 0, 0.5e-6, 0.5e-6], rtol=1e-11)
    assert env['kick_rad'] == pytest.approx(0.5e-6, rel=1e-11)


@pytest.mark.parametrize('epoch', [0.0, 1e6])
@pytest.mark.parametrize(('offset', 'delivery'), [(-10e-9, 4), (0, 4), (10e-9, 5)])
def test_placed_command_resolves_near_turn_deadlines(study, epoch, offset, delivery):
    """Distinguish commands due on either side of a traversal boundary."""
    _, line, fb, particles = study(
        enable=False, program='outputs.kick = 1e-6 + 0 * acquired.y'
    )
    period = line.get_length() / (float(particles.beta0[0]) * 299792458.0)
    fb.command_delay_seconds = 2 * period + offset
    particles.time_s = epoch
    particles.reference_turns = 0.0
    line.feedback.enable()
    line.track(particles, num_turns=6)

    first_delivery = next(row for row in fb.log if row['applied_output'][0] != 0)
    assert first_delivery['completed_turn'] == delivery
    assert first_delivery['time_seconds'] == epoch + delivery * period


@pytest.mark.parametrize('cadence', ['turns', 'seconds'])
def test_placed_cadence_defaults_start_at_tracking_epoch(study, cadence):
    """Keep equivalent first deadlines and elapsed times at a nonzero epoch."""
    _, line, fb, particles = study(enable=False, program='outputs.kick = 0.0')
    period = line.get_length() / (float(particles.beta0[0]) * 299792458.0)
    if cadence == 'seconds':
        fb.update_turns = None
        fb.update_frequency_hz = 1 / (2 * period)
    particles.at_turn[:] = 10
    particles.time_s = 1e6
    particles.reference_turns = 10
    line.feedback.enable()
    line.track(particles, num_turns=4)

    updates = [row for row in fb.log if row['executed']]
    assert [row['completed_turn'] for row in updates] == [12, 14]
    assert [row['missed_deadlines'] for row in updates] == [0, 0]
    assert updates[0]['dt_seconds'] == 0
    assert updates[1]['dt_seconds'] == pytest.approx(2 * period, rel=0, abs=1e-17)


@pytest.mark.parametrize('epoch', [0.0, 1e6])
def test_default_hz_deadline_uses_elapsed_turn_clock(study, epoch):
    """Resolve the first Hz deadline 100 ps after the first boundary."""
    env, line, fb, particles = study(
        enable=False, program='outputs.kick = 1e-6',
    )
    period = line.get_length() / (float(particles.beta0[0]) * 299792458.0)
    interval = period + 100e-12
    fb.update_turns = None
    fb.update_frequency_hz = 1 / interval
    particles.time_s = epoch
    particles.reference_turns = 0.0
    line.feedback.enable()
    line.track(particles, num_turns=1)
    assert env['applied'] == 0.0
    line.track(particles, num_turns=1)
    update, = [row for row in fb.log if row['executed']]
    assert update['completed_turn'] == 2
    assert update['nominal_time_seconds'] == epoch + interval
    assert env['applied'] == 1e-6


def test_stopped_actuator_edit_is_rejected_before_transport(study):
    """Protect the beam and pickup records from an external applied-field edit."""
    env, line, fb, particles = study()
    line.track(particles, num_turns=2)
    old_y = particles.y.copy()
    old_py = particles.py.copy()
    old_weights = np.asarray(env['bpm'].num_particles).copy()
    old_log_length = len(fb.log)
    env['applied'] = 2e-4
    with pytest.raises(RuntimeError, match='outside its response owner'):
        line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.at_turn, [2, 2])
    np.testing.assert_array_equal(particles.y, old_y)
    np.testing.assert_array_equal(particles.py, old_py)
    np.testing.assert_array_equal(env['bpm'].num_particles, old_weights)
    assert len(fb.log) == old_log_length


def test_prepopulated_source_records_reject_a_fresh_beam_before_transport(study):
    """Prevent an earlier beam's pickup records entering a new run."""
    env, line, fb, particles = study(active=False, enable=False)
    probe = particles.copy()
    line.track(probe, num_turns=1)
    old_weights = np.asarray(env['bpm'].num_particles).copy()
    fb.enable()
    line.feedback.enable()
    old_y = particles.y.copy()
    with pytest.raises(ValueError, match='record|monitor|empty'):
        line.track(particles, num_turns=2)
    np.testing.assert_array_equal(particles.y, old_y)
    np.testing.assert_array_equal(particles.at_turn, [0, 0])
    np.testing.assert_array_equal(env['bpm'].num_particles, old_weights)


@pytest.mark.parametrize('binding', ['same_knob', 'same_field', 'disjoint'])
@pytest.mark.parametrize('compose', [False, True])
def test_placed_writers_are_exclusive_across_lines(study, context, binding, compose):
    """Allow sibling lines to control only disjoint knobs and fields."""
    env, line, _, particles = study(
        enable=False, program='outputs.kick = 2e-6',
    )
    env['sibling_rad'] = 0.0
    if binding == 'same_field':
        env.ref['corrector'].ksl[0] = (
            env.ref['applied'] + env.ref['sibling_rad']
        )
    elif binding == 'disjoint':
        env.ref['corrector'].knl[0] = env.ref['sibling_rad']

    env.elements['sibling_bpm'] = env.get('bpm').copy(_context=context)
    sibling_feedback = xt.feedback.Feedback(name='sibling', _context=context)
    sibling_feedback.inputs.new(
        name='y', monitor='sibling_bpm', quantity='mean_y',
    )
    target = 'applied' if binding == 'same_knob' else 'sibling_rad'
    sibling_feedback.outputs.new(name='kick', target=target)
    sibling_feedback.program = 'outputs.kick = 3e-6'
    env.elements['sibling_feedback'] = sibling_feedback
    feedback_place = (xt.Place('sibling_feedback') if compose else 'sibling_feedback')
    sibling = env.new_line(
        components=['corrector', 'arc', 'sibling_bpm', feedback_place],
        compose=compose,
    )
    sibling.particle_ref = line.particle_ref.copy()
    sibling_particles = xt.Particles(_context=context, p0c=7e12, y=[1e-6])

    line.feedback.enable()
    line.track(particles, num_turns=2)
    line.feedback.enabled = False
    first_tracker = line.tracker

    with pytest.raises(RuntimeError, match=r'line.feedback.enable\(\)'):
        sibling.track(sibling_particles, num_turns=1)
    assert sibling.tracker is None
    assert line.tracker is first_tracker

    if binding == 'disjoint':
        sibling.feedback.enable()
        sibling.track(sibling_particles, num_turns=1)
        assert env.get('corrector').knl[0] == 3e-6
    else:
        with pytest.raises(ValueError, match='share a contribution or physical'):
            sibling.feedback.enable()
        assert not sibling.feedback.enabled
        np.testing.assert_array_equal(sibling_particles.at_turn, [0])
        np.testing.assert_array_equal(env.get('sibling_bpm').num_particles, 0)

    assert line.tracker is first_tracker
    line.track(particles, num_turns=1)
    assert env.get('corrector').ksl[0] == 2e-6


@pytest.mark.parametrize(
    'change', ['program', 'geometry', 'context', 'unprepared', 'name']
)
def test_invalid_preflight_configuration_does_not_advance_particles(study, change):
    """Reject an incompatible run before transporting the beam."""
    env, line, fb, particles = study(enable=change != 'unprepared')
    if change == 'program':
        fb.program = 'outputs.kick = acquired.y'
    elif change == 'geometry':
        env['arc'].length = 2000.0
    elif change == 'context':
        particles = particles.copy(_context=xo.ContextCpu())
    elif change == 'name':
        fb.name = 'renamed'
    old_y, old_turn = particles.y.copy(), particles.at_turn.copy()
    with pytest.raises((RuntimeError, ValueError), match='changed|context|enable'):
        line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.y, old_y)
    np.testing.assert_array_equal(particles.at_turn, old_turn)


def test_enable_rejects_geometry_changed_since_tracker_build(study):
    """Validate current circumference against the native tracker's geometry."""
    env, line, fb, particles = study(enable=False)
    env['arc'].length = 2000.0
    with pytest.raises(RuntimeError, match='line geometry changed'):
        line.feedback.enable()
    assert not fb.enabled
    np.testing.assert_array_equal(particles.at_turn, [0, 0])


def test_enable_rejects_moved_pickup_at_fixed_circumference(study):
    """Reject a displaced pickup even when the revolution period is unchanged."""
    env, original, fb, _ = study(enable=False)
    env.new('tail', xt.Drift, length=10.0)
    line = env.new_line(components=['corrector', 'arc', 'bpm', 'tail', 'feedback'])
    line.build_tracker(
        _buffer=original.tracker._buffer, use_prebuilt_kernels=False,
        track_kernel=original.tracker.track_kernel,
    )
    circumference = line.get_length()
    env['arc'].length += 1.0
    env['tail'].length -= 1.0

    assert line.get_length() == circumference
    with pytest.raises(RuntimeError, match='line geometry changed'):
        line.feedback.enable()
    assert not fb.enabled
