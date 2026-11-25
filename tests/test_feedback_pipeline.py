# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Actual tracking tests for the experimental end-turn pipeline coordinator."""

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts

from xtrack.feedback import (
    CommandQueue, FeedbackController, FeedbackSystem, KnobOutput,
    SharedFeedbackBoundary,
)
from xtrack.feedback.readout import bind_monitor

pytestmark = pytest.mark.usefixtures('sandbox_cwd')


class PairwiseExchange:
    """Toy existing collective: exchange centroids before applying a pair kick.

    Two stations use different partners. This exercises existing pipeline
    holds/resumes; it is not a beam-beam force or encounter-time model.
    """

    iscollective = True
    needs_cpu = False

    def __init__(self, station, name, partner, state):
        self.station, self.name, self.partner = station, name, partner
        self.state = state

    def track(self, particles):
        turn = int(round(float(particles.reference_turns)))
        own = (self.station, turn, self.name)
        other = (self.station, turn, self.partner)
        self.state['samples'].setdefault(own, float(np.mean(particles.y)))
        if other not in self.state['samples']:
            self.state['holds'] += 1
            return xt.PipelineStatus(on_hold=True, info='toy pair centroid exchange')
        if own not in self.state['applied']:
            particles.py += 0.02 * (
                self.state['samples'][other] - self.state['samples'][own]
            )
            self.state['applied'].add(own)


@pytest.fixture(scope='module')
def context():
    context = xo.ContextCpu()
    context.allow_kernel_compilation = True
    return context


def build_study(
    context,
    callback,
    *,
    order=('p1', 'p2', 'e1', 'e2'),
    epoch=0.0,
    length_by_branch=None,
    lost_branch=None,
    knob_outputs=False,
    pairwise_exchanges=False,
    before_install=None,
    after_feedback=None,
):
    names = ('p1', 'p2', 'e1', 'e2')
    boundary = SharedFeedbackBoundary(
        names, callback, reference_branch='e1', after_feedback=after_feedback,
    )
    lines = {}
    particles = {}
    monitors = {}
    exchange = {'samples': {}, 'applied': set(), 'holds': 0}
    partners = {
        'a': dict(p1='e1', e1='p1', p2='e2', e2='p2'),
        'b': dict(p1='e2', e2='p1', p2='e1', e1='p2'),
    }
    for i, name in enumerate(names):
        length = (length_by_branch or {}).get(name, 1.0)
        monitors[name] = xt.BeamStatsMonitor(
            start_at_turn=0,
            stop_at_turn=16,
            stats=['num_particles', 'mean_y', 'mean_py'],
        )
        if pairwise_exchanges:
            elements = [
                xt.Drift(length=length / 2),
                PairwiseExchange('a', name, partners['a'][name], exchange),
                xt.Drift(length=length / 2),
                PairwiseExchange('b', name, partners['b'][name], exchange),
                monitors[name],
                xt.Multipole(ksl=[0.0]),
            ]
            element_names = ['first_half', 'ip_a', 'second_half', 'ip_b', 'bpm', 'kick']
        else:
            elements = [
                xt.Drift(length=length),
                monitors[name],
                xt.Multipole(ksl=[0.0]),
            ]
            element_names = ['drift', 'bpm', 'kick']
        lines[name] = xt.Line(
            elements=elements,
            element_names=element_names,
        )
        if knob_outputs:
            lines[name]['feedback_rad'] = 0.0
            lines[name].ref['kick'].ksl[0] = lines[name].ref['feedback_rad']
        particles[name] = xt.Particles(
            _context=context,
            p0c=7e12,
            y=[(i + 1) * 1e-6, (i + 1) * 1e-6],
            weight=[1.0, 2.0],
            time_s=epoch,
        )
        if name == lost_branch:
            particles[name].state[:] = -1
    if before_install is not None:
        before_install(lines, particles)
    boundary.install(lines)
    for line in lines.values():
        # These test lines have identical element classes/configuration. Reuse
        # the supported tracker-kernel argument instead of recompiling each.
        line.build_tracker(
            _context=context,
            use_prebuilt_kernels=False,
            track_kernel=getattr(context, '_feedback_pipeline_test_kernel', None),
        )
        context._feedback_pipeline_test_kernel = line.tracker.track_kernel
    multitracker = xt.PipelineMultiTracker(
        branches=[
            xt.PipelineBranch(line=lines[name], particles=particles[name])
            for name in order
        ]
    )
    return boundary, lines, particles, monitors, multitracker


def test_incompatible_period_rejected(context):
    """Reject branches whose full-turn periods differ."""
    boundary, lines, particles, monitors, multitracker = build_study(
        context, lambda **kw: None, length_by_branch={'p2': 1.01}
    )
    with pytest.raises(ValueError, match='incompatible revolution period'):
        boundary.bind(multitracker)


@pytest.mark.parametrize('element, settings, as_replica', [
    (xt.ReferenceEnergyChange, {'p0c': 8e9}, False),
    (xt.ReferenceEnergyIncrease, {'Delta_p0c': 1e9}, False),
    (xt.ReferenceEnergyIncrease, {'Delta_p0c': 1e9}, True),
    (xt.LineSegmentMap, {'length': 1., 'energy_ref_increment': 1e6}, False),
])
def test_native_reference_energy_changes_are_rejected_at_bind(
    context, element, settings, as_replica
):
    """Reject a known reference-energy element before transport."""
    events = []
    boundary = SharedFeedbackBoundary(
        ['beam'], lambda **event: events.append(event)
    )
    env = xt.Environment(element_dict={'ramp': element(**settings)})
    env.new('drift', xt.Drift, length=10)
    if as_replica:
        env.elements['ramp_replica'] = xt.Replica(parent_name='ramp')
    ramp = 'ramp_replica' if as_replica else 'ramp'
    line = env.new_line(components=['drift', ramp])
    boundary.install({'beam': line})
    line.build_tracker(_context=context, compile=False)
    particles = xt.Particles(_context=context, p0c=7e9)
    multitracker = xt.PipelineMultiTracker([xt.PipelineBranch(line, particles)])

    with pytest.raises(NotImplementedError, match='element-driven reference-energy'):
        boundary.bind(multitracker)

    assert events == []
    np.testing.assert_array_equal(particles.at_turn, 0)
    np.testing.assert_array_equal(particles.p0c, 7e9)


@pytest.mark.parametrize('change, error, message', [
    ('p0c', NotImplementedError, 'reference energy changed while tracking'),
    ('reference_turns', ValueError, 'reference turns changed within'),
    ('time_s', ValueError, 'clock changed within'),
    ('t_sim', ValueError, 'reference period changed while tracking'),
])
def test_invalid_clock_at_boundary_prevents_callback(context, change, error, message):
    """Reject reference-state changes after transport before any callback."""
    class ChangeClock:
        iscollective = True

        def track(self, particles):
            if change == 'p0c':
                particles.update_p0c_and_energy_deviations(p0c=8e9)
            else:
                clock = particles._xobject
                setattr(clock, change, getattr(clock, change) + 0.25)

    events = []
    boundary = SharedFeedbackBoundary(
        ['beam'], lambda **event: events.append(event)
    )
    line = xt.Line(elements=[xt.Drift(length=10), ChangeClock()])
    boundary.install({'beam': line})
    line.build_tracker(_context=context)
    particles = xt.Particles(_context=context, p0c=7e9)
    multitracker = xt.PipelineMultiTracker([xt.PipelineBranch(line, particles)])
    boundary.bind(multitracker)

    with pytest.raises(error, match=message):
        boundary.track(num_turns=1)

    assert events == []
    with pytest.raises(RuntimeError, match='previously failed'):
        boundary.track(num_turns=1)


def test_lost_branch_rejected_instead_of_waiting_forever(context):
    """Reject an empty active branch before waiting for its arrival."""
    boundary, lines, particles, monitors, multitracker = build_study(
        context, lambda **kw: None, lost_branch='p2'
    )
    with pytest.raises(ValueError, match='no active particles'):
        boundary.bind(multitracker)


def test_callback_failure_is_not_retried(context):
    """A failed callback makes the coordinator unusable without reconstruction."""
    calls = []

    def callback(*, completed_turn, **kwargs):
        calls.append(completed_turn)
        raise RuntimeError('deliberate solver failure')

    boundary, lines, particles, monitors, multitracker = build_study(context, callback)
    boundary.bind(multitracker)
    with pytest.raises(RuntimeError, match='deliberate solver failure'):
        boundary.track(num_turns=2)
    with pytest.raises(RuntimeError, match='previously failed'):
        boundary.track(num_turns=1)
    assert calls == [1]
    assert boundary.callback_count == 0


@pytest.mark.parametrize('change', ['clock', 'position'])
def test_held_branch_is_revalidated_before_callback(context, change):
    """Certify each held beam when the last branch arrives at the barrier."""
    calls = []

    def insert_perturbation(lines, particles):
        held = particles['p1']

        class ChangeHeldBeam:
            iscollective = True

            def track(self, particles):
                assert held.s[0] == 1.0
                if change == 'clock':
                    held._xobject.time_s += 1e-9
                else:
                    held.s[0] = 0.0

        lines['p2'].append('change_held_beam', ChangeHeldBeam())

    boundary, _, _, _, multitracker = build_study(
        context, lambda **event: calls.append(event['completed_turn']),
        before_install=insert_perturbation,
    )
    boundary.bind(multitracker)

    with pytest.raises(ValueError, match='clock changed|full-turn boundary'):
        boundary.track(num_turns=1)

    assert calls == []
    assert boundary.callback_count == 0


def test_disabled_pipeline_holds_rejected_before_transport(context):
    """Reject a changed tracker setting while the beam and pickup are intact."""
    boundary, lines, particles, monitors, multitracker = build_study(
        context, lambda **event: None,
    )
    boundary.bind(multitracker)
    tracker = lines['p1'].tracker
    tracker.enable_pipeline_hold = False

    with pytest.raises(ValueError, match='pipeline holds'):
        boundary.track(num_turns=1)

    assert boundary.callback_count == 0
    np.testing.assert_array_equal(particles['p1'].s, 0)
    np.testing.assert_array_equal(monitors['p1'].num_particles, 0)
    tracker.enable_pipeline_hold = True
    boundary.track(num_turns=1)
    assert boundary.completed_turn == 1


def test_shared_environment_needs_common_machine_time_settings(context):
    """Reject branches that would hold different values of one native clock."""
    env = xt.Environment(particle_ref=xt.Particles(p0c=7e9))
    env.new('arc', xt.Drift, length=120.)
    lines = {name: env.new_line(components=['arc']) for name in ('a', 'b')}
    boundary = SharedFeedbackBoundary(tuple(lines))
    boundary.install(lines)
    branches = []
    buffer = None
    for line in lines.values():
        line.enable_time_dependent_vars = True
        line.build_tracker(_context=context, _buffer=buffer)
        buffer = line.tracker._buffer
        branches.append(xt.PipelineBranch(line, line.build_particles()))
    boundary.bind(xt.PipelineMultiTracker(branches))

    lines['a'].dt_update_time_dependent_vars = 1e-5
    with pytest.raises(ValueError, match='common machine-time settings'):
        boundary.track(num_turns=2)
    assert all(branch.particles.at_turn[0] == 0 for branch in branches)

    lines['b'].dt_update_time_dependent_vars = 1e-5
    boundary.track(num_turns=2)
    assert boundary.completed_turn == 2


def test_incompatible_time_epoch_rejected(context):
    """Reject equal reference turns at different physical epochs."""
    boundary, lines, particles, monitors, multitracker = build_study(
        context, lambda **kw: None
    )
    period = lines['p2'].get_length() / (float(particles['p2'].beta0[0]) * 299792458.0)
    particles['p2'].time_s = 0.1 * period
    particles['p2'].reference_turns = 0.0
    with pytest.raises(ValueError, match='incompatible time epoch'):
        boundary.bind(multitracker)


def test_nanosecond_epoch_offset_rejected_at_large_absolute_time(context):
    """Resolve a 10 ns mismatch at an epoch of one million seconds."""
    lengths = dict.fromkeys(('p1', 'p2', 'e1', 'e2'), 3000.0)
    boundary, _, particles, _, multitracker = build_study(
        context, lambda **event: None, epoch=1e6, length_by_branch=lengths,
    )
    particles['p2'].time_s += 10e-9
    for bunch in particles.values():
        bunch.reference_turns = 0.0

    with pytest.raises(ValueError, match='incompatible time epoch'):
        boundary.bind(multitracker)


def test_branch_clocks_use_their_native_rounded_periods(context):
    """Use one callback clock while each native clock advances its own period."""
    events = []
    boundary, _, particles, _, multitracker = build_study(
        context, lambda **event: events.append(event['time_seconds']),
        length_by_branch={'p2': 1.0 + 1e-14},
    )
    boundary.bind(multitracker)
    boundary.track(num_turns=4)
    boundary.track(num_turns=4)

    reference_period = particles['e1'].t_sim
    np.testing.assert_array_equal(events, np.arange(1, 9) * reference_period)
    assert particles['p2'].time_s > particles['e1'].time_s
    assert boundary.callback_count == 8


@pytest.mark.parametrize('after_bind', [False, True], ids=['bind', 'track'])
@pytest.mark.parametrize('change', ['hidden_start', 'moved_pickup'])
def test_invalid_geometry_or_start_is_rejected_before_transport(
    context, after_bind, change,
):
    """Protect the beam records from hidden starts and stale pickup positions."""
    boundary, lines, particles, monitors, multitracker = build_study(
        context, lambda **event: None, pairwise_exchanges=True,
    )
    if after_bind:
        boundary.bind(multitracker)
    line = lines['p2']
    beam = particles['p2']
    if change == 'hidden_start':
        beam.start_tracking_at_element = line.element_names.index('ip_b')
    else:
        line['first_half'].length += 0.125
        line['second_half'].length -= 0.125
        assert line.get_length() == 1.0

    message = 'start at the origin' if change == 'hidden_start' else 'line geometry changed'
    with pytest.raises(ValueError, match=message):
        if after_bind:
            boundary.track(num_turns=1)
        else:
            boundary.bind(multitracker)

    assert boundary.callback_count == 0
    for name, bunch in particles.items():
        np.testing.assert_array_equal(bunch.at_turn, 0)
        np.testing.assert_array_equal(monitors[name].num_particles, 0)


@pytest.mark.parametrize('aliased', [False, True], ids=['independent', 'aliased'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_particle_storage_ownership(test_context, aliased):
    """Distinguish overlapping xobjects from separate beams in one buffer."""
    boundary, _, particles, _, multitracker = build_study(
        test_context, lambda **event: None,
    )
    first = particles['p1']
    second = (xt.Particles(_xobject=first._xobject) if aliased
              else first.copy(_buffer=first._buffer))
    multitracker.branches[1].particles = second

    if aliased:
        with pytest.raises(ValueError, match='distinct particle storage'):
            boundary.bind(multitracker)
    else:
        boundary.bind(multitracker)
        boundary.track(num_turns=1)
        assert boundary.completed_turn == 1


def test_install_preflights_all_lines():
    """Keep lines untouched when one branch is still in compose mode."""
    boundary = SharedFeedbackBoundary(('first', 'second'), lambda **event: None)
    first = xt.Line(elements=[xt.Drift(length=1.0)], element_names=['arc'])
    env = xt.Environment()
    env.new('arc', xt.Drift, length=1.0)
    second = env.new_line(components=['arc'], compose=True)
    with pytest.raises(ValueError, match='end compose mode'):
        boundary.install({'first': first, 'second': second})
    assert first.element_names == ['arc']

    second.end_compose()
    boundary.install({'first': first, 'second': second})
    assert first.get(first.element_names[-1]) is boundary.elements['first']


def test_coarse_clock_epoch_rejected_before_shared_tracking(context):
    """Reject an epoch too coarse to resolve full-turn deadlines."""
    calls = []
    boundary, lines, particles, monitors, multitracker = build_study(
        context, lambda **kw: calls.append(kw['completed_turn']), epoch=1e9,
    )
    for bunch in particles.values():
        bunch.reference_turns = 0.0
    before = {name: bunch.y.copy() for name, bunch in particles.items()}
    with pytest.raises(ValueError, match='clock resolution'):
        boundary.bind(multitracker)
    assert not calls
    for name, bunch in particles.items():
        np.testing.assert_array_equal(bunch.y, before[name])


@pytest.mark.parametrize('pairwise_exchanges', [False, True])
def test_pipeline_feedback_system_pi_bulk_split_and_permuted_order(
    context, pairwise_exchanges
):
    """Preserve the measured kick and PI state across holds and chunking."""

    names = ('p1', 'p2', 'e1', 'e2')

    def pi(inputs, parameters, previous, event, xp):
        relative = -xp.atleast_1d(inputs['mean_y'])
        command = previous['command'] + (
            parameters['kp'] * (relative - previous['relative'])
            + parameters['ki_per_second'] * event['dt_seconds'] * relative
        )
        return {
            'memory': {'command': command, 'relative': relative},
            'outputs': {'command': command},
            'diagnostics': {'error_m': inputs['mean_y']},
        }

    def run(chunks, order):
        trace = []
        held_fields = None

        def before_feedback(
            *, completed_turn, time_seconds, particles_by_branch,
        ):
            nonlocal held_fields
            assert tuple(particles_by_branch) == names
            # These fields were held throughout the traversal that just ended.
            held_fields = np.asarray([float(lines[n]['kick'].ksl[0]) for n in names])
            py = np.asarray([float(particles_by_branch[n].py[0]) for n in names])
            before_kicker = np.asarray(
                [float(monitors[n].mean_py[completed_turn - 1]) for n in names]
            )
            actual_kicks = py - before_kicker
            np.testing.assert_allclose(
                actual_kicks, held_fields, rtol=1e-13, atol=1e-20
            )
        def after_feedback(*, completed_turn, time_seconds, updates):
            commands = np.asarray([float(lines[n]['feedback_rad']) for n in names])
            np.testing.assert_allclose(
                [float(lines[n]['kick'].ksl[0]) for n in names],
                commands,
                rtol=0.0,
                atol=0.0,
            )
            trace.append((completed_turn, time_seconds, held_fields, commands))

        boundary, lines, particles, monitors, multitracker = build_study(
            context,
            before_feedback,
            order=order,
            knob_outputs=True,
            pairwise_exchanges=pairwise_exchanges,
            after_feedback=after_feedback,
        )
        period = 1.0 / (float(particles['e1'].beta0[0]) * 299792458.0)
        controllers = [
            FeedbackController(
                name,
                sources={'mean_y': bind_monitor(monitors[name], 'mean_y')},
                program=pi,
                output=KnobOutput(lines[name], ['feedback_rad']),
                parameters={'kp': 0.3, 'ki_per_second': 0.02 / period},
                memory={'command': np.zeros(1), 'relative': np.zeros(1)},
            )
            for name in names
        ]
        system = FeedbackSystem(controllers, lines=lines)
        boundary.bind(multitracker, system=system)
        for turns in chunks:
            boundary.track(num_turns=turns)
        assert boundary.callback_count == sum(chunks)
        assert all(len(controller.log) == sum(chunks) for controller in controllers)
        if pairwise_exchanges:
            exchange = lines['p1'].get('ip_a').state
            assert exchange['holds'] > 0
            assert len(exchange['samples']) == 2 * len(names) * sum(chunks)
            assert len(exchange['applied']) == 2 * len(names) * sum(chunks)
        # The final callback exists, but its target has no earlier particle kick.
        np.testing.assert_array_equal(trace[0][2], np.zeros(4))
        for previous, current in zip(trace, trace[1:]):
            np.testing.assert_array_equal(current[2], previous[3])
        assert np.any(trace[-1][3] != trace[-1][2])
        state = {
            name: (
                particles[name].y.copy(),
                particles[name].py.copy(),
                controllers[i].memory['command'].copy(),
                controllers[i].memory['relative'].copy(),
            )
            for i, name in enumerate(names)
        }
        return trace, state

    full_trace, full_state = run([8], names)
    split_trace, split_state = run([1, 2, 1, 4], ('e2', 'p1', 'e1', 'p2'))
    for actual, expected in zip(split_trace, full_trace):
        assert actual[:2] == expected[:2]
        np.testing.assert_array_equal(actual[2], expected[2])
        np.testing.assert_array_equal(actual[3], expected[3])
    for name in names:
        for actual, expected in zip(split_state[name], full_state[name]):
            np.testing.assert_array_equal(actual, expected)


def test_shared_pickup_with_old_samples_is_rejected_before_tracking(context):
    """Reject a pilot's old data before any branch records the new bunch."""
    def callback(**event):
        pytest.fail('The stale record must prevent the first callback')

    boundary, lines, particles, monitors, multitracker = build_study(
        context, callback, knob_outputs=True,
    )
    line, bpm = lines['p1'], monitors['p1']
    pilot = particles['p1'].copy()
    pilot.y[:] = 9e-6
    bpm.track(pilot)
    controller = FeedbackController(
        'orbit', sources={'y': bind_monitor(bpm, 'mean_y')},
        program=lambda inputs: {'outputs': {'command': inputs['y']}},
        output=KnobOutput(line, 'feedback_rad'),
    )
    system = FeedbackSystem([controller], lines=[line])
    boundary.bind(multitracker, system=system)

    with pytest.raises(ValueError, match='already contains'):
        boundary.track(num_turns=2)

    assert boundary.callback_count == 0
    assert line['feedback_rad'] == 0.0
    assert not controller.log
    for name, bunch in particles.items():
        np.testing.assert_array_equal(bunch.at_turn, 0)
        np.testing.assert_array_equal(bunch.s, 0)
        if name != 'p1':
            np.testing.assert_array_equal(monitors[name].num_particles, 0)
    np.testing.assert_allclose(bpm.mean_y[0], 9e-6, rtol=0, atol=1e-20)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_shared_feedback_gain_changes_between_tracking_calls(test_context):
    """Keep the gain fixed while the shared boundary executes its callback."""
    line = xt.Line(elements={
        'ring': xt.Drift(length=120.),
        'bpm': xt.BeamStatsMonitor(stop_at_turn=2, stats=['mean_y']),
        'corrector': xt.Multipole(ksl=[0.]),
    })
    line['kick'] = 0.
    line.ref['corrector'].ksl[0] = line.ref['kick']

    def after_feedback(**event):
        with pytest.raises(RuntimeError, match='between tracking calls'):
            controller.set_parameter('gain', 99.)

    boundary = SharedFeedbackBoundary(['beam'], after_feedback=after_feedback)
    boundary.install({'beam': line})
    line.build_tracker(_context=test_context)
    particles = xt.Particles(_context=test_context, p0c=7e9, y=1e-3)
    controller = FeedbackController(
        'orbit', sources={'y': bind_monitor(line.get('bpm'), 'mean_y')},
        output=KnobOutput(line, 'kick'), parameters={'gain': 0.1},
        program=lambda inputs, parameters: {
            'outputs': {'command': parameters['gain'] * inputs['y']},
        },
    )
    system = FeedbackSystem([controller], lines=[line])
    multitracker = xt.PipelineMultiTracker([xt.PipelineBranch(line, particles)])
    boundary.bind(multitracker, system=system)

    boundary.track(num_turns=1)
    assert line['kick'] == pytest.approx(1e-4)
    controller.set_parameter('gain', 0.2)
    boundary.track(num_turns=1)
    assert line['kick'] == pytest.approx(2e-4)


@pytest.mark.parametrize('epoch', [0.0, 1e6])
@pytest.mark.parametrize(('offset', 'delivery'), [(0.0, 3), (100e-12, 4)])
def test_shared_delivery_turn_is_independent_of_epoch(
    context, epoch, offset, delivery,
):
    """Use elapsed seconds to distinguish a deadline 100 ps after a turn."""
    lengths = dict.fromkeys(('p1', 'p2', 'e1', 'e2'), 1000.0)
    boundary, lines, particles, monitors, multitracker = build_study(
        context, None, epoch=epoch, length_by_branch=lengths,
        knob_outputs=True,
    )
    for bunch in particles.values():
        bunch.reference_turns = 0.0
    bunch, line, bpm = particles['p1'], lines['p1'], monitors['p1']
    period = 1000.0 / (float(bunch.beta0[0]) * 299792458.0)
    controller = FeedbackController(
        'orbit', sources={'y': bind_monitor(bpm, 'mean_y')},
        program=lambda inputs: {'outputs': {'command': inputs['y']}},
        output=KnobOutput(line, 'feedback_rad'), update_turns=10,
        first_calculation_turn=1, command_delay_seconds=2 * period + offset,
    )
    system = FeedbackSystem([controller], lines=[line])
    boundary.bind(multitracker, system=system)
    boundary.track(num_turns=2)
    boundary.track(num_turns=3)

    first_applied = next(
        row for row in controller.log if row['applied_output'][0]
    )
    assert first_applied['completed_turn'] == delivery
    assert first_applied['time_seconds'] == epoch + delivery * period
    np.testing.assert_allclose(bunch.py, (5 - delivery) * 1e-6, rtol=1e-13)


def test_long_fixed_clock_delivers_one_second_delay_at_exact_boundary(context):
    """Deliver a one-second delay on the same turn in a bulk or split run."""

    def run(chunks):
        queue = CommandQueue()
        deliveries = []
        callback_times = []
        particles = xt.Particles(_context=context, p0c=7e12, y=[0.0, 0.0])
        period = 1.0 / 3300.0
        length = float(particles.beta0[0]) * 299792458.0 * period
        line = xt.Line(
            elements=[xt.Drift(length=length), xt.Multipole(ksl=[0.0])],
            element_names=['ring', 'kick'],
        )

        def callback(*, completed_turn, time_seconds, particles_by_branch):
            callback_times.append(time_seconds)
            if completed_turn == 40:
                queue.enqueue(
                    np.array([1e-9]),
                    time_seconds=time_seconds,
                    command_delay_seconds=1.0,
                )
            for command in queue.ready(time_seconds):
                line['kick'].ksl[0] = command.value[0]
                deliveries.append(
                    (completed_turn, time_seconds, command.ready_time_seconds)
                )

        boundary = SharedFeedbackBoundary(['beam'], callback).install({'beam': line})
        line.build_tracker(
            _context=context,
            use_prebuilt_kernels=False,
            track_kernel=getattr(context, '_feedback_pipeline_test_kernel', None),
        )
        multitracker = xt.PipelineMultiTracker(
            branches=[xt.PipelineBranch(line=line, particles=particles)]
        )
        boundary.bind(multitracker)
        for turns in chunks:
            boundary.track(num_turns=turns)
        np.testing.assert_array_equal(
            callback_times, np.arange(1, 3343) * boundary._period
        )
        assert [event[0] for event in deliveries] == [3340]
        # Delivery after turn 3340 affects the following two traversals only.
        np.testing.assert_allclose(particles.py, np.full(2, 2e-9), rtol=1e-14)
        return deliveries, particles.y.copy(), particles.py.copy(), callback_times

    bulk = run([3342])
    split = run([40, 1000, 2300, 2])
    assert bulk[0] == split[0]
    for actual, expected in zip(split[1:], bulk[1:]):
        np.testing.assert_array_equal(actual, expected)
