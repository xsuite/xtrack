# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from xobjects import ContextCpu
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)

import xtrack as xt
from xtrack.feedback import (
    Acquisition, FeedbackController, FeedbackSystem, KnobOutput,
)


def weighted_bpm(inputs, xp):
    """Divide integrated position moments by the corresponding weights."""
    weight = inputs['weight']
    valid = xp.all(weight > 0)
    mean_y = inputs['sum_y'] / xp.where(weight > 0, weight, 1.0)
    return {'mean_y': mean_y}, valid


def sample(
    acquisition,
    turn,
    weight=1.0,
    position=0.0,
    *,
    valid=True,
    ids=None,
    parameters=None,
    xp=np,
):
    """Submit one completed weighted measurement."""
    return acquisition.step(
        {
            'weight': xp.asarray(weight, dtype=float),
            'sum_y': xp.asarray(weight, dtype=float)
            * xp.asarray(position, dtype=float),
        },
        completed_turn=turn,
        time_seconds=turn * 0.001,
        valid=valid,
        record_ids=ids,
        parameters=parameters,
        xp=xp,
    )


def test_seven_record_boxcar_publishes_before_slow_controller_deadline():
    """Publish each full window independently of the controller cadence."""
    acquisition = Acquisition(window_samples=7, transform=weighted_bpm)
    publications, controller_measurements = [], []
    for turn in range(1, 81):
        candidate = sample(acquisition, turn, position=float(turn))
        if candidate.published:
            publications.append(turn)
        if turn % 40 == 0:
            assert candidate.valid
            controller_measurements.append(float(candidate.samples['mean_y']))
    assert publications == list(range(7, 81))
    assert_array_equal(controller_measurements, [37.0, 77.0])


def test_sampling_cadence_holds_readout_without_republishing_it():
    """Hold the latest sample between selected passages without appending it."""
    acquisition = Acquisition(
        window_samples=2, sample_turns=3, first_sample_turn=2,
        transform=weighted_bpm,
    )
    publications = []
    values = []
    for turn in range(1, 10):
        candidate = sample(acquisition, turn, position=float(turn))
        if candidate.published:
            publications.append(turn)
        if candidate.valid:
            values.append((turn, float(candidate.samples['mean_y'])))
    assert publications == [5, 8]
    assert values == [(5, 3.5), (6, 3.5), (7, 3.5), (8, 6.5), (9, 6.5)]
    assert candidate.first_completed_turn == 5
    assert candidate.last_completed_turn == 8
    assert candidate.sample_count == 2


# The reductions use the NumPy/CuPy stack and array arithmetic interface.
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_vector_bunch_bins_keep_separate_weights(test_context):
    """Take the weighted mean of each bunch independently on its context."""
    xp = test_context.nplike_lib
    acquisition = Acquisition(window_samples=2, transform=weighted_bpm)
    sample(acquisition, 1, weight=[1.0, 3.0], position=[1.0, 2.0], xp=xp)
    candidate = sample(
        acquisition, 2, weight=[9.0, 1.0], position=[3.0, 10.0], xp=xp)
    mean_y = candidate.samples['mean_y']
    assert isinstance(mean_y, test_context.nplike_array_type)
    assert_allclose(test_context.nparray_from_context_array(mean_y), [2.8, 4.0])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_unusable_selected_record_flushes_window_and_requires_refill(bad):
    """Discard a nonfinite selected measurement and wait for a full refill."""
    acquisition = Acquisition(window_samples=2, transform=weighted_bpm)
    sample(acquisition, 1, position=1.0)
    assert sample(acquisition, 2, position=2.0).published
    rejected = sample(acquisition, 3, position=bad)
    assert rejected.invalidated
    assert not rejected.valid and not rejected.published
    assert rejected.samples == {} and rejected.sample_count == 0
    assert not sample(acquisition, 4, position=4.0).published
    recovered = sample(acquisition, 5, position=5.0)
    assert recovered.published and recovered.samples['mean_y'] == 4.5
    assert recovered.first_completed_turn == 4


def test_missing_selected_record_is_detected_even_on_nonselected_boundary():
    """Invalidate the held sample as soon as a missed selection is known."""
    acquisition = Acquisition(sample_turns=2, transform=weighted_bpm)
    sample(acquisition, 1, position=1.0)
    candidate = sample(
        acquisition, 4, position=4.0
    )  # Selected turn 3 was never acquired.
    assert candidate.invalidated and not candidate.valid
    fresh = sample(acquisition, 5, position=5.0)
    assert fresh.published and fresh.samples['mean_y'] == 5.0


def test_replay_is_rejected_after_eviction_and_window_reset():
    """Keep rejecting old record indices after the original rows are gone."""
    acquisition = Acquisition(window_samples=2, transform=weighted_bpm)
    for turn in (1, 2, 3):
        sample(
            acquisition, turn, position=float(turn),
            ids={'weight': turn - 1, 'sum_y': turn - 1},
        )
    reused = sample(acquisition, 4, ids={'weight': 0, 'sum_y': 0})
    assert reused.invalidated
    assert not reused.published and reused.sample_count == 0
    older = sample(acquisition, 5, position=99.0, ids={'weight': 1, 'sum_y': 1})
    assert older.invalidated and older.sample_count == 0
    assert not sample(acquisition, 6, position=3.0).published
    assert sample(acquisition, 7, position=5.0).samples['mean_y'] == 4.0


@pytest.mark.parametrize('index', [-1, 1.0, True, 'previous'])
def test_record_indices_must_be_nonnegative_integers(index):
    """Reject record IDs that cannot order passages unambiguously."""
    acquisition = Acquisition()
    with pytest.raises(TypeError, match='nonnegative host integers'):
        acquisition.preview(
            {'position': 1.0}, completed_turn=1, time_seconds=0.001,
            record_ids={'position': index},
        )


def test_one_held_channel_invalidates_a_joint_measurement():
    """Reject a mixed window when just one channel repeats a record."""
    acquisition = Acquisition(transform=weighted_bpm)
    sample(acquisition, 1, ids={'weight': 0, 'sum_y': 0})
    candidate = sample(acquisition, 2, ids={'weight': 1, 'sum_y': 0})
    assert candidate.invalidated and not candidate.valid


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_preview_failure_and_discard_preserve_state(test_context):
    """Isolate callback configuration and retry without advancing the window."""
    xp = test_context.nplike_lib
    to_host = test_context.nparray_from_context_array
    gain = xp.asarray([2.0])
    scale = xp.asarray([3.0])

    def transform(inputs, parameters, constants, xp):
        if parameters['fail']:
            for array in (parameters['gain'], constants['scale']):
                if xp is np:
                    with pytest.raises(ValueError, match='read-only'):
                        array[:] = -1.0
                else:
                    array[:] = -1.0
            raise RuntimeError('deliberate transform failure')
        values, valid = weighted_bpm(inputs, xp)
        values['mean_y'] = (
            values['mean_y'] * parameters['gain'][0] * constants['scale'][0])
        return values, valid

    acquisition = Acquisition(window_samples=2, transform=transform)
    sample(acquisition, 1, position=1.0, xp=xp)
    inputs = {'weight': xp.asarray(1.0), 'sum_y': xp.asarray(2.0)}
    setup = dict(
        completed_turn=2, time_seconds=0.002, constants={'scale': scale}, xp=xp)
    with pytest.raises(RuntimeError, match='deliberate'):
        acquisition.preview(inputs, parameters={'fail': True, 'gain': gain},
                            **setup)
    assert_array_equal(to_host(gain), [2.0])
    assert_array_equal(to_host(scale), [3.0])
    if xp is np:
        assert gain.flags.writeable and scale.flags.writeable

    parameters = {'fail': False, 'gain': gain}
    discarded = acquisition.preview(inputs, parameters=parameters, **setup)
    assert discarded.published
    accepted = acquisition.preview(inputs, parameters=parameters, **setup)
    assert accepted.sample_count == 2
    assert_allclose(to_host(accepted.samples['mean_y']), 9.0)
    acquisition.commit(accepted)
    with pytest.raises(ValueError, match='Stale'):
        acquisition.commit(discarded)


@pytest.mark.parametrize('nested_commit', [False, True], ids=['step', 'commit'])
def test_transform_cannot_reenter_acquisition(nested_commit):
    """Keep a callback from committing live state during another preview."""
    def transform(inputs, parameters):
        if parameters['reenter']:
            if nested_commit:
                acquisition.commit(pending)
            else:
                acquisition.step(
                    {'x': 2.0}, completed_turn=2, time_seconds=0.002,
                    parameters={'reenter': False},
                )
        return inputs, True

    acquisition = Acquisition(transform=transform)
    step = dict(completed_turn=1, time_seconds=0.001)
    pending = acquisition.preview({'x': 1.0}, parameters={'reenter': False},
                                  **step)
    with pytest.raises(RuntimeError, match='reenter'):
        acquisition.step({'x': 3.0}, parameters={'reenter': True}, **step)

    acquisition.commit(pending)
    assert pending.samples['x'] == 1.0
    result = acquisition.step(
        {'x': 2.0}, completed_turn=2, time_seconds=0.002,
        parameters={'reenter': False},
    )
    assert result.samples['x'] == 2.0


def test_commit_rejects_changed_acquisition_configuration():
    """Keep a pending sample from committing under a different cadence."""
    acquisition = Acquisition()
    candidate = acquisition.preview(
        {'x': 1.0}, completed_turn=1, time_seconds=0.001)
    acquisition.sample_turns = 2
    with pytest.raises(RuntimeError, match='configuration changed'):
        acquisition.commit(candidate)


def test_bound_sources_must_share_a_tracking_context():
    """Reject a source map spanning distinct tracking contexts at binding."""
    sources = {
        name: (xt.BeamStatsMonitor(
            _context=ContextCpu(), stop_at_turn=2, stats=['mean_y']), 'mean_y')
        for name in ('first', 'second')
    }
    with pytest.raises(ValueError, match='share one context'):
        Acquisition(sources=sources)


def test_copy_isolation_and_pause_resume_preserve_window_phase():
    """Keep original and copied windows independent at the same cadence."""
    acquisition = Acquisition(
        window_samples=2, sample_turns=2, transform=weighted_bpm)
    sample(acquisition, 1, position=1.0)
    branch = acquisition.copy()
    changed = sample(branch, 3, position=30.0)
    original = sample(acquisition, 3, position=3.0)
    assert changed.samples['mean_y'] == 15.5
    assert original.samples['mean_y'] == 2.0
    resumed = acquisition.copy()
    held = sample(resumed, 4, position=4.0)
    assert held.valid and not held.published
    assert held.samples['mean_y'] == 2.0
    assert sample(resumed, 5, position=5.0).samples['mean_y'] == 4.0


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_samples_and_input_snapshots_do_not_alias_retained_data(test_context):
    """Keep published data independent of both caller input and preview."""
    xp = test_context.nplike_lib
    acquisition = Acquisition(sample_turns=2)
    original = xp.array([1.0, 2.0])
    candidate = acquisition.preview(
        {'signal': original}, completed_turn=1, time_seconds=0.001, xp=xp,
    )
    original[:] = -1.0
    candidate.samples['signal'][:] = 99.0
    acquisition.commit(candidate)
    held = acquisition.preview(
        {'signal': original}, completed_turn=2, time_seconds=0.002, xp=xp,
    )
    to_host = test_context.nparray_from_context_array
    held_signal = to_host(held.samples['signal'])
    assert_array_equal(held_signal, [1., 2.])


def test_source_and_sample_schema_survive_invalid_gaps():
    """Retain the input and publication schemas across an invalid record."""
    acquisition = Acquisition(transform=weighted_bpm)
    sample(acquisition, 1, position=1.0)
    sample(acquisition, 2, position=2.0, valid=False)
    with pytest.raises(ValueError, match='source names, shapes or dtypes'):
        sample(acquisition, 3, position=[3.0, 4.0], weight=[1.0, 1.0])

    def variable(inputs, parameters):
        return {'signal': np.zeros(parameters['size'])}, parameters['valid']

    acquisition = Acquisition(transform=variable)
    acquisition.step(
        {'x': 1.0},
        completed_turn=1,
        time_seconds=0.001,
        parameters={'size': 1, 'valid': True},
    )
    acquisition.step(
        {'x': 1.0},
        completed_turn=2,
        time_seconds=0.002,
        parameters={'size': 1, 'valid': False},
    )
    with pytest.raises(ValueError, match='sample names, shapes or dtypes'):
        acquisition.preview(
            {'x': 1.0},
            completed_turn=3,
            time_seconds=0.003,
            parameters={'size': 2, 'valid': True},
        )


@pytest.mark.parametrize(
    'valid', [1.0, 0, np.nan, None, np.array([], dtype=bool)],
)
def test_validity_is_a_boolean_predicate_not_a_truthy_number(valid):
    """Reject invalid predicates before and after a transform."""
    acquisition = Acquisition()
    with pytest.raises(TypeError, match='Boolean'):
        acquisition.preview(
            {'x': 1.0}, completed_turn=1, time_seconds=0.001, valid=valid
        )
    acquisition = Acquisition(transform=lambda inputs: (inputs, valid))
    with pytest.raises(TypeError, match='Boolean'):
        acquisition.preview({'x': 1.0}, completed_turn=1, time_seconds=0.001)


def track_monitor(test_context):
    """Record two traversals with different particle weights and offsets."""
    line = xt.Line(
        elements=[
            xt.BeamStatsMonitor(
                _context=test_context, stop_at_turn=2, stats=['mean_y']),
            xt.Drift(length=1.0),
        ],
        element_names=['bpm', 'drift'],
    )
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    particles = xt.Particles(
        _context=test_context, p0c=45.6e9, y=[1e-6], weight=[1.0],
    )
    line.track(particles, num_turns=1)
    particles.y[:] = 3e-6
    particles.weight[:] = 3.0
    line.track(particles, num_turns=1)
    return line


def raw_at(acquisition, turn):
    """Read the selected completed passage from each bound channel."""
    return {
        name: source.read_record(turn, completed_turn=turn + 1).value
        for name, source in acquisition.sources.items()
    }


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_monitor_reduces_moments_before_taking_mean(test_context, sandbox_cwd):
    """Compare a weighted centroid with a sum of per-turn centroids."""
    tracked = track_monitor(test_context)
    to_host = test_context.nparray_from_context_array
    acquisition = Acquisition.from_monitor(
        tracked.get('bpm'),
        {'weight': 'num_particles', 'sum_y': 'sum_y'},
        window_samples=2,
        transform=weighted_bpm,
    )
    first = acquisition.step(
        raw_at(acquisition, 0), completed_turn=1, time_seconds=0.1)
    assert not first.published
    second = acquisition.step(
        raw_at(acquisition, 1), completed_turn=2, time_seconds=0.2,
    )
    assert second.published
    assert_allclose(to_host(second.samples['mean_y']), 2.5e-6, rtol=1e-14)

    declared_sum = Acquisition.from_monitor(
        tracked.get('bpm'), {'y': 'mean_y'}, window_samples=2, reduction='sum',
    )
    declared_sum.step(
        raw_at(declared_sum, 0), completed_turn=1, time_seconds=0.1)
    result = declared_sum.step(
        raw_at(declared_sum, 1), completed_turn=2, time_seconds=0.2,
    )
    assert_allclose(to_host(result.samples['y']), 4e-6, rtol=1e-14)


@pytest.mark.parametrize('custom_drive', [False, True])
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_controller_uses_bound_monitor_with_optional_custom_drive(
    test_context, sandbox_cwd, custom_drive,
):
    """Pair a custom drive with the pickup from the same traversal."""
    tracked = track_monitor(test_context)
    xp = test_context.nplike_lib
    acquisition = Acquisition.from_monitor(tracked.get('bpm'), {'y': 'mean_y'})
    explicit = {}
    if custom_drive:

        def read_drive(turn, *, completed_turn):
            assert completed_turn == turn + 1
            return SimpleNamespace(
                value=xp.asarray(7.0 + turn), valid=True, record_index=turn
            )

        explicit['drive'] = SimpleNamespace(
            context=test_context, read_record=read_drive
        )

    def program(inputs):
        command = inputs['y'] + inputs.get('drive', 0.0) * 1e-6
        return {'outputs': {'command': command}}

    tracked['command'] = 0.0
    controller = FeedbackController(
        'bound_monitor',
        acquisition=acquisition,
        sources=explicit,
        program=program,
        output=KnobOutput(tracked, 'command'),
    )
    system = FeedbackSystem([controller], lines=[tracked])
    system.step(1, 0.1)
    assert tracked['command'] == pytest.approx(8e-6 if custom_drive else 1e-6)
    system.step(2, 0.2)
    assert tracked['command'] == pytest.approx(11e-6 if custom_drive else 3e-6)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_controller_delivers_between_strided_monitor_records(test_context, sandbox_cwd):
    """Read even completed turns and deliver one turn later between records."""
    bpm = xt.BeamStatsMonitor(
        _context=test_context, start_at_turn=1, stop_at_turn=6,
        every_n_turns=2, stats=['mean_y'],
    )
    line = xt.Line(elements=[bpm, xt.Drift(length=1.0)])
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    particles = xt.Particles(_context=test_context, p0c=45.6e9)
    for turn in range(1, 7):
        particles.y[:] = turn * 1e-6
        line.track(particles, num_turns=1)
    line['trim'] = 0.0
    output = KnobOutput(line, 'trim')

    def program(inputs):
        return {'outputs': {'command': -inputs['y']}}

    misaligned = Acquisition.from_monitor(
        bpm, {'y': 'mean_y'}, sample_turns=2, first_sample_turn=1,
    )
    with pytest.raises(ValueError, match='cadence must align'):
        FeedbackController('orbit', acquisition=misaligned,
                           program=program, output=output)

    aligned = Acquisition.from_monitor(
        bpm, {'y': 'mean_y'}, sample_turns=2, first_sample_turn=2,
    )
    controller = FeedbackController(
        'orbit', acquisition=aligned, program=program, output=output,
        update_turns=2, command_delay_seconds=0.5,
    )
    system = FeedbackSystem([controller], lines=[line])
    applied, published = [], []
    for turn in range(1, 7):
        entry = system.step(turn, float(turn))[0]
        applied.append(line['trim'])
        published.append(entry['acquisition_published'])
    assert_allclose(applied, [0, 0, -2e-6, -2e-6, -4e-6, -4e-6], atol=1e-18)
    assert published == [False, True, False, True, False, True]


def test_missing_ignored_turns_preserves_selected_cadence():
    """Keep the window intact when only unsampled turns are skipped."""
    acquisition = Acquisition(
        window_samples=2, sample_turns=2, transform=weighted_bpm)
    sample(acquisition, 1, position=1.0)
    candidate = sample(acquisition, 3, position=3.0)
    assert candidate.published and not candidate.invalidated
    assert candidate.samples['mean_y'] == 2.0


def test_transform_receives_owned_readonly_reduction_and_window_metadata():
    """Expose completion metadata and protect the reduced inputs."""
    def transform(inputs, constants, event, xp):
        assert isinstance(inputs, MappingProxyType)
        assert event['first_completed_turn'] == 1
        assert event['last_completed_turn'] == 2
        assert event['sample_count'] == 2
        assert event['record_ids']['weight'] == (0, 1)
        with pytest.raises(ValueError):
            inputs['sum_y'][...] = 100.0
        signal = xp.asarray(inputs['sum_y'] * constants['scale'])
        return {'signal': signal}, True

    acquisition = Acquisition(window_samples=2, transform=transform)
    for turn in [1, 2]:
        candidate = acquisition.step(
            {'weight': 1.0, 'sum_y': float(turn)},
            completed_turn=turn,
            time_seconds=turn * 0.001,
            constants={'scale': 2.0},
        )
    assert candidate.samples['signal'] == 6.0
