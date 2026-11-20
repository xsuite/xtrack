# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)

import xtrack as xt
from xtrack.feedback.readout import (
    IncompleteRecordError,
    bind_monitor,
    SourceInvalidatedError,
)


# The readout uses NumPy-style reshaping and where on context arrays.
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_readout_keeps_weighted_moments_on_context(
    test_context, tmp_path, monkeypatch,
):
    """Read tracked weighted sums and means as owned context arrays."""
    monkeypatch.chdir(tmp_path)
    to_host = test_context.nparray_from_context_array
    monitor = xt.BeamStatsMonitor(
        stop_at_turn=3, stats=['mean_y', 'mean_pzeta'])
    line = xt.Line(
        elements=[monitor, xt.Drift(length=1.0)],
        element_names=['bpm', 'drift'],
        particle_ref=xt.Particles(p0c=7e12),
    )
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    weights = [2.0, 3.0]
    particles = xt.Particles(
        _context=test_context, p0c=7e12, y=[1e-3, 3e-3], weight=weights,
        delta=[1e-4, -2e-4],
    )
    line.track(particles, num_turns=2)

    monkeypatch.setattr(
        monitor, 'get', lambda *args, **kwargs: pytest.fail('Host getter used'))
    count = bind_monitor(monitor, 'num_particles')
    total = bind_monitor(monitor, 'sum_y')
    mean = bind_monitor(line['bpm'], 'mean_y')
    assert mean.monitor is monitor
    assert mean.context is test_context
    assert mean.quantity == 'mean_y'
    assert mean.shape == ()
    with pytest.raises(AttributeError):
        mean.quantity = 'sum_y'

    value = mean.read(0, completed_turn=2)
    assert isinstance(value, test_context.nplike_array_type)
    assert_allclose(to_host(value), 0.0022)
    assert_allclose(to_host(count.read(0, completed_turn=2)), 5.0)
    assert_allclose(to_host(total.read(0, completed_turn=2)), 0.011)
    pzeta = bind_monitor(monitor, 'sum_pzeta')
    assert_allclose(to_host(pzeta.read(1, completed_turn=2)),
                    np.dot(weights, to_host(particles.pzeta)))

    record = mean.read_record(1, completed_turn=2)
    for array in (record.value, record.weight, record.touched, record.valid):
        assert isinstance(array, test_context.nplike_array_type)
    assert bool(to_host(record.valid))
    assert int(to_host(record.touched)) == 1
    assert record.turn == 1 and record.record_index == 1

    monitor.clear()
    assert_allclose(to_host(record.value), 0.0022)
    assert_allclose(to_host(record.weight), 5.0)
    with pytest.raises(SourceInvalidatedError, match='bind a new'):
        mean.read(0, completed_turn=2)


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_completion_watermark_and_empty_record_are_distinct(test_context):
    """Require completion even for a passage that had no particles."""
    to_host = test_context.nparray_from_context_array
    monitor = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=2, stats=['mean_y'])
    source = bind_monitor(monitor, 'mean_y')
    source.validate_binding()
    with pytest.raises(IncompleteRecordError, match='has not completed'):
        source.read(0, completed_turn=0)
    record = source.read_record(0, completed_turn=1)
    assert np.isnan(to_host(record.value))
    assert float(to_host(record.weight)) == 0.0
    assert not bool(to_host(record.valid))
    assert not bool(to_host(record.touched))


def test_frame_and_stride_validation():
    """Enforce frame limits and stride, then invalidate a retargeted frame."""
    monitor = xt.BeamStatsMonitor(
        start_at_turn=3, stop_at_turn=9, every_n_turns=2, stats=['mean_y']
    )
    source = bind_monitor(monitor, 'sum_y')
    with pytest.raises(IndexError, match='outside monitor frame'):
        source.read(2, completed_turn=3)
    with pytest.raises(IndexError, match='not recorded by stride'):
        source.read(4, completed_turn=5)
    with pytest.raises(IndexError, match='outside monitor frame'):
        source.read(9, completed_turn=10)
    with pytest.raises(TypeError, match='host integer'):
        source.read(0.5, completed_turn=1)
    with pytest.raises(TypeError, match='host integer'):
        source.read(3, completed_turn=True)
    assert source.read_record(5, completed_turn=6).record_index == 1
    monitor.start_new_frame(9)
    with pytest.raises(SourceInvalidatedError, match='bind a new'):
        source.read(5, completed_turn=6)


def test_source_validation_rejects_unallocated_and_unsupported_data():
    """Reject bindings to missing quantities, slots or acquisition modes."""
    monitor = xt.BeamStatsMonitor(stop_at_turn=2, stats=['mean_x'])
    with pytest.raises(ValueError, match='was not allocated'):
        bind_monitor(monitor, 'sum_y')
    with pytest.raises(ValueError, match='Unsupported'):
        bind_monitor(monitor, 'sigma_x')
    with pytest.raises(ValueError, match='not available'):
        bind_monitor(monitor, 'mean_x', bunch_slot=0)
    with pytest.raises(TypeError, match='Unsupported monitor'):
        bind_monitor(object(), 'num_particles')
    bunch = xt.BeamStatsMonitor(
        stop_at_turn=2, stats=['mean_y'], bunch_spacing_zeta=5.0,
        filled_slots=[0, 2],
    )
    with pytest.raises(ValueError, match='not recorded'):
        bind_monitor(bunch, 'mean_y', bunch_slot=1)
    with pytest.raises(TypeError, match='integer slot'):
        bind_monitor(bunch, 'mean_y', bunch_slot=True)
    coasting = xt.BeamStatsMonitor(
        stop_at_turn=2, stats=['mean_y'], coasting=True, num_slices=2
    )
    with pytest.raises(NotImplementedError, match='Coasting'):
        bind_monitor(coasting, 'mean_y')


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_slice_records_preserve_physical_slots_and_empty_bins(test_context):
    """Select slices by physical slot and flag empty bins independently."""
    to_host = test_context.nparray_from_context_array
    to_context = test_context.nparray_to_context_array
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        stop_at_turn=2,
        stats=['mean_y'],
        zeta_range=(-1.0, 1.0),
        num_slices=2,
        bunch_spacing_zeta=5.0,
        filled_slots=[0, 2],
        selected_slots=[2, 0],
    )
    monitor.data.num_particles[:4] = to_context(np.array([2., 4., 3., 0.]))
    monitor.data.sum_y[:4] = to_context(np.array([12., 4., 30., 0.]) * 1e-6)
    full = bind_monitor(monitor, 'mean_y').read_record(0, completed_turn=1)
    assert_allclose(to_host(full.value), [[6e-6, 1e-6], [10e-6, np.nan]])
    assert_array_equal(to_host(full.valid), [[True, True], [True, False]])

    slot = bind_monitor(monitor, 'mean_y', bunch_slot=2)
    assert slot.bunch_slot == 2
    with pytest.raises(AttributeError):
        slot.bunch_slot = 0
    assert_allclose(to_host(slot.read(0, completed_turn=1)), [6e-6, 1e-6])


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_buffer_growth_invalidates_cached_array_views(test_context):
    """Require rebinding when the buffer replaces its backing array."""
    to_host = test_context.nparray_from_context_array
    monitor = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=2, stats=['mean_y'])
    monitor.data.num_particles[:] = 2.0
    monitor.data.sum_y[:] = 4e-6
    source = bind_monitor(monitor, 'mean_y')
    source.validate_binding()
    assert_allclose(to_host(source.read(0, completed_turn=1)), 2e-6)
    # Growth preserves the Buffer object but replaces its raw allocation.
    monitor._buffer.grow(monitor._buffer.capacity * 2)
    monitor.data.sum_y[:] = 20e-6
    with pytest.raises(SourceInvalidatedError, match='bind a new'):
        source.validate_binding()
    with pytest.raises(SourceInvalidatedError, match='bind a new'):
        source.read(0, completed_turn=1)
    rebound = bind_monitor(monitor, 'mean_y')
    assert_allclose(to_host(rebound.read(0, completed_turn=1)), 1e-5)


@pytest.mark.parametrize('same_xobject', [True, False])
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_alias_clear_invalidates_the_physical_monitor_frame(
    test_context, same_xobject,
):
    """Bind through either wrapper and invalidate both views on clear."""
    to_host = test_context.nparray_from_context_array
    monitor = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=2, stats=['mean_y'])
    monitor.data.num_particles[:] = 2.0
    monitor.data.sum_y[:] = 4e-6
    xobject = monitor._xobject
    if not same_xobject:
        xobject = type(xobject)._from_buffer(monitor._buffer, monitor._offset)
        assert xobject is not monitor._xobject
    alias = xt.BeamStatsMonitor(_xobject=xobject)
    source = bind_monitor(monitor, 'mean_y')
    aliased = bind_monitor(alias, 'mean_y')
    assert alias is not monitor and alias._buffer is monitor._buffer
    assert_allclose(to_host(aliased.read(0, completed_turn=1)), 2e-6)

    alias.clear()
    with pytest.raises(SourceInvalidatedError, match='bind a new'):
        source.validate_binding()
    with pytest.raises(SourceInvalidatedError, match='bind a new'):
        aliased.read(0, completed_turn=1)
    monitor.data.num_particles[:] = 3.0
    monitor.data.sum_y[:] = 9e-6
    rebound = bind_monitor(monitor, 'mean_y')
    assert_allclose(to_host(rebound.read(0, completed_turn=1)), 3e-6)


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_window_checks_only_aligned_acquisition_rows(test_context):
    """Detect a previously recorded sample on a decimated pickup grid."""
    monitor = xt.BeamStatsMonitor(
        _context=test_context, stop_at_turn=8, every_n_turns=2,
        stats=['mean_y'],
    )
    source = bind_monitor(monitor, 'mean_y')
    monitor.data.num_particles[1] = 2.0  # Native row for passage turn 2.

    source.validate_window(0, 8, sample_turns=4)
    with pytest.raises(ValueError, match='already contains'):
        source.validate_window(2, 8, sample_turns=4)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_particles_source_masks_unwritten_ids(test_context, sandbox_cwd):
    """Distinguish a recorded zero coordinate from an unwritten particle ID."""
    monitor = xt.ParticlesMonitor(
        _context=test_context, start_at_turn=0, stop_at_turn=2,
        particle_id_range=(1, 4),
    )
    line = xt.Line(elements=[monitor, xt.Drift(length=1.0)])
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    source = bind_monitor(monitor, 'y')
    particles = xt.Particles(
        _context=test_context, p0c=7e12, y=[0.0, 3e-6], particle_id=[1, 3],
    )

    source.validate_window(0, 2)
    line.track(particles, num_turns=1)
    record = source.read_record(0, completed_turn=1)
    to_host = test_context.nparray_from_context_array
    assert isinstance(record.value, test_context.nplike_array_type)
    assert_array_equal(to_host(record.value), [0.0, 0.0, 3e-6])
    assert_array_equal(to_host(record.valid), [True, False, True])
    with pytest.raises(IncompleteRecordError, match='has not completed'):
        source.read_record(1, completed_turn=1)
    with pytest.raises(ValueError, match='already contains'):
        source.validate_window(0, 1)
    source.validate_window(1, 2)


def test_particle_source_rejects_incompatible_selectors():
    """Require a coordinate and a single full-turn frame."""
    monitor = xt.ParticlesMonitor(
        start_at_turn=0, stop_at_turn=2, num_particles=1,
    )
    with pytest.raises(ValueError, match='Unsupported readout quantity'):
        bind_monitor(monitor, 'mean_y')
    with pytest.raises(ValueError, match='particle IDs'):
        bind_monitor(monitor, 'y', bunch_slot=1)
    repeated = xt.ParticlesMonitor(
        start_at_turn=0, stop_at_turn=2, num_particles=1,
        n_repetitions=2, repetition_period=4,
    )
    with pytest.raises(NotImplementedError, match='single full-turn'):
        bind_monitor(repeated, 'y')


@pytest.mark.parametrize(
    'beam_stats', [True, False], ids=['stats', 'particles'],
)
def test_readout_checks_frame_with_one_buffer_read(mocker, beam_stats):
    """Check the native frame in one read and reject a changed turn range."""
    if beam_stats:
        monitor = xt.BeamStatsMonitor(stop_at_turn=2, stats=['mean_y'])
        source = bind_monitor(monitor, 'mean_y')
    else:
        monitor = xt.ParticlesMonitor(
            start_at_turn=0, stop_at_turn=2, num_particles=2)
        source = bind_monitor(monitor, 'y')

    buffer_read = mocker.spy(monitor._buffer, 'to_bytearray')
    source.read_record(0, completed_turn=1)
    buffer_read.assert_called_once()
    monitor.stop_at_turn = 3
    with pytest.raises(SourceInvalidatedError, match='bind a new source'):
        source.read_record(0, completed_turn=1)
