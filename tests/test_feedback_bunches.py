# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Physical slot identity and arrival eligibility for sparse bunch feedback."""

import numpy as np
import pytest
from scipy.constants import c
import xtrack as xt
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)

from xtrack.feedback.readout import bind_monitor

pytestmark = pytest.mark.usefixtures('sandbox_cwd')


# Placed feedback uses NumPy and CuPy for numerical calculations.
@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_physical_slots_survive_empty_bunch_and_particle_reorganization(
    test_context,
):
    """Keep physical slot identity when a bunch loses all its particles."""
    circumference = 80.0
    slots = np.array([5, 2, 0, 1, 5, 0, 2])
    weights = np.array([1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0])
    particles = xt.Particles(
        _context=test_context,
        p0c=7e9,
        zeta=-(slots + 0.25) * circumference / 8,
        weight=weights,
        y=np.zeros(len(slots)),
        py=np.zeros(len(slots)),
    )
    filling = np.zeros(8, dtype=int)
    filling[[0, 2, 5]] = 1
    selected = [5, 0, 2]
    monitor = xt.BeamStatsMonitor(
        _context=test_context,
        start_at_turn=0,
        stop_at_turn=2,
        bunch_spacing_zeta=circumference / 8,
        filling_pattern=filling,
        selected_slots=selected,
        stats=['num_particles', 'mean_py'],
    )
    waveform = np.zeros(8)
    waveform[[0, 2, 5]] = [1e-6, -2e-6, 3e-6]
    frev = float(particles.beta0[0]) * c / circumference
    exciter = xt.Exciter(
        _context=test_context,
        samples=waveform,
        ksl=[1.0],
        sampling_frequency=8 * frev,
        frev=frev,
        duration=4 / frev,
    )
    line = xt.Line(
        elements=[exciter, monitor, xt.LineSegmentMap(length=circumference)],
        element_names=['kicker', 'bpm', 'ring'],
    )
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    source = bind_monitor(line.get('bpm'), 'mean_py')
    to_host = test_context.nparray_from_context_array
    line.track(particles, num_turns=1)
    np.testing.assert_array_equal(line.get('bpm').selected_slots, selected)
    first = source.read_record(0, completed_turn=1)
    np.testing.assert_allclose(
        to_host(first.value), waveform[selected], rtol=1e-7,
    )
    np.testing.assert_array_equal(to_host(first.weight), [6.0, 9.0, 9.0])

    lost = (particles.particle_id == 1) | (particles.particle_id == 6)
    particles.state[lost] = -1
    particles.reorganize()
    line.track(particles, num_turns=1)
    second = source.read_record(1, completed_turn=2)
    np.testing.assert_array_equal(to_host(second.valid), [True, True, False])
    np.testing.assert_array_equal(to_host(second.weight), [6.0, 9.0, 0.0])
    recorded = to_host(second.value)
    np.testing.assert_allclose(
        recorded[:2], 2 * waveform[selected[:2]], rtol=1e-7,
    )
    assert np.isnan(recorded[2])
    active = to_host(particles.state) > 0
    ids = to_host(particles.particle_id)
    np.testing.assert_allclose(
        to_host(particles.py)[active], 2 * waveform[slots[ids[active]]],
        rtol=1e-7,
    )


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_bunch_feedback_kicks_only_the_selected_physical_slots(test_context):
    """Apply only the two selected kicks, on the first eligible passage."""
    length, slot_count = 80.0, 8
    selected = [5, 2]
    physical_slots = np.array([2, 5, 2, 5, 0])
    y = np.array([1.0, 4.0, 3.0, 2.0, 9.0]) * 1e-6
    beam = xt.Particles(
        _context=test_context, p0c=7e9, y=y, weight=[1, 3, 3, 1, 2],
        zeta=-(physical_slots + 0.25) * length / slot_count,
    )
    frev = float(beam.beta0[0]) * c / length
    pickup = xt.BeamStatsMonitor(
        _context=test_context,
        stop_at_turn=4, bunch_spacing_zeta=length / slot_count,
        filled_slots=[0, 2, 5], selected_slots=selected, stats=['mean_y'],
    )
    corrector = xt.Exciter(
        samples=np.zeros(slot_count), ksl=[1.0], frev=frev,
        sampling_frequency=slot_count * frev, duration=4 / frev,
    )
    fb = xt.feedback.Feedback(
        name='bunch_orbit', update_turns=10, first_calculation_turn=1,
        command_delay_seconds=1.5 / frev,
    )
    fb.inputs.new(name='y', monitor='pickup', quantity='mean_y')
    fb.outputs.new(name='kick', target=['slot_5', 'slot_2'])
    fb.program = 'outputs.kick = -acquired.y'
    elements = {
        'pickup': pickup, 'kicker': corrector,
        'ring': xt.LineSegmentMap(length=length), 'feedback': fb,
    }
    line = xt.Line(elements=elements, element_names=list(elements))
    for slot in selected:
        key = f'slot_{slot}'
        line[key] = 0.0
        line.ref['kicker'].samples[slot] = line.ref[key]

    line.build_tracker(_context=test_context)
    line.feedback.enable()
    to_host = test_context.nparray_from_context_array
    line.track(beam, num_turns=3)
    np.testing.assert_array_equal(to_host(beam.py), 0)
    line.track(beam, num_turns=1)

    # The first eligible traversal is turn index 3. Filled slot 0 was
    # deliberately omitted from the pickup and the output channel list.
    expected = np.zeros(slot_count)
    expected[5], expected[2] = -3.5e-6, -2.5e-6
    np.testing.assert_allclose(
        to_host(line.get('kicker').samples), expected, rtol=1e-7,
    )
    np.testing.assert_allclose(
        to_host(beam.py), expected[physical_slots], rtol=1e-7,
    )
