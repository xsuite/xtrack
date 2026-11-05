import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_line_segment_map_on_absolute_time_with_detuned_rf_reaches_the_analytic_momentum(test_context):
    # dp/p = -df / (f eta) for RF detuned from a harmonic of the revolution.
    alpha = 1e-3
    length = 628.
    detuning = 20.
    reference = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
    frequency = 8 * reference.beta0[0] * clight / length
    expected = -detuning / frequency / (alpha - 1 / reference.gamma0[0]**2)
    line = xt.Line(elements=[xt.LineSegmentMap(
        length=length, qx=.1, qy=.2, momentum_compaction_factor=alpha, longitudinal_mode='nonlinear',
        voltage_rf=[1e5], frequency_rf=[frequency + detuning], phase_rf=[0.], absolute_time=True)])
    line.particle_ref = reference
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.], delta=[expected])

    line.track(particles, num_turns=4000, turn_by_turn_monitor=True)

    delta = line._context.nparray_from_context_array(line.record_last_track.delta)[0]
    xo.assert_allclose(delta.mean(), expected, rtol=.01, atol=0)


@for_all_test_contexts
def test_line_segment_map_on_absolute_time_reads_the_clock_at_the_middle_of_the_segment(test_context):
    ctx2np = test_context.nparray_from_context_array
    length = 10.
    frequency = 4e8
    time_s = 1.3e-9
    voltage = 1e5
    segment = xt.LineSegmentMap(_context=test_context, length=length, qx=.1, qy=.2, momentum_compaction_factor=0.,
                                longitudinal_mode='nonlinear', voltage_rf=[voltage], frequency_rf=[frequency],
                                phase_rf=[0.], absolute_time=True)
    particles = xt.Particles(_context=test_context, p0c=2e9, mass0=xt.PROTON_MASS_EV, time_s=time_s)

    segment.track(particles)

    beta0 = ctx2np(particles.beta0)[0]
    energy_gain = ctx2np(particles.ptau)[0] * ctx2np(particles.p0c)[0]
    expected = voltage * np.sin(2 * np.pi * frequency * (time_s + .5 * length / (beta0 * clight)))
    xo.assert_allclose(energy_gain, expected, rtol=1e-9, atol=0)


@for_all_test_contexts
def test_line_segment_map_on_absolute_time_tracks_like_a_cavity_between_half_segments(test_context):
    alpha = 1e-3
    length = 628.
    reference = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
    frequency = 8 * reference.beta0[0] * clight / length + 20.
    arc = dict(qx=.1, qy=.2, momentum_compaction_factor=alpha, longitudinal_mode='nonlinear')
    whole = xt.Line(elements=[xt.LineSegmentMap(length=length, voltage_rf=[1e5], frequency_rf=[frequency],
                                                phase_rf=[0.], absolute_time=True, **arc)])
    split = xt.Line(elements=[
        xt.LineSegmentMap(length=length / 2, voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.], **arc),
        xt.Cavity(voltage=1e5, frequency=frequency, absolute_time=True),
        xt.LineSegmentMap(length=length / 2, voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.], **arc)])
    whole.particle_ref = reference.copy()
    split.particle_ref = reference.copy()
    whole.build_tracker(_context=test_context)
    split.build_tracker(_context=test_context)
    particles_whole = whole.build_particles(zeta=[.1], delta=[1e-4])
    particles_split = split.build_particles(zeta=[.1], delta=[1e-4])

    whole.track(particles_whole, num_turns=500)
    split.track(particles_split, num_turns=500)

    ctx2np = test_context.nparray_from_context_array
    xo.assert_allclose(ctx2np(particles_whole.zeta), ctx2np(particles_split.zeta), atol=1e-11, rtol=0)
    xo.assert_allclose(ctx2np(particles_whole.delta), ctx2np(particles_split.delta), atol=1e-13, rtol=0)


@for_all_test_contexts
def test_zero_length_line_segment_map_places_absolute_time_rf_at_its_position(test_context):
    # RF a quarter of the way round, not at the middle: it must track like a Cavity there.
    alpha = 1e-3
    length = 628.
    reference = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
    frequency = 8 * reference.beta0[0] * clight / length + 20.
    arc = dict(qx=.1, qy=.2, momentum_compaction_factor=alpha, longitudinal_mode='nonlinear')
    with_map = xt.Line(elements=[
        xt.LineSegmentMap(length=length / 4, voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.], **arc),
        xt.LineSegmentMap(length=0., voltage_rf=[1e5], frequency_rf=[frequency], phase_rf=[0.], absolute_time=True,
                          momentum_compaction_factor=alpha, longitudinal_mode='nonlinear'),
        xt.LineSegmentMap(length=3 * length / 4, voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.], **arc)])
    with_cavity = xt.Line(elements=[
        xt.LineSegmentMap(length=length / 4, voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.], **arc),
        xt.Cavity(voltage=1e5, frequency=frequency, absolute_time=True),
        xt.LineSegmentMap(length=3 * length / 4, voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.], **arc)])
    with_map.particle_ref = reference.copy()
    with_cavity.particle_ref = reference.copy()
    with_map.build_tracker(_context=test_context)
    with_cavity.build_tracker(_context=test_context)
    particles_map = with_map.build_particles(zeta=[.1], delta=[1e-4])
    particles_cavity = with_cavity.build_particles(zeta=[.1], delta=[1e-4])

    with_map.track(particles_map, num_turns=500)
    with_cavity.track(particles_cavity, num_turns=500)

    ctx2np = test_context.nparray_from_context_array
    xo.assert_allclose(ctx2np(particles_map.zeta), ctx2np(particles_cavity.zeta), atol=1e-11, rtol=0)
    xo.assert_allclose(ctx2np(particles_map.delta), ctx2np(particles_cavity.delta), atol=1e-13, rtol=0)


def test_line_segment_map_refuses_absolute_time_with_a_linear_mode():
    with pytest.raises(ValueError, match='absolute_time'):
        xt.LineSegmentMap(length=1., qx=.1, qy=.2, qs=.01, bets=10., longitudinal_mode='linear_fixed_qs',
                          absolute_time=True)


def test_line_segment_map_refuses_absolute_time_set_later_on_a_linear_mode():
    segment = xt.LineSegmentMap(length=1., qx=.1, qy=.2, qs=.01, bets=10., longitudinal_mode='linear_fixed_qs')
    with pytest.raises(ValueError, match='absolute_time'):
        segment.absolute_time = True


def test_line_segment_map_absolute_time_survives_serialization():
    segment = xt.LineSegmentMap(length=1., qx=.1, qy=.2, momentum_compaction_factor=1e-3,
                                longitudinal_mode='nonlinear', voltage_rf=[1e5], frequency_rf=[4e8],
                                phase_rf=[0.], absolute_time=True)
    assert xt.LineSegmentMap.from_dict(segment.to_dict()).absolute_time == 1
