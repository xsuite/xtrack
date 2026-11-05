import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_exciter_on_absolute_time_reads_the_sample_at_the_arrival_time(test_context):
    # Sample i has the value i, so the kick says which sample was read.
    ctx2np = test_context.nparray_from_context_array
    sampling_frequency = 1e9
    time_s = 2.2e-6
    s = 15.
    zeta = .3
    exciter = xt.Exciter(_context=test_context, knl=[1e-6], samples=np.arange(4000.), absolute_time=True,
                         sampling_frequency=sampling_frequency, frev=1e6, start_turn=2)
    particles = xt.Particles(_context=test_context, p0c=1e10, time_s=time_s, s=s, zeta=zeta)

    exciter.track(particles)

    arrival = time_s - 2 / 1e6 + (s - zeta) / (ctx2np(particles.beta0)[0] * clight)
    assert np.round(-ctx2np(particles.px)[0] / 1e-6) == np.floor(sampling_frequency * arrival)


@for_all_test_contexts
def test_exciter_on_the_turn_clock_ignores_time_s_and_s(test_context):
    ctx2np = test_context.nparray_from_context_array
    sampling_frequency = 1e9
    exciter = xt.Exciter(_context=test_context, knl=[1e-6], samples=np.arange(4000.),
                         sampling_frequency=sampling_frequency, frev=1e6, start_turn=1)
    particles = xt.Particles(_context=test_context, p0c=1e10, at_turn=2, zeta=-.03, time_s=5., s=9.)

    exciter.track(particles)

    arrival = 1 / 1e6 + .03 / (ctx2np(particles.beta0)[0] * clight)
    assert np.round(-ctx2np(particles.px)[0] / 1e-6) == np.floor(sampling_frequency * arrival)


@for_all_test_contexts
def test_exciter_gives_no_kick_before_its_first_sample(test_context):
    ctx2np = test_context.nparray_from_context_array
    sampling_frequency = 1e9
    speed = xt.Particles(p0c=1e10).beta0[0] * clight
    exciter = xt.Exciter(_context=test_context, knl=[1e-6], samples=np.ones(10), frev=1e6,
                         sampling_frequency=sampling_frequency)
    before_first_sample = xt.Particles(_context=test_context, p0c=1e10, zeta=.8 / sampling_frequency * speed)
    after_first_sample = xt.Particles(_context=test_context, p0c=1e10, zeta=-.2 / sampling_frequency * speed)

    exciter.track(before_first_sample)
    exciter.track(after_first_sample)

    assert ctx2np(before_first_sample.px)[0] == 0.
    assert ctx2np(after_first_sample.px)[0] != 0.


@for_all_test_contexts
def test_exciter_on_absolute_time_gives_no_kick_before_its_first_sample(test_context):
    ctx2np = test_context.nparray_from_context_array
    sampling_frequency = 1e9
    speed = xt.Particles(p0c=1e10).beta0[0] * clight
    exciter = xt.Exciter(_context=test_context, knl=[1e-6], samples=np.ones(10), absolute_time=True,
                         sampling_frequency=sampling_frequency)
    before_first_sample = xt.Particles(_context=test_context, p0c=1e10, zeta=.8 / sampling_frequency * speed)
    after_first_sample = xt.Particles(_context=test_context, p0c=1e10, zeta=-.2 / sampling_frequency * speed)

    exciter.track(before_first_sample)
    exciter.track(after_first_sample)

    assert ctx2np(before_first_sample.px)[0] == 0.
    assert ctx2np(after_first_sample.px)[0] != 0.


def test_exciter_with_samples_refuses_no_revolution_frequency():
    with pytest.raises(ValueError, match='frev'):
        xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6)


def test_exciter_with_a_start_turn_refuses_no_revolution_frequency_on_absolute_time():
    with pytest.raises(ValueError, match='frev'):
        xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, absolute_time=True, start_turn=2)


def test_exciter_refuses_revolution_frequency_set_to_zero():
    exciter = xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, frev=1e6)
    with pytest.raises(ValueError, match='frev'):
        exciter.frev = 0.


def test_exciter_refuses_leaving_absolute_time_with_no_revolution_frequency():
    exciter = xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, absolute_time=True)
    with pytest.raises(ValueError, match='frev'):
        exciter.absolute_time = False


def test_exciter_refuses_a_start_turn_set_later_with_no_revolution_frequency():
    exciter = xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, absolute_time=True)
    with pytest.raises(ValueError, match='frev'):
        exciter.start_turn = 2


def test_exciter_guarded_fields_survive_serialization():
    exciter = xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, frev=1e6, start_turn=2,
                         absolute_time=True)
    loaded = xt.Exciter.from_dict(exciter.to_dict())
    assert (loaded.frev, loaded.start_turn, loaded.absolute_time) == (1e6, 2, 1)


@for_all_test_contexts
def test_exciter_on_absolute_time_follows_the_lab_clock_through_a_ramp(test_context):
    ctx2np = test_context.nparray_from_context_array
    sampling_frequency = 1e7
    s_exciter = 100.
    frev = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9).beta0[0] * clight / 628.
    exciter = xt.Exciter(knl=[0.], samples=np.arange(int(1e6), dtype=float), frev=frev,
                         sampling_frequency=sampling_frequency, absolute_time=True)
    line = xt.Line(elements={'before': xt.Drift(length=s_exciter), 'exciter': exciter, 'after': xt.Drift(length=528.)})
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9)
    line.energy_program = xt.EnergyProgram(t_s=[0., .01], kinetic_energy0=[1.4e9, 2e9])
    line.enable_time_dependent_vars = True
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.])

    line.track(particles, num_turns=2000)

    line.enable_time_dependent_vars = False
    exciter.knl[0] = 1e-9
    particles.zeta = 0.
    particles.px = 0.
    particles.s = s_exciter
    line.track(particles, ele_start='exciter', ele_stop='after')
    sample_read = -ctx2np(particles.px)[0] / 1e-9
    lab_clock_sample = sampling_frequency * (ctx2np(particles.time_s) + s_exciter / (ctx2np(particles.beta0)[0] * clight))
    assert abs(sample_read - lab_clock_sample) <= 1.
