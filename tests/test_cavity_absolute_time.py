import pathlib
import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from scipy.constants import c as clight

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_cavity_absolute_time(test_context):
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    line['vrf400'] = 16

    for vv in line.vars.get_table().rows[
        'on_x.*|on_sep.*|on_crab.*|on_alice|on_lhcb|corr_.*'].name:
        line.vars[vv] = 0

    tw = line.twiss()

    df_hz = -50
    h_rf = 35640
    f_rev = 1/tw.t_rev0
    df_rev = df_hz / h_rf
    eta = tw.slip_factor

    f_rf0 = 1/tw.t_rev0 * h_rf

    f_rf = f_rf0 + df_hz
    line.vars['f_rf'] = f_rf
    tt = line.get_table()
    for nn in tt.rows[tt.element_type=='Cavity'].name:
        line[nn].absolute_time = 1 # Need property
        line[nn].harmonic = 0
        line[nn].frequency = line.vars['f_rf']
        # absolute_time uses one laboratory origin for every cavity. Preserve
        # the synchronous phase at the kick in the center of each thick cavity.
        s_kick = tt['s', nn] + line[nn].length / 2
        line[nn].phase -= 2 * np.pi * f_rf * s_kick / (tw.beta0 * clight)

    tw1 = line.twiss(search_for_t_rev=True)

    f_rev_expected = f_rf / h_rf

    xo.assert_allclose(f_rev_expected, 1/tw1.t_rev, atol=1e-5, rtol=0)
    xo.assert_allclose(tw1.delta, tw1.delta[0], atol=1e-5, rtol=0) # Check that it is flat
    delta_expected = -df_rev / f_rev / eta
    xo.assert_allclose(tw1.delta, delta_expected, atol=2e-6, rtol=0)
    tw_off_mom = line.twiss(method='4d', delta0=tw1.delta[0])
    xo.assert_allclose(tw1.x, tw_off_mom.x, atol=1e-5, rtol=0)


@for_all_test_contexts
def test_locked_cavity_keeps_the_reference_on_phase_through_an_energy_program(test_context):
    # A harmonic cavity is assumed locked to h cycles per revolution regardless of beam energy.
    # Ensure that the RF phase observed by the beam is identical on each turn.
    ctx2np = test_context.nparray_from_context_array
    cavity = xt.Cavity(voltage=0., harmonic=8, absolute_time=True)
    line = xt.Line(elements={'rf': cavity, 'arc': xt.Drift(length=628.)})
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9)
    line.energy_program = xt.EnergyProgram(t_s=[0., .01], kinetic_energy0=[1.4e9, 2e9])
    line.enable_time_dependent_vars = True
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.])

    line.track(particles, num_turns=100)

    # With `phase` zero, a reference particle on phase crosses the cavity with no energy gain.
    line.enable_time_dependent_vars = False
    cavity.voltage = 1e5
    particles.zeta = 0.
    particles.delta = 0.
    line.track(particles, ele_stop='arc')
    energy_gain = ctx2np(particles.ptau)[0] * ctx2np(particles.p0c)[0]
    assert abs(energy_gain) < 1e-3


@for_all_test_contexts
@pytest.mark.parametrize('tracking_calls', [1, 100])
def test_locked_cavity_keeps_the_reference_on_phase_through_a_reference_energy_increase(
        test_context, tracking_calls):
    ctx2np = test_context.nparray_from_context_array
    cavity = xt.Cavity(voltage=0., harmonic=8, absolute_time=True)
    line = xt.Line(elements={'rf': cavity, 'arc': xt.Drift(length=628.),
                             'ramp': xt.ReferenceEnergyIncrease(Delta_p0c=1e6)})
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=2e9)
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.])

    for _ in range(tracking_calls):
        line.track(particles, num_turns=100 // tracking_calls)

    cavity.voltage = 1e5
    particles.zeta = 0.
    particles.delta = 0.
    line.track(particles, ele_stop='arc')
    energy_gain = ctx2np(particles.ptau)[0] * ctx2np(particles.p0c)[0]
    assert abs(energy_gain) < 1e-3


@for_all_test_contexts
def test_locked_cavity_phase_includes_its_position_in_the_ring(test_context):
    ctx2np = test_context.nparray_from_context_array
    circumference = 628.
    s_cavity = 100.
    harmonic = 8
    cavity = xt.Cavity(voltage=1e5, harmonic=harmonic, absolute_time=True,
                       phase=-2 * np.pi * harmonic * s_cavity / circumference)
    line = xt.Line(elements={'before': xt.Drift(length=s_cavity), 'rf': cavity,
                             'after': xt.Drift(length=circumference - s_cavity)})
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=2e9)
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.], delta=[0.])

    line.track(particles, ele_stop='after')

    energy_gain = ctx2np(particles.ptau)[0] * ctx2np(particles.p0c)[0]
    assert abs(energy_gain) < 1e-3


@for_all_test_contexts
def test_locked_cavity_on_the_absolute_clock_has_the_analytic_synchrotron_tune(test_context):
    # Qs^2 = h |eta| q V / (2 pi beta0^2 E0), measured by counting oscillations of zeta.
    harmonic = 8
    voltage = 1e5
    alpha = 1e-3
    turns = 20000
    arc = xt.LineSegmentMap(length=628., qx=.1, qy=.2, momentum_compaction_factor=alpha,
                            longitudinal_mode='nonlinear', voltage_rf=[0.], frequency_rf=[0.], phase_rf=[0.])
    line = xt.Line(elements=[xt.Cavity(voltage=voltage, harmonic=harmonic, absolute_time=True), arc])
    line.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
    line.build_tracker(_context=test_context)
    beta0 = line.particle_ref.beta0[0]
    eta = alpha - 1 / line.particle_ref.gamma0[0]**2
    expected = np.sqrt(harmonic * abs(eta) * voltage / (2 * np.pi * beta0**2 * line.particle_ref.energy0[0]))

    particles = line.build_particles(zeta=[1e-3], delta=[0.])
    line.track(particles, num_turns=turns, turn_by_turn_monitor=True)

    zeta = line._context.nparray_from_context_array(line.record_last_track.zeta)[0]
    zero_crossings = np.count_nonzero(np.diff(np.sign(zeta)))
    xo.assert_allclose(zero_crossings / 2 / turns, expected, rtol=.01, atol=0)


@for_all_test_contexts
@pytest.mark.parametrize('mode', ['thin', 'thick'])
def test_locked_sliced_cavity_keeps_the_reference_on_phase_through_an_energy_program(test_context, mode):
    # With the phase zero at the middle of the cavity, the slices either side cancel.
    ctx2np = test_context.nparray_from_context_array
    circumference = 628.
    length = 1.
    harmonic = 8
    cavity = xt.Cavity(voltage=0., length=length, harmonic=harmonic, absolute_time=True,
                       phase=-2 * np.pi * harmonic * (length / 2) / circumference)
    line = xt.Line(elements={'rf': cavity, 'arc': xt.Drift(length=circumference - length)})
    line.slice_thick_elements(slicing_strategies=[
        xt.Strategy(None), xt.Strategy(xt.Uniform(2, mode=mode), element_type=xt.Cavity)])
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9)
    line.energy_program = xt.EnergyProgram(t_s=[0., .01], kinetic_energy0=[1.4e9, 2e9])
    line.enable_time_dependent_vars = True
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.])

    line.track(particles, num_turns=100)

    line.enable_time_dependent_vars = False
    cavity.voltage = 1e5
    particles.zeta = 0.
    particles.delta = 0.
    line.track(particles, ele_stop='arc')
    energy_gain = ctx2np(particles.ptau)[0] * ctx2np(particles.p0c)[0]
    assert abs(energy_gain) < 1e-3
