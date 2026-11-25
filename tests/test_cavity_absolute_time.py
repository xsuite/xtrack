import pathlib
from decimal import Decimal, localcontext
import numpy as np

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
def test_absolute_rf_phase_precision(test_context):

    ctx2np = test_context.nparray_from_context_array
    voltage = 1e6
    p0c = 45.6e9
    phase = .23
    zeta = np.array([0., 100e-6])
    for frequency in (400e6, 400780313.0016023):
        elements = (
            (xt.Cavity(_context=test_context, voltage=voltage, frequency=frequency,
                       phase=phase, absolute_time=True), 'ptau', 0.),
            (xt.RFMultipole(_context=test_context, voltage=voltage, frequency=frequency,
                            phase=phase, absolute_time=True), 'ptau', 0.),
            (xt.CrabCavity(_context=test_context, crab_voltage=voltage, frequency=frequency,
                           phase=phase, absolute_time=True), 'px', 0.),
            (xt.LineSegmentMap(_context=test_context, length=2., qx=0., qy=0.,
                longitudinal_mode='nonlinear', momentum_compaction_factor=0.,
                voltage_rf=[voltage], frequency_rf=[frequency], phase_rf=[phase],
                absolute_time=True), 'ptau', 1.),
        )
        for element, coordinate, kick_offset in elements:
            for s in (0., 1234.5):
                for time_s in (0., 86400., -86400.):
                    particles = xt.Particles(_context=test_context, p0c=p0c,
                        mass0=xt.ELECTRON_MASS_EV, q0=-1, zeta=zeta, s=s, time_s=time_s)
                    beta0 = ctx2np(particles.beta0)[0]
                    with localcontext() as ctx:
                        ctx.prec = 60
                        cycles = (Decimal.from_float(frequency)
                                  * Decimal.from_float(time_s)) % 1

                    expected = voltage * np.sin(phase + 2*np.pi*(
                        float(cycles) + frequency*(s + kick_offset - zeta)/(beta0*clight)))
                    element.track(particles)
                    actual = ctx2np(getattr(particles, coordinate))*p0c
                    xo.assert_allclose(actual, expected, atol=1e-4, rtol=0)
