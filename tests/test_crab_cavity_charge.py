import numpy as np
import pytest
from cpymad.madx import Madx
from scipy.constants import c

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.linear_normal_form import S


@for_all_test_contexts
def test_crab_cavity_charge_kicks_and_symplecticity(test_context):
    cavity = xt.CrabCavity(frequency=400e6, crab_voltage=1e6, phase=1.)
    line = xt.Line(elements=[cavity])
    line.build_tracker(_context=test_context)

    for q0, charge_ratio, mass_ratio in [
            (1, 1., 1.), (2, 1., 1.), (-2, 0.75, 1.25), (82, 1., 1.)]:
        for tilt in [0., np.pi / 2]:
            cavity.rot_s_rad = tilt
            p0 = xt.Particles(_context=test_context, p0c=1e9, q0=q0,
                              charge_ratio=charge_ratio, mass_ratio=mass_ratio,
                              x=0.02, px=1e-4, y=-0.015, py=2e-4,
                              zeta=0.05, delta=1e-3)
            beta0 = float(p0.beta0[0])
            k = 2 * np.pi * cavity.frequency / c
            phase = cavity.phase - k * 0.05 / beta0
            kick = abs(q0) * charge_ratio / mass_ratio * cavity.crab_voltage / 1e9
            x_local = 0.02 * np.cos(tilt) - 0.015 * np.sin(tilt)

            p = p0.copy()
            line.track(p)
            xo.assert_allclose(p.px, p0.px + kick * np.sin(phase) * np.cos(tilt),
                               rtol=1e-12, atol=1e-15)
            xo.assert_allclose(p.py, p0.py + kick * np.sin(phase) * np.sin(tilt),
                               rtol=1e-12, atol=1e-15)
            xo.assert_allclose(p.pzeta,
                               p0.pzeta - kick * k / beta0 * x_local * np.cos(phase),
                               rtol=1e-12, atol=1e-15)
            matrix = line.get_R_matrix(particle_on_co=p0.copy())['R_matrix']
            xo.assert_allclose(matrix.T @ S @ matrix, S, rtol=0, atol=2e-8)

            line.track(p, backtrack=True)
            for field in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
                xo.assert_allclose(getattr(p, field), getattr(p0, field),
                                   rtol=0, atol=1e-14)


@pytest.mark.parametrize('loader,charge', [
    ('cpymad', 1), ('cpymad', 2), ('cpymad', 82), ('native', 1)])
def test_crab_cavity_charge_madx_round_trip(loader, charge):
    source = f'''
        v = 4;
        phi = 0.3;
        cc: crabcavity, volt := v, freq = 400, lag := phi;
        rf: rfcavity, volt := v, freq = 400, lag := phi;
        seq: sequence, l = 1;
            cc1: cc, at = 0.5;
            rf1: rf, at = 0.75;
        endsequence;
        beam, particle = ion, charge = {charge}, mass = 193.6872729, pc = 500;
    '''
    with Madx(stdout=False) as mad:
        mad.input(source)
        mad.use(sequence='seq')
        if loader == 'cpymad':
            line = xt.Line.from_madx_sequence(mad.sequence.seq,
                                              deferred_expressions=True)
            line.particle_ref = xt.Particles(p0c=500e9, mass0=193.6872729e9,
                                             q0=charge)
        else:
            line = xt.load(string=source, format='madx')['seq']

        for voltage in [4., 7.]:
            line['v'] = voltage
            mad.globals.v = voltage
            xo.assert_allclose(line['cc1'].crab_voltage, voltage * 1e6 / charge,
                               rtol=1e-14, atol=0)
            xo.assert_allclose(line['rf1'].voltage, voltage * 1e6 / charge,
                               rtol=1e-14, atol=0)
            mad.twiss(betx=1, bety=1, x=0.01)
            tw = line.twiss(betx=1, bety=1, x=0.01)
            xo.assert_allclose(tw.px[-1], mad.table.twiss.px[-1],
                               rtol=1e-8, atol=1e-13)
            xo.assert_allclose(tw.ptau[-1], mad.table.twiss.pt[-1],
                               rtol=1e-8, atol=1e-13)

    exported = line.to_madx_sequence('roundtrip')
    with Madx(stdout=False) as mad:
        mad.input(exported)
        mad.command.beam(sequence='roundtrip', particle='ion', charge=charge,
                         mass=193.6872729, pc=500)
        mad.use(sequence='roundtrip')
        xo.assert_allclose(mad.elements.cc1.volt, 7., rtol=1e-14, atol=0)
        xo.assert_allclose(mad.elements.rf1.volt, 7., rtol=1e-14, atol=0)
        xo.assert_allclose(mad.elements.cc1.lag, 0.3, rtol=0, atol=1e-14)
        mad.twiss(betx=1, bety=1, x=0.01)
        xo.assert_allclose(tw.px[-1], mad.table.twiss.px[-1], rtol=1e-8, atol=1e-13)
        xo.assert_allclose(tw.ptau[-1], mad.table.twiss.pt[-1], rtol=1e-8, atol=1e-13)


@pytest.mark.parametrize('charge', [1, 2, 82])
def test_native_cavity_voltage_assumes_unit_charge(charge):
    loader = xt.mad_parser.MadxLoader()
    loader.load_string(f'''
        v = 4;
        cc: crabcavity, volt := v;
        rf: rfcavity, volt := v;
        a: line = (cc, rf, cc);
        b: line = (cc, rf);
        beam, sequence = a, particle = ion, charge = {charge}, mass = 193.7, pc = 500;
        beam, sequence = b, particle = ion, charge = 1, mass = 193.7, pc = 500;
    ''')
    env = loader.env
    assert env['a'].particle_ref.q0 == charge
    assert env['b'].particle_ref.q0 == 1
    for _ in range(2):
        xo.assert_allclose(env['cc'].crab_voltage, 4e6, rtol=1e-14)
        xo.assert_allclose(env['rf'].voltage, 4e6, rtol=1e-14)
        loader.load_string('v = 4;')

    loader.load_string('cc, volt := 2*v; rf, volt := 3*v;')
    env['v'] = 5
    xo.assert_allclose(env['cc'].crab_voltage, 10e6, rtol=1e-14)
    xo.assert_allclose(env['rf'].voltage, 15e6, rtol=1e-14)
