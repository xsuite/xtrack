import pathlib

import numpy as np

import xobjects as xo
import xtrack as xt

from cpymad.madx import Madx

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_coupling_edwards_teng():

    mad = Madx()
    mad.call(str(test_data_folder / 'lhc_2024/lhc.seq'))
    mad.call(str(test_data_folder / 'lhc_2024/injection_optics.madx'))

    mad.beam()
    mad.use('lhcb1')

    mad.globals.on_x1 = 0
    mad.globals.on_x2h = 0
    mad.globals.on_x2v = 0
    mad.globals.on_x5 = 0
    mad.globals.on_x8h = 0
    mad.globals.on_x8v = 0

    mad.globals.on_sep1 = 0
    mad.globals.on_sep2h = 0
    mad.globals.on_sep2v = 0
    mad.globals.on_sep5 = 0
    mad.globals.on_sep8h = 0
    mad.globals.on_sep8v = 0

    mad.globals.on_a2 = 0
    mad.globals.on_a8 = 0

    mad.globals['kqs.a67b1'] = 1e-4

    twmad = mad.twiss()

    line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
    line.particle_ref = xt.Particles(p0c=450e9)
    tw = line.twiss4d(coupling_edw_teng=True)


    r11_mad, r12_mad, r21_mad, r22_mad = twmad.r11, twmad.r12, twmad.r21, twmad.r22

    s_mad = twmad.s

    r11_mad_at_s = np.interp(tw.s, s_mad, r11_mad)
    r12_mad_at_s = np.interp(tw.s, s_mad, r12_mad)
    r21_mad_at_s = np.interp(tw.s, s_mad, r21_mad)
    r22_mad_at_s = np.interp(tw.s, s_mad, r22_mad)
    betx_mad_at_s = np.interp(tw.s, s_mad, twmad.betx)
    bety_mad_at_s = np.interp(tw.s, s_mad, twmad.bety)
    alfx_mad_at_s = np.interp(tw.s, s_mad, twmad.alfx)
    alfy_mad_at_s = np.interp(tw.s, s_mad, twmad.alfy)

    rdt_mad_at_s = compute_rdt(r11_mad_at_s, r12_mad_at_s, r21_mad_at_s, r22_mad_at_s,
                            betx_mad_at_s, bety_mad_at_s, alfx_mad_at_s, alfy_mad_at_s)

    xo.assert_allclose(tw.r11_edw_teng, r11_mad_at_s,
                    rtol=1e-5, atol=1e-5*np.max(np.abs(r11_mad_at_s)))
    xo.assert_allclose(tw.r12_edw_teng, r12_mad_at_s,
                    rtol=1e-5, atol=1e-5*np.max(np.abs(r12_mad_at_s)))
    xo.assert_allclose(tw.r21_edw_teng, r21_mad_at_s,
                    rtol=1e-5, atol=1e-5*np.max(np.abs(r21_mad_at_s)))
    xo.assert_allclose(tw.r22_edw_teng, r22_mad_at_s,
                    rtol=1e-5, atol=1e-5*np.max(np.abs(r22_mad_at_s)))
    xo.assert_allclose(tw.betx_edw_teng, betx_mad_at_s, atol=0, rtol=1e-5)

    xo.assert_allclose(tw.betx_edw_teng, betx_mad_at_s, atol=0, rtol=5e-8)
    xo.assert_allclose(tw.alfx_edw_teng, alfx_mad_at_s, atol=1e-4, rtol=1e-8)
    xo.assert_allclose(tw.bety_edw_teng, bety_mad_at_s, atol=0, rtol=5e-8)
    xo.assert_allclose(tw.alfy_edw_teng, alfy_mad_at_s, atol=1e-4, rtol=1e-8)

    xo.assert_allclose(tw.f1001, rdt_mad_at_s['f1001'],
                    atol=1e-5 * np.max(np.abs(rdt_mad_at_s['f1001'])))
    xo.assert_allclose(tw.f1010, rdt_mad_at_s['f1010'],
                    atol=1e-5 * np.max(np.abs(rdt_mad_at_s['f1010'])))


def compute_rdt(r11, r12, r21, r22, betx, bety, alfx, alfy):

    '''
    Developed by CERN OMC team.
    Ported from:
    https://pypi.org/project/optics-functions/
    https://github.com/pylhc/optics_functions

    Based on Calaga, Tomas, https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.8.034001
    '''

    n = len(r11)
    assert len(r12) == n
    assert len(r21) == n
    assert len(r22) == n
    gx, r, inv_gy = np.zeros((n, 2, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))

    # Eq. (16)  C = 1 / (1 + |R|) * -J R J
    # rs form after -J R^T J
    r[:, 0, 0] = r22
    r[:, 0, 1] = -r12
    r[:, 1, 0] = -r21
    r[:, 1, 1] = r11

    r *= 1 / np.sqrt(1 + np.linalg.det(r)[:, None, None])

    # Cbar = Gx * C * Gy^-1,   Eq. (5)
    sqrt_betax = np.sqrt(betx)
    sqrt_betay = np.sqrt(bety)

    gx[:, 0, 0] = 1 / sqrt_betax
    gx[:, 1, 0] = alfx * gx[:, 0, 0]
    gx[:, 1, 1] = sqrt_betax

    inv_gy[:, 1, 1] = 1 / sqrt_betay
    inv_gy[:, 1, 0] = -alfy * inv_gy[:, 1, 1]
    inv_gy[:, 0, 0] = sqrt_betay

    c = np.matmul(gx, np.matmul(r, inv_gy))
    gamma = np.sqrt(1 - np.linalg.det(c))

    # Eq. (9) and Eq. (10)
    denom = 1 / (4 * gamma)
    f1001 = denom * (+c[:, 0, 1] - c[:, 1, 0] + (c[:, 0, 0] + c[:, 1, 1]) * 1j)
    f1010 = denom * (-c[:, 0, 1] - c[:, 1, 0] + (c[:, 0, 0] - c[:, 1, 1]) * 1j)

    return {'f1001': f1001, 'f1010': f1010}
