from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np
import pathlib

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_native_madloader_ps():
    env = xt.load([test_data_folder / 'ps_sftpro/ps.seq',
                   test_data_folder / 'ps_sftpro/ps_hs_sftpro.str'])
    env.ps.set_particle_ref('proton', p0c=450e9)

    mad = Madx()
    mad.call(str(test_data_folder / 'ps_sftpro/ps.seq'))
    mad.call(str(test_data_folder / 'ps_sftpro/ps_hs_sftpro.str'))
    mad.beam()
    mad.use('ps')

    lref = xt.Line.from_madx_sequence(mad.sequence.ps, deferred_expressions=True)
    lref.set_particle_ref('proton', p0c=450e9)

    ltest = env.ps

    tt_ref = lref.get_table()
    tt_test = ltest.get_table()

    tt_ref_nodr = tt_ref.rows[
        (tt_ref.element_type != 'Drift') & (tt_ref.element_type != 'UniformSolenoid')
        & (tt_ref.rows.mask['.*_aper'] == False)]
    tt_test_nodr = tt_test.rows[
        (tt_test.element_type != 'Drift') & (tt_test.element_type != 'UniformSolenoid')
        & (tt_test.rows.mask['.*_aper'] == False)]

    # Check s
    lref_names = list(tt_ref_nodr.name)
    ltest_names = list(tt_test_nodr.name)

    for nn in lref_names.copy():
        if "$" in nn:
            lref_names.remove(nn)

    for nn_test, nn_ref in zip(ltest_names, lref_names):
        assert nn_test == nn_ref, f'Element name mismatch: {nn_test} != {nn_ref}'

    xo.assert_allclose(
        tt_ref_nodr.rows[lref_names].s_center, tt_test_nodr.rows[ltest_names].s_center,
        rtol=0, atol=1e-9)

    for nn in ltest_names:
        print(f'Checking: {nn}                     ', end='\r', flush=True)
        if nn == '_end_point':
            continue
        nn_straight = nn
        eref = lref[nn_straight]
        etest = ltest[nn]
        dref = eref.to_dict()
        dtest = etest.to_dict()
        is_rbend = isinstance(etest, xt.RBend)

        for kk in dref.keys():

            if kk == 'prototype':
                continue  # prototype is always None from cpymad

            if kk in ('__class__', 'model', 'side'):
                assert dref[kk] == dtest[kk]
                continue

            if kk == 'k0_from_h' and dref['k0'] == 0 and dref['h'] == 0:
                continue  # Skip the check when k0 is computed from h (for now)

            if kk in ('isthick', '_isthick') and eref.length == 0:
                continue  # Skip the check for zero-length elements

            if kk in {
                'order',  # Always assumed to be 5, not always the same
                'frequency',  # If not specified, depends on the beam,
                                # so for now we ignore it
            }:
                continue

            if kk in {'knl', 'ksl'}:
                maxlen = max(len(dref[kk]), len(dtest[kk]))
                lhs = np.pad(dref[kk], (0, maxlen - len(dref[kk])), mode='constant')
                rhs = np.pad(dtest[kk], (0, maxlen - len(dtest[kk])), mode='constant')
                xo.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-16)
                continue

            if is_rbend and kk in ('length', 'length_straight'):
                xo.assert_allclose(dref[kk], dtest[kk], rtol=1e-7, atol=1e-6)
                continue

            if is_rbend and kk in ('h', 'k0'):
                xo.assert_allclose(dref[kk], dtest[kk], rtol=1e-7, atol=5e-10)
                continue

            xo.assert_allclose(dref[kk], dtest[kk], rtol=1e-10, atol=1e-16)

    twref = lref.twiss4d()
    twtest = ltest.twiss4d()

    xo.assert_allclose(twtest.betx[-1], twref.betx[-1], rtol=1e-6, atol=0)
    xo.assert_allclose(twtest.bety[-1], twref.bety[-1], rtol=1e-6, atol=0)
    xo.assert_allclose(twtest.dx[-1], twref.dx[-1], rtol=0, atol=1e-6)
    xo.assert_allclose(twtest.dy[-1], twref.dy[-1], rtol=1e-6, atol=1e-6)
    xo.assert_allclose(twtest.dpx[-1], twref.dpx[-1], rtol=0, atol=1e-6)
    xo.assert_allclose(twtest.dpy[-1], twref.dpy[-1], rtol=0, atol=1e-6)
    xo.assert_allclose(twtest.mux[-1], twref.mux[-1], rtol=1e-6, atol=0)
    xo.assert_allclose(twtest.muy[-1], twref.muy[-1], rtol=1e-6, atol=0)
    xo.assert_allclose(twtest.wx_chrom[-1], twref.wx_chrom[-1], rtol=1e-4, atol=0)
    xo.assert_allclose(twtest.wy_chrom[-1], twref.wy_chrom[-1], rtol=1e-4, atol=0)
    xo.assert_allclose(twtest.ax_chrom[-1], twref.ax_chrom[-1], rtol=1e-4, atol=0)
    xo.assert_allclose(twtest.ay_chrom[-1], twref.ay_chrom[-1], rtol=1e-4, atol=0)
    xo.assert_allclose(twtest.dqx, twref.dqx, rtol=1e-4, atol=0)
    xo.assert_allclose(twtest.dqy, twref.dqy, rtol=1e-4, atol=0)
