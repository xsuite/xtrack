from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np
import pathlib

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_native_madloader_lhc_thin():

    env = xt.load(test_data_folder / 'hllhc15_noerrors_nobb/sequence.madx',
                reverse_lines=['lhcb2'])
    env.lhcb1.set_particle_ref('proton', p0c=7000e9)
    env.lhcb2.set_particle_ref('proton', p0c=7000e9)

    mad = Madx()
    mad.call(str(test_data_folder / 'hllhc15_noerrors_nobb/sequence.madx'))
    mad.use('lhcb1')
    mad.use('lhcb2')

    lb1_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
    lb2_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb2)

    lb1_ref.set_particle_ref('proton', p0c=7000e9)
    lb2_ref.set_particle_ref('proton', p0c=7000e9)

    for lref, ltest, beam in [(lb1_ref, env.lhcb1, 1), (lb2_ref, env.lhcb2, 2)]:

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

        for nn in ['lhcb1$start', 'lhcb1$end', 'lhcb2$start', 'lhcb2$end']:
            if nn in lref_names:
                lref_names.remove(nn)

        ltest_names_modif = [
            nn[:-len(f'/lhcb{beam}')] if nn.endswith(f'/lhcb{beam}') else nn for nn in ltest_names]

        for nn_test, nn_ref in zip(ltest_names_modif, lref_names):
            assert nn_test == nn_ref, f'Element name mismatch: {nn_test} != {nn_ref}'

        xo.assert_allclose(
            tt_ref_nodr.rows[lref_names].s_center, tt_test_nodr.rows[ltest_names].s_center,
            rtol=0, atol=5e-9)

        for nn in ltest_names:
            print(f'Checking: {nn}                     ', end='\r', flush=True)
            if nn == '_end_point':
                continue
            nn_straight = nn[:-len(f'/lhcb{beam}')] if nn.endswith(f'/lhcb{beam}') else nn
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

        xo.assert_allclose(twref.rows['ip.*'].betx, twtest.rows['ip.*'].betx, rtol=1e-6,
                        atol=0)
        xo.assert_allclose(twref.rows['ip.*'].bety, twtest.rows['ip.*'].bety, rtol=1e-6,
                        atol=0)
        xo.assert_allclose(twref.rows['ip.*'].dx, twtest.rows['ip.*'].dx, rtol=0,
                        atol=1e-6)
        xo.assert_allclose(twref.rows['ip.*'].dy, twtest.rows['ip.*'].dy, rtol=1e-6,
                        atol=1e-6)
        xo.assert_allclose(twref.rows['ip.*'].ax_chrom, twtest.rows['ip.*'].ax_chrom,
                        rtol=1e-4, atol=1e-5)
        xo.assert_allclose(twref.rows['ip.*'].ay_chrom, twtest.rows['ip.*'].ay_chrom,
                        rtol=1e-4, atol=1e-5)