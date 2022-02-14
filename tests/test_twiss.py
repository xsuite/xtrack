import sys
import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from cpymad.madx import Madx

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

path = test_data_folder.joinpath('hllhc14_input_mad/')

mad = Madx(command_log="mad_final.log")
mad.call(str(path.joinpath("final_seq.madx")))
mad.use(sequence="lhcb1")
mad.twiss()
mad.readtable(file=str(path.joinpath("final_errors.tfs")),
                table="errtab")
mad.seterr(table="errtab")
mad.set(format=".15g")
twmad = mad.twiss(rmatrix=True, chrom=True)

def test_twiss():

    line = xt.Line.from_madx_sequence(
            mad.sequence['lhcb1'], apply_madx_errors=True)
    line.elements[10].iscollective = True # we make it artificially collective to test this option
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                            gamma0=mad.sequence.lhcb1.beam.gamma)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        tracker = xt.Tracker(_context=context, line=line)
        assert tracker.iscollective

        twxt = tracker.twiss()
        assert np.isclose(mad.table.summ.q1[0], twxt['qx'], rtol=1e-4, atol=0)
        assert np.isclose(mad.table.summ.q2[0], twxt['qy'], rtol=1e-4, atol=0)
        assert np.isclose(mad.table.summ.dq1, twxt['dqx'], atol=0.1, rtol=0)
        assert np.isclose(mad.table.summ.dq2, twxt['dqy'], atol=0.1, rtol=0)
        assert np.isclose(mad.table.summ.alfa[0],
            twxt['momentum_compaction_factor'],
            atol=2e-10, rtol=0)
        assert np.isclose(twxt['qs'], 0.0021, atol=1e-4, rtol=0)

        for name in ['mb.b19r5.b1', 'mb.b19r1.b1',
                    'ip1', 'ip2', 'ip5', 'ip8',
                    'mbxf.4l1', 'mbxf.4l5']:

            imad = list(twmad['name']).index(name+':1')
            ixt = list(twxt['name']).index(name)

            assert np.isclose(twxt['betx'][ixt], twmad['betx'][imad],
                            atol=0, rtol=3e-4)
            assert np.isclose(twxt['bety'][ixt], twmad['bety'][imad],
                            atol=0, rtol=3e-4)
            assert np.isclose(twxt['alfx'][ixt], twmad['alfx'][imad],
                              atol=1e-1, rtol=0)
            assert np.isclose(twxt['alfy'][ixt], twmad['alfy'][imad],
                              atol=1e-1, rtol=0)
            assert np.isclose(twxt['dx'][ixt], twmad['dx'][imad],
                              atol=1e-2, rtol=0)
            assert np.isclose(twxt['dy'][ixt], twmad['dy'][imad],
                              atol=1e-2, rtol=0)
            assert np.isclose(twxt['dpx'][ixt], twmad['dpx'][imad],
                              atol=3e-4, rtol=0)
            assert np.isclose(twxt['dpy'][ixt], twmad['dpy'][imad],
                              atol=3e-4, rtol=0)
            assert np.isclose(twxt['mux'][ixt], twmad['mux'][imad],
                              atol=1e-4, rtol=0)
            assert np.isclose(twxt['muy'][ixt], twmad['muy'][imad],
                              atol=1e-4, rtol=0)

            assert np.isclose(twxt['s'][ixt], twmad['s'][imad], atol=5e-6, rtol=0)
            assert np.isclose(twxt['x'][ixt], twmad['x'][imad], atol=5e-6, rtol=0)
            assert np.isclose(twxt['y'][ixt], twmad['y'][imad], atol=5e-6, rtol=0)
            assert np.isclose(twxt['px'][ixt], twmad['px'][imad], atol=1e-7, rtol=0)
            assert np.isclose(twxt['py'][ixt], twmad['py'][imad], atol=1e-7, rtol=0)
def norm(x):
    return np.sqrt(np.sum(np.array(x) ** 2))

def test_line_import_from_madx():

    rtol = 1e-7
    strict = True
    atol = 1e-14

    print('Build line with expressions...')
    line_with_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=True,
        deferred_expressions=True)
    line_with_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    print('Build line without expressions...')
    line_no_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=True,
        deferred_expressions=False)
    line_no_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    ltest = line_with_expressions
    lref = line_no_expressions

    print('Start consistency check')
    # Check that the two machines are identical
    assert len(ltest) == len(lref)

    assert (ltest.get_length() - lref.get_length()) < 1e-6

    for ii, (ee_test, ee_six, nn_test, nn_six) in enumerate(
        zip(ltest.elements, lref.elements, ltest.element_names, lref.element_names)
    ):
        assert type(ee_test) == type(ee_six)

        dtest = ee_test.to_dict()
        dref = ee_six.to_dict()

        for kk in dtest.keys():

            # Check if they are identical
            if np.isscalar(dref[kk]) and dtest[kk] == dref[kk]:
                continue

            if isinstance(dref[kk], dict):
                if kk=='fieldmap':
                    continue
                if kk=='boost_parameters':
                    continue
                if kk=='Sigmas_0_star':
                    continue

            # Check if the relative error is small
            val_test = dtest[kk]
            val_ref = dref[kk]
            if ((not np.isscalar(val_ref) and len(val_ref) != len(val_test))
                    or norm(val_test) < 1e-14) :
                diff_rel = 100
            else:
                diff_rel = norm(np.array(val_test) - np.array(val_ref)) / norm(val_test)
            if diff_rel < rtol:
                continue

            # Check if absolute error is small

            if not np.isscalar(val_ref) and len(val_ref) != len(val_test):
                diff_abs = 1000
            else:
                diff_abs = norm(np.array(val_test) - np.array(val_ref))
            if diff_abs < atol:
                continue

            # If it got here it means that no condition above is met
            raise ValueError("Too large discrepancy!")