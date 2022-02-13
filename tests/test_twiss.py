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
    atol=1e1

    line_with_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False,
        deferred_expressions=True)
    line_with_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    line_no_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False,
        deferred_expressions=False)
    line_no_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    ltest = line_with_expressions
    lref = line_no_expressions

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
            try:
                if not np.isscalar(val_ref) and len(val_ref) != len(val_test):
                        diff_rel = 100
                    #lmin = min(len(val_ref), len(val_test))
                    #val_test = val_test[:lmin]
                    #val_ref = val_ref[:lmin]
                else:
                    diff_rel = norm(np.array(val_test) - np.array(val_ref)) / norm(val_test)
            except ZeroDivisionError:
                diff_rel = 100.0
            if diff_rel < rtol:
                continue

            # Check if absolute error is small

            if not np.isscalar(val_ref) and len(val_ref) != len(val_test):
                diff_abs = 1000
            else:
                diff_abs = norm(np.array(val_test) - np.array(val_ref))
            if diff_abs > 0:
                print(f"{nn_test}[{kk}] - test:{dtest[kk]} six:{dref[kk]}")
            if diff_abs < atol:
                continue

            # Exception: drift length (100 um tolerance)
            if not(strict) and isinstance(ee_test, xt.Drift):
                if kk == "length":
                    if diff_abs < 1e-4:
                        continue

            # Exception: multipole lrad is not passed to sixtraxk
            if isinstance(ee_test, xt.Multipole):
                if kk == "length":
                    if np.abs(ee_test.hxl) + np.abs(ee_test.hyl) == 0.0:
                        continue
                if kk == "order":
                    # Checked through bal
                    continue
                if kk == 'knl' or kk == 'ksl' or kk == 'bal':
                    if len(val_ref) != len(val_test):
                        lmin = min(len(val_ref), len(val_test))
                        for vv in [val_ref,val_test]:
                            if len(vv)> lmin:
                                for oo in range(lmin, len(vv)):
                                    # we do not care about errors above 10
                                    if vv[oo] != 0 and oo < {'knl':10,
                                                         'ksl':10, 'bal':20}[kk]:
                                        raise ValueError(
                                            'Missing significant multipole strength')

                        val_ref = val_ref[:lmin]
                        val_test = val_test[:lmin]

                    if len(val_ref) == 0 and len(val_test) == 0:
                        continue
                    else:
                        diff_abs = norm(np.array(val_test) - np.array(val_ref))
                        diff_rel = diff_abs/norm(val_test)
                        if diff_rel < rtol:
                            continue
                        if diff_abs < atol:
                            continue

            # Exception: correctors involved in lumi leveling
            passed_corr = False
            for nn_corr in [
                'mcbcv.5l8.b1', 'mcbyv.a4l8.b1', 'mcbxv.3l8',
                'mcbyv.4r8.b1', 'mcbyv.b5r8.b1',
                'mcbyh.b5l2.b1', 'mcbyh.4l2.b1', 'mcbxh.3l2', 'mcbyh.a4r2.b1',
                'mcbch.5r2.b1',
                'mcbcv.5l8.b2', 'mcbyv.a4l8.b2', 'mcbxv.3l8',
                'mcbyv.4r8.b2', 'mcbyv.b5r8.b2',
                'mcbyh.b5l2.b2', 'mcbyh.4l2.b2', 'mcbxh.3l2', 'mcbyh.a4r2.b2',
                'mcbch.5r2.b2', 'mcbch.a5r2.b2', 'mcbyh.4r2.b2', 'mcbxh.3r2',
                'mcbyh.a4l2.b2', 'mcbyh.5l2.b2', 'mcbyv.5r8.b2', 'mcbyv.a4r8.b2',
                'mcbxv.3r8', 'mcbyv.4l8.b2', 'mcbcv.b5l8.b2']:
                if nn_corr in nn_test and diff_rel < 1e-2:
                    passed_corr = True
                    break
            if not(strict) and  passed_corr:
                continue

            # Exceptions BB4D (separations are recalculated)
            if not(strict) and isinstance(ee_test, xf.BeamBeamBiGaussian2D):
                if kk == "x_bb":
                    if diff_abs / dtest["sigma_x"] < 0.01: # This is neede to accommodate different leveling routines (1% difference)
                        continue
                if kk == "y_bb":
                    if diff_abs / dtest["sigma_y"] < 0.01:
                        continue
                if kk == "sigma_x":
                    if diff_rel < 1e-5:
                        continue
                if kk == "sigma_y":
                    if diff_rel < 1e-5:
                        continue
            if isinstance(ee_test, xf.BeamBeamBiGaussian2D):
                if kk == 'q0' or kk == 'n_particles':
                    # ambiguity due to old interface
                    if np.abs(ee_test.n_particles*ee_test.q0 -
                            ee_six.n_particles*ee_six.q0 ) < 1.: # charges
                        continue

            # Exceptions BB6D (angles and separations are recalculated)
            if not(strict) and isinstance(ee_test, xf.BeamBeamBiGaussian3D):
                if kk == "alpha":
                    if diff_abs < 10e-6:
                        continue
                if kk == "x_co" or kk == "x_bb_co" or kk == 'delta_x':
                    if diff_abs / np.sqrt(dtest["sigma_11"]) < 0.015:
                        continue
                if kk == "y_co" or kk == "y_bb_co" or kk == 'delta_y':
                    if diff_abs / np.sqrt(dtest["sigma_33"]) < 0.015:
                        continue
                if kk == "zeta_co":
                    if diff_abs <1e-5:
                        continue
                if kk == "delta_co":
                    if diff_abs <1e-5:
                        continue
                if kk == "px_co" or kk == 'py_co':
                    if diff_abs <30e-9:
                        continue

            # If it got here it means that no condition above is met
            raise ValueError("Too large discrepancy!")