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

mad_with_errors = Madx(command_log="mad_final.log")
mad_with_errors.call(str(path.joinpath("final_seq.madx")))
mad_with_errors.use(sequence="lhcb1")
mad_with_errors.twiss()
mad_with_errors.readtable(file=str(path.joinpath("final_errors.tfs")),
                          table="errtab")
mad_with_errors.seterr(table="errtab")
mad_with_errors.set(format=".15g")

mad_no_errors = Madx(command_log="mad_final.log")
mad_no_errors.call(str(test_data_folder.joinpath(
                               'hllhc15_noerrors_nobb/sequence.madx')))
mad_no_errors.use(sequence="lhcb1")
mad_no_errors.globals['vrf400'] = 16
mad_no_errors.globals['lagrf400.b1'] = 0.5
mad_no_errors.twiss()

def test_twiss():

    mad = mad_with_errors
    twmad = mad.twiss(chrom=True)

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
            atol=1e-8, rtol=0)
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

            assert np.isclose(twxt['s'][ixt], twmad['s'][imad],
                              atol=5e-6, rtol=0)
            assert np.isclose(twxt['x'][ixt], twmad['x'][imad],
                              atol=5e-6, rtol=0)
            assert np.isclose(twxt['y'][ixt], twmad['y'][imad],
                              atol=5e-6, rtol=0)
            assert np.isclose(twxt['px'][ixt], twmad['px'][imad],
                              atol=1e-7, rtol=0)
            assert np.isclose(twxt['py'][ixt], twmad['py'][imad],
                              atol=1e-7, rtol=0)

        # Test custom s locations
        s_test = [2e3, 1e3, 3e3, 10e3]
        twats = tracker.twiss(at_s = s_test)
        for ii, ss in enumerate(s_test):
            assert np.isclose(twats['s'][ii], ss, rtol=0, atol=1e-14)
            assert np.isclose(twats['alfx'][ii], np.interp(ss, twxt['s'], twxt['alfx']),
                            rtol=1e-5, atol=0)
            assert np.isclose(twats['alfy'][ii], np.interp(ss, twxt['s'], twxt['alfy']),
                            rtol=1e-5, atol=0)
            assert np.isclose(twats['dpx'][ii], np.interp(ss, twxt['s'], twxt['dpx']),
                            rtol=1e-5, atol=0)
            assert np.isclose(twats['dpy'][ii], np.interp(ss, twxt['s'], twxt['dpy']),
                            rtol=1e-5, atol=0)

def norm(x):
    return np.sqrt(np.sum(np.array(x) ** 2))

def test_line_import_from_madx():

    mad = mad_with_errors

    rtol = 1e-7
    strict = True
    atol = 1e-14

    print('Build line with expressions...')
    line_with_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'],
        apply_madx_errors=True,
        install_apertures=True,
        deferred_expressions=True)
    line_with_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    print('Build line without expressions...')
    line_no_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'],
        apply_madx_errors=True,
        install_apertures=True,
        deferred_expressions=False)
    line_no_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    # Profit to test to_dict/from_dict
    print('Test to_dict/from_dict')
    line_with_expressions = xt.Line.from_dict(line_with_expressions.to_dict())

    print('Start consistency check')

    ltest = line_with_expressions
    lref = line_no_expressions

    # Check that the two machines are identical
    assert len(ltest) == len(lref)

    assert (ltest.get_length() - lref.get_length()) < 1e-6

    for ii, (ee_test, ee_six, nn_test, nn_six) in enumerate(
        zip(ltest.elements, lref.elements,
            ltest.element_names, lref.element_names)
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
            if not np.isscalar(val_ref):
                if len(val_ref) != len(val_test):
                    diff_rel = 100
                else:
                    for iiii, (vvr, vvt) in enumerate(list(zip(val_ref, val_test))):
                        if not np.isclose(vvr, vvt, atol=atol, rtol=rtol):
                            print(f'Issue found on `{kk}[{iiii}]`')
                            diff_rel = 1000
                        else:
                            diff_rel = 0
            else:
                if val_ref > 0:
                    diff_rel = np.abs((val_test - val_ref)/val_ref)
                else:
                    diff_rel = 100
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

    print('\nTest tracker and xsuite vars...\n')
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        tracker = xt.Tracker(line=line_with_expressions.copy(),
                             _context=context)
        assert np.isclose(tracker.twiss()['qx'], 62.31, rtol=0, atol=1e-4)
        tracker.vars['kqtf.b1'] = -2e-4
        assert np.isclose(tracker.twiss()['qx'], 62.2834, rtol=0, atol=1e-4)

        assert np.isclose(tracker.line.element_dict['acsca.b5l4.b1'].voltage,
                          2e6, rtol=0, atol=1e-14)
        tracker.vars['vrf400'] = 8
        assert np.isclose(tracker.line.element_dict['acsca.b5l4.b1'].voltage,
                          1e6, rtol=0, atol=1e-14)

        assert np.isclose(tracker.line.element_dict['acsca.b5l4.b1'].lag, 180,
                        rtol=0, atol=1e-14)
        tracker.vars['lagrf400.b1'] = 0.75
        assert np.isclose(tracker.line.element_dict['acsca.b5l4.b1'].lag, 270,
                        rtol=0, atol=1e-14)

        assert np.abs(
            tracker.line.element_dict['acfcav.bl5.b1'].to_dict()['ksl'][0]) > 0
        tracker.vars['on_crab5'] = 0
        assert np.abs(
            tracker.line.element_dict['acfcav.bl5.b1'].to_dict()['ksl'][0]) == 0

        assert np.isclose(
            tracker.line.element_dict['acfcav.bl5.b1'].to_dict()['ps'][0], 90,
            rtol=0, atol=1e-14)
        tracker.vars['phi_crab_l5b1'] = 0.5
        assert np.isclose(
            tracker.line.element_dict['acfcav.bl5.b1'].to_dict()['ps'][0], 270,
                        rtol=0, atol=1e-14)

        assert np.abs(
            tracker.line.element_dict['acfcah.bl1.b1'].to_dict()['knl'][0]) > 0
        tracker.vars['on_crab1'] = 0
        assert np.abs(
            tracker.line.element_dict['acfcah.bl1.b1'].to_dict()['knl'][0]) == 0

        assert np.isclose(
            tracker.line.element_dict['acfcah.bl1.b1'].to_dict()['pn'][0], 90,
            rtol=0, atol=1e-14)
        tracker.vars['phi_crab_l1b1'] = 0.5
        assert np.isclose(
            tracker.line.element_dict['acfcah.bl1.b1'].to_dict()['pn'][0], 270,
            rtol=0, atol=1e-14)

        assert np.abs(tracker.line.element_dict['acfcah.bl1.b1'].frequency) > 0
        assert np.abs(tracker.line.element_dict['acfcav.bl5.b1'].frequency) > 0
        tracker.vars['crabrf'] = 0.
        assert np.abs(tracker.line.element_dict['acfcah.bl1.b1'].frequency) == 0
        assert np.abs(tracker.line.element_dict['acfcav.bl5.b1'].frequency) == 0

def test_orbit_knobs():

    mad = mad_no_errors

    line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False,
        deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        tracker = xt.Tracker(line=line.copy(), _context=context)

        tracker.vars['on_x1'] = 250
        assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                    atol=1e-6, rtol=0)
        tracker.vars['on_x1'] = -300
        assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                    atol=1e-6, rtol=0)

        tracker.vars['on_x5'] = 130
        assert np.isclose(tracker.twiss(at_elements=['ip5'])['py'][0], 130e-6,
                    atol=1e-6, rtol=0)
        tracker.vars['on_x5'] = -270
        assert np.isclose(tracker.twiss(at_elements=['ip5'])['py'][0], -270e-6,
                    atol=1e-6, rtol=0)