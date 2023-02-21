# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

from cpymad.madx import Madx


test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

path = test_data_folder.joinpath('hllhc14_input_mad/')

mad_with_errors = Madx()
mad_with_errors.call(str(path.joinpath("final_seq.madx")))
mad_with_errors.use(sequence='lhcb1')
mad_with_errors.twiss()
mad_with_errors.readtable(file=str(path.joinpath("final_errors.tfs")),
                          table="errtab")
mad_with_errors.seterr(table="errtab")
mad_with_errors.set(format=".15g")

mad_b12_no_errors = Madx()
mad_b12_no_errors.call(str(test_data_folder.joinpath(
                               'hllhc15_noerrors_nobb/sequence.madx')))
mad_b12_no_errors.globals['vrf400'] = 16
mad_b12_no_errors.globals['lagrf400.b1'] = 0.5
mad_b12_no_errors.globals['lagrf400.b2'] = 0
mad_b12_no_errors.use(sequence='lhcb1')
mad_b12_no_errors.twiss()
mad_b12_no_errors.use(sequence='lhcb2')
mad_b12_no_errors.twiss()

mad_b4_no_errors = Madx()
mad_b4_no_errors.call(str(test_data_folder.joinpath(
                               'hllhc15_noerrors_nobb/sequence_b4.madx')))
mad_b4_no_errors.globals['vrf400'] = 16
mad_b4_no_errors.globals['lagrf400.b2'] = 0
mad_b4_no_errors.use(sequence='lhcb2')
mad_b4_no_errors.twiss()

mad_with_errors.sequence.lhcb1.beam.sigt = 1e-10
mad_with_errors.sequence.lhcb1.beam.sige = 1e-10
mad_with_errors.sequence.lhcb1.beam.et = 1e-10

surv_starting_point = {
    "theta0": -np.pi / 9, "psi0": np.pi / 7, "phi0": np.pi / 11,
    "X0": -300, "Y0": 150, "Z0": -100}

b4_b2_mapping = {
    'mbxf.4l1..1': 'mbxf.4l1..4',
    'mbxf.4l5..1': 'mbxf.4l5..4',
    'mb.b19r5.b2..1': 'mb.b19r5.b2..2',
    'mb.b19r1.b2..1': 'mb.b19r1.b2..2',
    }

@for_all_test_contexts
def test_twiss_and_survey(test_context):

    for configuration in ['b1_with_errors', 'b2_no_errors']:

        print(f"Test configuration: {configuration}")

        if configuration == 'b1_with_errors':
            mad_load = mad_with_errors
            mad_ref = mad_with_errors
            ref_element_for_mu = 'acsca.d5l4.b1'
            seq_name = 'lhcb1'
            reverse = False
            use = False
            range_for_partial_twiss = ('mb.b19r3.b1..1', 'mb.b19l3.b1..1')
        elif configuration == 'b2_no_errors':
            mad_load = mad_b4_no_errors
            mad_ref = mad_b12_no_errors
            ref_element_for_mu = 'acsca.d5r4.b2'
            seq_name = 'lhcb2'
            reverse = True
            use = True
            range_for_partial_twiss = ('mb.b19l3.b2..1', 'mb.b19r3.b2..1')

        if use:
            mad_ref.use(sequence=seq_name)
            mad_load.use(sequence=seq_name)

        # I want only the betatron part in the sigma matrix
        mad_ref.sequence[seq_name].beam.sigt = 1e-10
        mad_ref.sequence[seq_name].beam.sige = 1e-10
        mad_ref.sequence[seq_name].beam.et = 1e-10

        # I set asymmetric emittances
        mad_ref.sequence[seq_name].beam.exn = 2.5e-6
        mad_ref.sequence[seq_name].beam.eyn = 3.5e-6

        twmad = mad_ref.twiss(chrom=True)
        survmad = mad_ref.survey(**surv_starting_point)

        line_full = xt.Line.from_madx_sequence(
                mad_load.sequence[seq_name], apply_madx_errors=True)
        line_full.elements[10].iscollective = True # we make it artificially collective to test this option
        line_full.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                                gamma0=mad_load.sequence[seq_name].beam.gamma)

        # Test twiss also on simplified line
        line_simplified = line_full.copy()

        print('Simplifying line...')
        line_simplified.remove_inactive_multipoles()
        line_simplified.merge_consecutive_multipoles()
        line_simplified.remove_zero_length_drifts()
        line_simplified.merge_consecutive_drifts()
        line_simplified.use_simple_bends()
        line_simplified.use_simple_quadrupoles()
        print('Done simplifying line')

        line_full.tracker = None
        line_simplified.tracker = None

        tracker_full = xt.Tracker(_context=test_context, line=line_full)
        assert tracker_full.iscollective

        tracker_simplified = line_simplified.build_tracker(_context=test_context)

        for simplified, tracker in zip((False, True), [tracker_full, tracker_simplified]):

            print(f"Simplified: {simplified}")

            twxt = tracker.twiss(reverse=reverse)
            twxt4d = tracker.twiss(method='4d', reverse=reverse)
            survxt = tracker.survey(**surv_starting_point, reverse=reverse)

            assert len(twxt.name) == len(tracker.line.element_names) + 1

            # Check value_at_element_exit
            if not reverse: # TODO: to be generalized...
                twxt_exit = tracker.twiss(values_at_element_exit=True)
                for nn in['s', 'x','px','y','py', 'zeta','delta','ptau',
                        'betx','bety','alfx','alfy','gamx','gamy','dx','dpx','dy',
                        'dpy','mux','muy', 'name']:
                    assert np.all(twxt[nn][1:] == twxt_exit[nn])

            # Twiss a part of the machine
            tw_init = tracker.twiss().get_twiss_init(at_element=range_for_partial_twiss[0])
            tw4d_init = tracker.twiss(method='4d').get_twiss_init(at_element=range_for_partial_twiss[0])
            tw_part = tracker.twiss(ele_start=range_for_partial_twiss[0],
                                    ele_stop=range_for_partial_twiss[1], twiss_init=tw_init, reverse=reverse)
            tw4d_part = tracker.twiss(method='4d', ele_start=range_for_partial_twiss[0],
                                    ele_stop=range_for_partial_twiss[1], twiss_init=tw4d_init, reverse=reverse)

            ipart_start = tracker.line.element_names.index(range_for_partial_twiss[0])
            ipart_stop = tracker.line.element_names.index(range_for_partial_twiss[1])
            assert len(tw_part.name) == ipart_stop - ipart_start + 1
            assert len(tw4d_part.name) == ipart_stop - ipart_start + 1

            # Check against mad
            for twtst in [twxt, twxt4d]:
                assert np.isclose(mad_ref.table.summ.q1[0], twtst['qx'], rtol=1e-4, atol=0)
                assert np.isclose(mad_ref.table.summ.q2[0], twtst['qy'], rtol=1e-4, atol=0)
                assert np.isclose(mad_ref.table.summ.dq1, twtst['dqx'], atol=0.1, rtol=0)
                assert np.isclose(mad_ref.table.summ.dq2, twtst['dqy'], atol=0.1, rtol=0)
                assert np.isclose(mad_ref.table.summ.alfa[0],
                    twtst['momentum_compaction_factor'], atol=7e-8, rtol=0)
                if twtst is not tw4d_part:
                    assert np.isclose(twxt['qs'], 0.0021, atol=1e-4, rtol=0)


            for is_part, twtst in zip([False, False, True, True],
                                       [twxt, twxt4d, tw_part, tw4d_part]):

                nemitt_x = mad_ref.sequence[seq_name].beam.exn
                nemitt_y = mad_ref.sequence[seq_name].beam.eyn
                Sigmas = twtst.get_betatron_sigmas(nemitt_x, nemitt_y)

                for nn in twtst._ebe_fields:
                    assert len(twtst[nn]) == len(twtst['name'])

                test_at_elements = []
                test_at_elements.extend(['mbxf.4l1..1', 'mbxf.4l5..1'])

                if seq_name.endswith('b1'):
                    test_at_elements.extend(['mb.b19r5.b1..1', 'mb.b19r1.b1..1'])
                elif seq_name.endswith('b2'):
                    test_at_elements.extend(['mb.b19r5.b2..1', 'mb.b19r1.b2..1'])

                if tracker is tracker_full:
                    test_at_elements += ['ip1', 'ip2', 'ip5', 'ip8']

                for name in test_at_elements:

                    if reverse:
                        name_mad = b4_b2_mapping.get(name, name)
                    else:
                        name_mad = name

                    imad = list(twmad['name']).index(name_mad+':1')
                    ixt = list(twtst['name']).index(name) + 1 # MAD measures at exit

                    eemad = mad_ref.sequence[seq_name].expanded_elements[name]

                    mad_shift_x = eemad.align_errors.dx if eemad.align_errors else 0
                    mad_shift_y = eemad.align_errors.dy if eemad.align_errors else 0

                    assert np.isclose(twtst['betx'][ixt], twmad['betx'][imad],
                                    atol=0, rtol=7e-4)
                    assert np.isclose(twtst['bety'][ixt], twmad['bety'][imad],
                                    atol=0, rtol=7e-4)
                    assert np.isclose(twtst['alfx'][ixt], twmad['alfx'][imad],
                                    atol=1e-1, rtol=0)
                    assert np.isclose(twtst['alfy'][ixt], twmad['alfy'][imad],
                                    atol=1e-1, rtol=0)
                    assert np.isclose(twtst['dx'][ixt], twmad['dx'][imad],
                                    atol=1e-2, rtol=0)
                    assert np.isclose(twtst['dy'][ixt], twmad['dy'][imad],
                                    atol=1e-2, rtol=0)
                    assert np.isclose(twtst['dpx'][ixt], twmad['dpx'][imad],
                                    atol=3e-4, rtol=0)
                    assert np.isclose(twtst['dpy'][ixt], twmad['dpy'][imad],
                                    atol=3e-4, rtol=0)

                    if is_part:
                        # I chck the phase advance w.r.t. ip1
                        mux0_mad = twmad['mux'][list(twmad.name).index(ref_element_for_mu + ':1')]
                        muy0_mad = twmad['muy'][list(twmad.name).index(ref_element_for_mu + ':1')]
                        mux0_tst = twtst['mux'][list(twtst.name).index(ref_element_for_mu)]
                        muy0_tst = twtst['muy'][list(twtst.name).index(ref_element_for_mu)]
                    else:
                        # I check the absolute phase advance
                        mux0_mad = 0
                        muy0_mad = 0
                        mux0_tst = 0
                        muy0_tst = 0
                    assert np.isclose(twtst['mux'][ixt] - mux0_tst,
                                      twmad['mux'][imad] - mux0_mad,
                                      atol=1e-4, rtol=0)
                    assert np.isclose(twtst['muy'][ixt] - muy0_tst,
                                      twmad['muy'][imad] - muy0_mad,
                                      atol=1e-4, rtol=0)

                    assert np.isclose(twtst['s'][ixt], twmad['s'][imad],
                                    atol=5e-6, rtol=0)

                    # I check the orbit relative to sigma to be more accurate at the IP
                    sigx = np.sqrt(twmad['sig11'][imad])
                    sigy = np.sqrt(twmad['sig33'][imad])

                    assert np.isclose(twtst['x'][ixt], (twmad['x'][imad] - mad_shift_x),
                                    atol=0.03*sigx, rtol=0)
                    assert np.isclose(twtst['y'][ixt],
                                    (twmad['y'][imad] - mad_shift_y),
                                    atol=0.03*sigy, rtol=0)

                    assert np.isclose(twtst['px'][ixt], twmad['px'][imad],
                                    atol=2e-7, rtol=0)
                    assert np.isclose(twtst['py'][ixt], twmad['py'][imad],
                                    atol=2e-7, rtol=0)

                    assert np.isclose(Sigmas.Sigma11[ixt], twmad['sig11'][imad], atol=5e-10)
                    assert np.isclose(Sigmas.Sigma12[ixt], twmad['sig12'][imad], atol=3e-12)
                    assert np.isclose(Sigmas.Sigma13[ixt], twmad['sig13'][imad], atol=2e-10)
                    assert np.isclose(Sigmas.Sigma14[ixt], twmad['sig14'][imad], atol=2e-12)
                    assert np.isclose(Sigmas.Sigma22[ixt], twmad['sig22'][imad], atol=1e-12)
                    assert np.isclose(Sigmas.Sigma23[ixt], twmad['sig23'][imad], atol=1e-12)
                    assert np.isclose(Sigmas.Sigma24[ixt], twmad['sig24'][imad], atol=1e-12)
                    assert np.isclose(Sigmas.Sigma33[ixt], twmad['sig33'][imad], atol=5e-10)
                    assert np.isclose(Sigmas.Sigma34[ixt], twmad['sig34'][imad], atol=3e-12)
                    assert np.isclose(Sigmas.Sigma44[ixt], twmad['sig44'][imad], atol=1e-12)

                    # check matrix is symmetric
                    assert np.isclose(Sigmas.Sigma12[ixt], Sigmas.Sigma21[ixt], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma13[ixt], Sigmas.Sigma31[ixt], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma14[ixt], Sigmas.Sigma41[ixt], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma23[ixt], Sigmas.Sigma32[ixt], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma24[ixt], Sigmas.Sigma42[ixt], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma34[ixt], Sigmas.Sigma43[ixt], atol=1e-16)

                    # check matrix consistency with Sigma.Sigma
                    assert np.isclose(Sigmas.Sigma11[ixt], Sigmas.Sigma[ixt][0, 0], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma12[ixt], Sigmas.Sigma[ixt][0, 1], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma13[ixt], Sigmas.Sigma[ixt][0, 2], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma14[ixt], Sigmas.Sigma[ixt][0, 3], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma21[ixt], Sigmas.Sigma[ixt][1, 0], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma22[ixt], Sigmas.Sigma[ixt][1, 1], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma23[ixt], Sigmas.Sigma[ixt][1, 2], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma24[ixt], Sigmas.Sigma[ixt][1, 3], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma31[ixt], Sigmas.Sigma[ixt][2, 0], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma32[ixt], Sigmas.Sigma[ixt][2, 1], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma33[ixt], Sigmas.Sigma[ixt][2, 2], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma34[ixt], Sigmas.Sigma[ixt][2, 3], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma41[ixt], Sigmas.Sigma[ixt][3, 0], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma42[ixt], Sigmas.Sigma[ixt][3, 1], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma43[ixt], Sigmas.Sigma[ixt][3, 2], atol=1e-16)
                    assert np.isclose(Sigmas.Sigma44[ixt], Sigmas.Sigma[ixt][3, 3], atol=1e-16)

                    # Check sigma_x, sigma_y
                    assert np.isclose(Sigmas.sigma_x[ixt], np.sqrt(Sigmas.Sigma11[ixt]), atol=1e-16)
                    assert np.isclose(Sigmas.sigma_y[ixt], np.sqrt(Sigmas.Sigma33[ixt]), atol=1e-16)

                    if not(is_part): # We don't have survey on a part of the machine
                        # Check survey
                        assert np.isclose(survxt.X[ixt], survmad['X'][imad], atol=1e-6)
                        assert np.isclose(survxt.Y[ixt], survmad['Y'][imad], atol=1e-6)
                        assert np.isclose(survxt.Z[ixt], survmad['Z'][imad], atol=1e-6)
                        assert np.isclose(survxt.s[ixt], survmad['s'][imad], atol=5e-6)
                        assert np.isclose(survxt.phi[ixt], survmad['phi'][imad], atol=1e-10)
                        assert np.isclose(survxt.theta[ixt], survmad['theta'][imad], atol=1e-10)
                        assert np.isclose(survxt.psi[ixt], survmad['psi'][imad], atol=1e-10)

                        # angle and tilt are associated to the element itself (ixt - 1)
                        # For now not checking the sign of the angles, convetion in mad-X to be calrified
                        assert np.isclose(np.abs(survxt.angle[ixt-1]),
                                np.abs(survmad['angle'][imad]), atol=1e-10)
                        assert np.isclose(survxt.tilt[ixt-1], survmad['tilt'][imad], atol=1e-10)

            # Check to_pandas (not extensively for now)
            dftw = twtst.to_pandas()
            dfsurv = survxt.to_pandas()
            assert np.all(dftw['s'] == twtst['s'])
            assert np.all(dfsurv['s'] == survxt['s'])

            # Test custom s locations
            if not reversed:
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


@for_all_test_contexts
def test_line_import_from_madx(test_context):

    mad = mad_with_errors

    rtol = 1e-7
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

        skip_order = False
        if isinstance(ee_test, xt.Multipole):
            if ee_test.order != ee_six.order:
                min_order = min(ee_test.order, ee_six.order)
                if len(dtest['knl']) > min_order+1:
                    assert np.all(dtest['knl'][min_order+1]  == 0)
                    dtest['knl'] = dtest['knl'][:min_order+1]
                if len(dref['knl']) > min_order+1:
                    assert np.all(dref['knl'][min_order+1]  == 0)
                    dref['knl'] = dref['knl'][:min_order+1]
                if len(dtest['ksl']) > min_order+1:
                    assert np.all(dtest['ksl'][min_order+1]  == 0)
                    dtest['ksl'] = dtest['ksl'][:min_order+1]
                if len(dref['ksl']) > min_order+1:
                    assert np.all(dref['ksl'][min_order+1]  == 0)
                    dref['ksl'] = dref['ksl'][:min_order+1]
                skip_order = True

        for kk in dtest.keys():

            if skip_order and kk == 'order':
                continue

            if skip_order and kk == 'inv_factorial_order':
                continue

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
    tracker = xt.Tracker(line=line_with_expressions.copy(),
                         _context=test_context)
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


@for_all_test_contexts
def test_orbit_knobs(test_context):

    mad = mad_b12_no_errors

    line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False,
        deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

    tracker = xt.Tracker(line=line.copy(), _context=test_context)

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
