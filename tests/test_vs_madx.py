# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

path = test_data_folder.joinpath('hllhc14_input_mad/')


@pytest.fixture(scope='module')
def mad_with_errors():
    mad_with_errors = Madx(stdout=False)
    mad_with_errors.call(str(path.joinpath("final_seq.madx")))
    mad_with_errors.use(sequence='lhcb1')
    mad_with_errors.twiss()
    mad_with_errors.readtable(file=str(path.joinpath("final_errors.tfs")),
                              table="errtab")
    mad_with_errors.seterr(table="errtab")
    mad_with_errors.set(format=".15g")

    mad_with_errors.sequence.lhcb1.beam.sigt = 1e-10
    mad_with_errors.sequence.lhcb1.beam.sige = 1e-10
    mad_with_errors.sequence.lhcb1.beam.et = 1e-10

    return mad_with_errors


@pytest.fixture(scope='module')
def mad_b12_no_errors():
    mad_b12_no_errors = Madx(stdout=False)
    mad_b12_no_errors.call(str(test_data_folder.joinpath(
                                   'hllhc15_noerrors_nobb/sequence.madx')))
    mad_b12_no_errors.globals['vrf400'] = 16
    mad_b12_no_errors.globals['lagrf400.b1'] = 0.5
    mad_b12_no_errors.globals['lagrf400.b2'] = 0
    mad_b12_no_errors.use(sequence='lhcb1')
    mad_b12_no_errors.twiss()
    mad_b12_no_errors.use(sequence='lhcb2')
    mad_b12_no_errors.twiss()

    return mad_b12_no_errors


@pytest.fixture(scope='module')
def mad_b4_no_errors():
    mad_b4_no_errors = Madx(stdout=False)
    mad_b4_no_errors.call(str(test_data_folder.joinpath(
                                   'hllhc15_noerrors_nobb/sequence_b4.madx')))
    mad_b4_no_errors.globals['vrf400'] = 16
    mad_b4_no_errors.globals['lagrf400.b2'] = 0
    mad_b4_no_errors.use(sequence='lhcb2')
    mad_b4_no_errors.twiss()

    return mad_b4_no_errors


surv_starting_point = {
    "theta0": -np.pi / 9, "psi0": np.pi / 7, "phi0": np.pi / 11,
    "X0": -300, "Y0": 150, "Z0": -100}


b4_b2_mapping = {
    'mbxf.4l1..1': 'mbxf.4l1..4',
    'mbxf.4l5..1': 'mbxf.4l5..4',
    'mb.b19r5.b2..1': 'mb.b19r5.b2..2',
    'mb.b19r1.b2..1': 'mb.b19r1.b2..2',
    }


@pytest.mark.parametrize('configuration', ['b1_with_errors', 'b2_no_errors'])
@for_all_test_contexts
def test_twiss_and_survey(
        test_context,
        mad_with_errors,
        mad_b4_no_errors,
        mad_b12_no_errors,
        configuration,
):
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

    line_full.line = None
    line_simplified.tracker = None

    line_full.build_tracker(_context=test_context)
    assert line_full.iscollective

    line_simplified.build_tracker(_context=test_context)

    for simplified, line in zip((False, True), [line_full, line_simplified]):

        print(f"Simplified: {simplified}")

        twxt = line.twiss()
        twxt4d = line.twiss(method='4d')
        survxt = line.survey(**surv_starting_point)

        if reverse:
            twxt = twxt.reverse()
            twxt4d = twxt4d.reverse()
            survxt = survxt.reverse()

        assert len(twxt.name) == len(line.element_names) + 1

        # Check value_at_element_exit (not implemented for now, to be reintroduced)
        # if not reverse: # TODO: to be generalized...
        #     twxt_exit = line.twiss(values_at_element_exit=True)
        #     for nn in['s', 'x','px','y','py', 'zeta','delta','ptau',
        #             'betx','bety','alfx','alfy','gamx','gamy','dx','dpx','dy',
        #             'dpy','mux','muy', 'name']:
        #         assert np.all(twxt[nn][1:] == twxt_exit[nn])

        # Twiss a part of the machine
        tw_init = line.twiss().get_twiss_init(
                                    at_element=range_for_partial_twiss[0])
        tw4d_init = line.twiss(method='4d').get_twiss_init(
                                    at_element=range_for_partial_twiss[0])
        tw_part = line.twiss(start=range_for_partial_twiss[0],
                end=range_for_partial_twiss[1], init=tw_init)
        tw4d_part = line.twiss(method='4d',
                start=range_for_partial_twiss[0],
                end=range_for_partial_twiss[1], init=tw4d_init)
        if reverse:
            tw_part = tw_part.reverse()
            tw4d_part = tw4d_part.reverse()

        ipart_start = line.element_names.index(range_for_partial_twiss[0])
        ipart_stop = line.element_names.index(range_for_partial_twiss[1])
        assert len(tw_part.name) == ipart_stop - ipart_start + 2
        assert len(tw4d_part.name) == ipart_stop - ipart_start + 2

        # Check against mad
        for twtst in [twxt, twxt4d]:
            xo.assert_allclose(mad_ref.table.summ.q1[0], twtst['qx'], rtol=1e-4, atol=0)
            xo.assert_allclose(mad_ref.table.summ.q2[0], twtst['qy'], rtol=1e-4, atol=0)
            xo.assert_allclose(mad_ref.table.summ.dq1, twtst['dqx'], atol=0.1, rtol=0)
            xo.assert_allclose(mad_ref.table.summ.dq2, twtst['dqy'], atol=0.1, rtol=0)
            xo.assert_allclose(mad_ref.table.summ.alfa[0],
                twtst['momentum_compaction_factor'], atol=7e-8, rtol=0)
            if twtst is not tw4d_part:
                xo.assert_allclose(twxt['qs'], 0.0021, atol=1e-4, rtol=0)


        for is_part, twtst in zip([False, False, True, True],
                                   [twxt, twxt4d, tw_part, tw4d_part]):

            nemitt_x = mad_ref.sequence[seq_name].beam.exn
            nemitt_y = mad_ref.sequence[seq_name].beam.eyn
            Sigmas = twtst.get_betatron_sigmas(nemitt_x, nemitt_y)

            for nn in twtst._col_names:
                assert len(twtst[nn]) == len(twtst['name'])

            test_at_elements = []
            test_at_elements.extend(['mbxf.4l1..1', 'mbxf.4l5..1'])

            if seq_name.endswith('b1'):
                test_at_elements.extend(['mb.b19r5.b1..1', 'mb.b19r1.b1..1'])
            elif seq_name.endswith('b2'):
                test_at_elements.extend(['mb.b19r5.b2..1', 'mb.b19r1.b2..1'])

            if line is line_full:
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

                xo.assert_allclose(twtst['s'][ixt], twmad['s'][imad],
                                atol=1e-6, rtol=0)
                xo.assert_allclose(twtst['betx'][ixt], twmad['betx'][imad],
                                atol=0, rtol=7e-4)
                xo.assert_allclose(twtst['bety'][ixt], twmad['bety'][imad],
                                atol=0, rtol=7e-4)
                xo.assert_allclose(twtst['alfx'][ixt], twmad['alfx'][imad],
                                atol=1e-1, rtol=0)
                xo.assert_allclose(twtst['alfy'][ixt], twmad['alfy'][imad],
                                atol=1e-1, rtol=0)
                xo.assert_allclose(twtst['dx'][ixt], twmad['dx'][imad],
                                atol=1e-2, rtol=0)
                xo.assert_allclose(twtst['dy'][ixt], twmad['dy'][imad],
                                atol=1e-2, rtol=0)
                xo.assert_allclose(twtst['dpx'][ixt], twmad['dpx'][imad],
                                atol=3e-4, rtol=0)
                xo.assert_allclose(twtst['dpy'][ixt], twmad['dpy'][imad],
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
                xo.assert_allclose(twtst['mux'][ixt] - mux0_tst,
                                  twmad['mux'][imad] - mux0_mad,
                                  atol=1e-4, rtol=0)
                xo.assert_allclose(twtst['muy'][ixt] - muy0_tst,
                                  twmad['muy'][imad] - muy0_mad,
                                  atol=1e-4, rtol=0)

                xo.assert_allclose(twtst['s'][ixt], twmad['s'][imad],
                                atol=5e-6, rtol=0)

                # I check the orbit relative to sigma to be more accurate at the IP
                sigx = np.sqrt(twmad['sig11'][imad])
                sigy = np.sqrt(twmad['sig33'][imad])

                xo.assert_allclose(twtst['x'][ixt], twmad['x'][imad],
                                atol=0.03*sigx, rtol=0)
                xo.assert_allclose(twtst['y'][ixt],
                                (twmad['y'][imad]),
                                atol=0.03*sigy, rtol=0)

                xo.assert_allclose(twtst['px'][ixt], twmad['px'][imad],
                                atol=2e-7, rtol=0)
                xo.assert_allclose(twtst['py'][ixt], twmad['py'][imad],
                                atol=2e-7, rtol=0)

                xo.assert_allclose(Sigmas.Sigma11[ixt], twmad['sig11'][imad], atol=5e-10)
                xo.assert_allclose(Sigmas.Sigma12[ixt], twmad['sig12'][imad], atol=3e-12)
                xo.assert_allclose(Sigmas.Sigma22[ixt], twmad['sig22'][imad], atol=1e-12)
                xo.assert_allclose(Sigmas.Sigma33[ixt], twmad['sig33'][imad], atol=5e-10)
                xo.assert_allclose(Sigmas.Sigma34[ixt], twmad['sig34'][imad], rtol=1e-4, atol=3e-12)
                xo.assert_allclose(Sigmas.Sigma44[ixt], twmad['sig44'][imad], atol=1e-12)

                if twtst not in [twxt4d, tw4d_part]: # 4d less precise due to different momentum (coupling comes from feeddown)
                    xo.assert_allclose(Sigmas.Sigma13[ixt], twmad['sig13'][imad], atol=5e-10)
                    xo.assert_allclose(Sigmas.Sigma14[ixt], twmad['sig14'][imad], atol=2e-12)
                    xo.assert_allclose(Sigmas.Sigma23[ixt], twmad['sig23'][imad], atol=1e-12)
                    xo.assert_allclose(Sigmas.Sigma24[ixt], twmad['sig24'][imad], atol=1e-12)

                # check matrix is symmetric
                xo.assert_allclose(Sigmas.Sigma12[ixt], Sigmas.Sigma21[ixt], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma13[ixt], Sigmas.Sigma31[ixt], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma14[ixt], Sigmas.Sigma41[ixt], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma23[ixt], Sigmas.Sigma32[ixt], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma24[ixt], Sigmas.Sigma42[ixt], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma34[ixt], Sigmas.Sigma43[ixt], atol=1e-16)

                # check matrix consistency with Sigma.Sigma
                xo.assert_allclose(Sigmas.Sigma11[ixt], Sigmas.Sigma[ixt][0, 0], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma12[ixt], Sigmas.Sigma[ixt][0, 1], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma13[ixt], Sigmas.Sigma[ixt][0, 2], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma14[ixt], Sigmas.Sigma[ixt][0, 3], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma21[ixt], Sigmas.Sigma[ixt][1, 0], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma22[ixt], Sigmas.Sigma[ixt][1, 1], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma23[ixt], Sigmas.Sigma[ixt][1, 2], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma24[ixt], Sigmas.Sigma[ixt][1, 3], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma31[ixt], Sigmas.Sigma[ixt][2, 0], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma32[ixt], Sigmas.Sigma[ixt][2, 1], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma33[ixt], Sigmas.Sigma[ixt][2, 2], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma34[ixt], Sigmas.Sigma[ixt][2, 3], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma41[ixt], Sigmas.Sigma[ixt][3, 0], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma42[ixt], Sigmas.Sigma[ixt][3, 1], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma43[ixt], Sigmas.Sigma[ixt][3, 2], atol=1e-16)
                xo.assert_allclose(Sigmas.Sigma44[ixt], Sigmas.Sigma[ixt][3, 3], atol=1e-16)

                # Check sigma_x, sigma_y
                xo.assert_allclose(Sigmas.sigma_x[ixt], np.sqrt(Sigmas.Sigma11[ixt]), atol=1e-16)
                xo.assert_allclose(Sigmas.sigma_y[ixt], np.sqrt(Sigmas.Sigma33[ixt]), atol=1e-16)
                xo.assert_allclose(Sigmas.sigma_px[ixt], np.sqrt(Sigmas.Sigma22[ixt]), atol=1e-16)
                xo.assert_allclose(Sigmas.sigma_py[ixt], np.sqrt(Sigmas.Sigma44[ixt]), atol=1e-16)

                if not(is_part): # We don't have survey on a part of the machine
                    # Check survey
                    xo.assert_allclose(survxt.X[ixt], survmad['X'][imad], atol=1e-6)
                    xo.assert_allclose(survxt.Y[ixt], survmad['Y'][imad], rtol=2e-7, atol=1e-6)
                    xo.assert_allclose(survxt.Z[ixt], survmad['Z'][imad], atol=1e-6)
                    xo.assert_allclose(survxt.s[ixt], survmad['s'][imad], atol=5e-6)
                    xo.assert_allclose(survxt.phi[ixt], survmad['phi'][imad], atol=1e-10)
                    xo.assert_allclose(survxt.theta[ixt], survmad['theta'][imad], atol=1e-10)
                    xo.assert_allclose(survxt.psi[ixt], survmad['psi'][imad], atol=1e-10)

                    # angle and tilt are associated to the element itself (ixt - 1)
                    # For now not checking the sign of the angles, convetion in mad-X to be calrified
                    xo.assert_allclose(np.abs(survxt.angle[ixt-1]),
                            np.abs(survmad['angle'][imad]), atol=1e-10)
                    xo.assert_allclose(survxt.tilt[ixt-1], survmad['tilt'][imad], atol=1e-10)

        # Check to_pandas (not extensively for now)
        dftw = twtst.to_pandas()
        dfsurv = survxt.to_pandas()
        assert np.all(dftw['s'] == twtst['s'])
        assert np.all(dfsurv['s'] == survxt['s'])

        # Test custom s locations
        if not reverse:
            s_test = [2e3, 1e3, 3e3, 10e3]
            twats = line.twiss(at_s = s_test)
            for ii, ss in enumerate(s_test):
                xo.assert_allclose(twats['s'][ii], ss, rtol=0, atol=1e-14)
                xo.assert_allclose(twats['alfx'][ii], np.interp(ss, twxt['s'], twxt['alfx']),
                                rtol=1e-5, atol=0)
                xo.assert_allclose(twats['alfy'][ii], np.interp(ss, twxt['s'], twxt['alfy']),
                                rtol=1e-5, atol=0)
                xo.assert_allclose(twats['dpx'][ii], np.interp(ss, twxt['s'], twxt['dpx']),
                                rtol=1e-5, atol=0)
                xo.assert_allclose(twats['dpy'][ii], np.interp(ss, twxt['s'], twxt['dpy']),
                                rtol=1e-5, atol=0)


def norm(x):
    return np.sqrt(np.sum(np.array(x) ** 2))


@for_all_test_contexts
def test_line_import_from_madx(test_context, mad_with_errors):

    mad = mad_with_errors

    rtol = 1e-7
    atol = 1e-14

    print('Build line with expressions...')
    line_with_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'],
        apply_madx_errors=True,
        install_apertures=True,
        deferred_expressions=True,
    )
    line_with_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    print('Build line without expressions...')
    line_no_expressions = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'],
        apply_madx_errors=True,
        install_apertures=True,
        deferred_expressions=False,
    )
    line_no_expressions.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                        q0=1, gamma0=mad.sequence.lhcb1.beam.gamma)

    # Profit to test to_dict/from_dict
    print('Test to_dict/from_dict')
    line_with_expressions = xt.Line.from_dict(line_with_expressions.to_dict())

    print('Start consistency check')

    ltest = line_with_expressions
    lref = line_no_expressions

    ltest.merge_consecutive_drifts()
    lref.merge_consecutive_drifts()

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

        ee_test_cpu = ee_test.copy(_context=xo.ContextCpu())
        ee_six_cpu = ee_six.copy(_context=xo.ContextCpu())

        skip_order = False
        if isinstance(ee_test, xt.Multipole):
            if ee_test._order != ee_six._order:
                min_order = min(ee_test._order, ee_six._order)
                xo.assert_allclose(
                    ee_test.knl[:min_order+1],
                    ee_six.knl[:min_order+1],
                    atol=1e-16,
                )
                xo.assert_allclose(
                    ee_test.ksl[:min_order+1],
                    ee_six.ksl[:min_order+1],
                    atol=1e-16,
                )
                xo.assert_allclose(ee_test.knl[min_order+1:], 0, atol=1e-16)
                xo.assert_allclose(ee_test.ksl[min_order+1:], 0, atol=1e-16)
                xo.assert_allclose(ee_six.knl[min_order+1:], 0, atol=1e-16)
                xo.assert_allclose(ee_six.ksl[min_order+1:], 0, atol=1e-16)

                skip_order = True

        for kk in dtest.keys():

            if skip_order and kk in ('order', 'knl', 'ksl'):
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
    line = line_with_expressions.copy()
    line.build_tracker(_context=test_context)
    xo.assert_allclose(line.twiss()['qx'], 62.31, rtol=0, atol=1e-4)
    line.vars['kqtf.b1'] = -2e-4
    xo.assert_allclose(line.twiss()['qx'], 62.2834, rtol=0, atol=1e-4)

    xo.assert_allclose(line.element_dict['acsca.b5l4.b1'].voltage,
                      2e6, rtol=0, atol=1e-14)
    line.vars['vrf400'] = 8
    xo.assert_allclose(line.element_dict['acsca.b5l4.b1'].voltage,
                      1e6, rtol=0, atol=1e-14)

    xo.assert_allclose(line.element_dict['acsca.b5l4.b1'].lag, 180,
                    rtol=0, atol=1e-14)
    line.vars['lagrf400.b1'] = 0.75
    xo.assert_allclose(line.element_dict['acsca.b5l4.b1'].lag, 270,
                    rtol=0, atol=1e-14)

    assert np.abs(
        line.element_dict['acfcav.bl5.b1'].to_dict()['ksl'][0]) > 0
    line.vars['on_crab5'] = 0
    assert np.abs(
        line.element_dict['acfcav.bl5.b1'].to_dict()['ksl'][0]) == 0

    xo.assert_allclose(
        line.element_dict['acfcav.bl5.b1'].to_dict()['ps'][0], 90,
        rtol=0, atol=1e-14)
    line.vars['phi_crab_l5b1'] = 0.5
    xo.assert_allclose(
        line.element_dict['acfcav.bl5.b1'].to_dict()['ps'][0], 270,
                    rtol=0, atol=1e-14)

    assert np.abs(
        line.element_dict['acfcah.bl1.b1'].to_dict()['knl'][0]) > 0
    line.vars['on_crab1'] = 0
    assert np.abs(
        line.element_dict['acfcah.bl1.b1'].to_dict()['knl'][0]) == 0

    xo.assert_allclose(
        line.element_dict['acfcah.bl1.b1'].to_dict()['pn'][0], 90,
        rtol=0, atol=1e-14)
    line.vars['phi_crab_l1b1'] = 0.5
    xo.assert_allclose(
        line.element_dict['acfcah.bl1.b1'].to_dict()['pn'][0], 270,
        rtol=0, atol=1e-14)

    assert np.abs(line.element_dict['acfcah.bl1.b1'].frequency) > 0
    assert np.abs(line.element_dict['acfcav.bl5.b1'].frequency) > 0
    line.vars['crabrf'] = 0.
    assert np.abs(line.element_dict['acfcah.bl1.b1'].frequency) == 0
    assert np.abs(line.element_dict['acfcav.bl5.b1'].frequency) == 0


@for_all_test_contexts
def test_orbit_knobs(test_context, mad_b12_no_errors):

    mad = mad_b12_no_errors

    line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False,
        deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

    line.build_tracker(_context=test_context)

    line.vars['on_x1'] = 250
    xo.assert_allclose(line.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                atol=1e-6, rtol=0)
    line.vars['on_x1'] = -300
    xo.assert_allclose(line.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                atol=1e-6, rtol=0)

    line.vars['on_x5'] = 130
    xo.assert_allclose(line.twiss(at_elements=['ip5'])['py'][0], 130e-6,
                atol=1e-6, rtol=0)
    line.vars['on_x5'] = -270
    xo.assert_allclose(line.twiss(at_elements=['ip5'])['py'][0], -270e-6,
                atol=1e-6, rtol=0)


@for_all_test_contexts
def test_low_beta_twiss(test_context):

    path_line = test_data_folder / 'psb_injection/line_and_particle.json'

    line = xt.Line.from_json(path_line)
    line.build_tracker(_context=test_context)
    tw = line.twiss()

    path_madseq = test_data_folder / 'psb_injection/psb_injection.seq'

    mad = Madx(stdout=False)
    mad.call(str(path_madseq))

    mad.use(sequence='psb')
    mad.twiss()
    mad.emit()

    emitdf = mad.table.emitsumm.dframe()

    xo.assert_allclose(mad.sequence.psb.beam.gamma, line.particle_ref.gamma0,
                      rtol=0, atol=1e-6)
    xo.assert_allclose(mad.sequence.psb.beam.beta, line.particle_ref.beta0,
                    rtol=0, atol=1e-10)

    beta0 = line.particle_ref.beta0

    xo.assert_allclose(mad.table.summ['q1'][0], tw['qx'], rtol=0, atol=1e-6)
    xo.assert_allclose(mad.table.summ['q2'][0], tw['qy'], rtol=0, atol=1e-6)
    xo.assert_allclose(mad.table.summ['dq1'][0]*beta0, tw['dqx'], rtol=0,
                        atol=1e-3)
    xo.assert_allclose(mad.table.summ['dq2'][0]*beta0, tw['dqy'], rtol=0,
                        atol=1e-3)
    xo.assert_allclose(tw.qs, emitdf.qs.iloc[0], rtol=0, atol=1e-8)
