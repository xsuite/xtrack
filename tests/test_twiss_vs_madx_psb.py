import pathlib

import numpy as np
from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from scipy.constants import c as clight

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_twiss_psb(test_context):

    mad = Madx(stdout=False)
    mad.call(str(test_data_folder / 'psb_injection/psb_injection.seq'))
    mad.use('psb')
    twmad = mad.twiss()

    line_ref = xt.Line.from_madx_sequence(mad.sequence['psb'])
    line_ref.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1.,
                                    gamma0=mad.sequence['psb'].beam.gamma)

    env = xt.load(test_data_folder / 'psb_injection/psb_injection.seq')
    line = env['psb']
    line.set_particle_ref('proton', gamma0=mad.sequence['psb'].beam.gamma)

    line.build_tracker(_context=test_context)
    beta0 = line.particle_ref.beta0[0]

    # Check consistency of element parameters
    lref = line_ref
    ltest = line

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
        assert nn_test.split(':')[0] == nn_ref.split(':')[0], f'Element name mismatch: {nn_test} != {nn_ref}'

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

            if kk == '_isthick' and eref.length == 0:
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


    tw = line.twiss()

    # With the approximation beta ~= beta0 we have delta ~= pzeta ~= 1/beta0 ptau
    # ==> ptau ~= beta0 delta ==> dptau / ddelta~= beta0

    dx_ref = np.interp(tw.s, twmad.s, twmad.dx * beta0)
    betx_ref = np.interp(tw.s, twmad.s, twmad.betx)
    bety_ref = np.interp(tw.s, twmad.s, twmad.bety)

    xo.assert_allclose(tw.dx, dx_ref, rtol=0, atol=1e-3)
    xo.assert_allclose(tw.betx, betx_ref, rtol=1e-5, atol=0)
    xo.assert_allclose(tw.bety, bety_ref, rtol=1e-5, atol=0)

    xo.assert_allclose(tw.momentum_compaction_factor, twmad.summary.alfa,
                      atol=0, rtol=5e-3)

    xo.assert_allclose(tw.dqx, twmad.summary.dq1 * beta0, rtol=0, atol=1e-3)
    xo.assert_allclose(tw.dqy, twmad.summary.dq2 * beta0, rtol=0, atol=1e-3)

