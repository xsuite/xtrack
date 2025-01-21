import json
import pathlib

from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()

correction_config = {
    'IR1 left': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.r8.b1',
        end='e.ds.l1.b1',
        vary=(
            'corr_co_acbh14.l1b1',
            'corr_co_acbh12.l1b1',
            'corr_co_acbv15.l1b1',
            'corr_co_acbv13.l1b1',
            ),
        targets=('e.ds.l1.b1',),
    ),
    'IR1 right': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r1.b1',
        end='s.ds.l2.b1',
        vary=(
            'corr_co_acbh13.r1b1',
            'corr_co_acbh15.r1b1',
            'corr_co_acbv12.r1b1',
            'corr_co_acbv14.r1b1',
            ),
        targets=('s.ds.l2.b1',),
    ),
    'IR5 left': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.r4.b1',
        end='e.ds.l5.b1',
        vary=(
            'corr_co_acbh14.l5b1',
            'corr_co_acbh12.l5b1',
            'corr_co_acbv15.l5b1',
            'corr_co_acbv13.l5b1',
            ),
        targets=('e.ds.l5.b1',),
    ),
    'IR5 right': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r5.b1',
        end='s.ds.l6.b1',
        vary=(
            'corr_co_acbh13.r5b1',
            'corr_co_acbh15.r5b1',
            'corr_co_acbv12.r5b1',
            'corr_co_acbv14.r5b1',
            ),
        targets=('s.ds.l6.b1',),
    ),
    'IP1': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l1.b1',
        end='s.ds.r1.b1',
        vary=(
            'corr_co_acbch6.l1b1',
            'corr_co_acbcv5.l1b1',
            'corr_co_acbch5.r1b1',
            'corr_co_acbcv6.r1b1',
            'corr_co_acbyhs4.l1b1',
            'corr_co_acbyhs4.r1b1',
            'corr_co_acbyvs4.l1b1',
            'corr_co_acbyvs4.r1b1',
        ),
        targets=('ip1', 's.ds.r1.b1'),
    ),
    'IP2': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l2.b1',
        end='s.ds.r2.b1',
        vary=(
            'corr_co_acbyhs5.l2b1',
            'corr_co_acbchs5.r2b1',
            'corr_co_acbyvs5.l2b1',
            'corr_co_acbcvs5.r2b1',
            'corr_co_acbyhs4.l2b1',
            'corr_co_acbyhs4.r2b1',
            'corr_co_acbyvs4.l2b1',
            'corr_co_acbyvs4.r2b1',
        ),
        targets=('ip2', 's.ds.r2.b1'),
    ),
    'IP5': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l5.b1',
        end='s.ds.r5.b1',
        vary=(
            'corr_co_acbch6.l5b1',
            'corr_co_acbcv5.l5b1',
            'corr_co_acbch5.r5b1',
            'corr_co_acbcv6.r5b1',
            'corr_co_acbyhs4.l5b1',
            'corr_co_acbyhs4.r5b1',
            'corr_co_acbyvs4.l5b1',
            'corr_co_acbyvs4.r5b1',
        ),
        targets=('ip5', 's.ds.r5.b1'),
    ),
    'IP8': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l8.b1',
        end='s.ds.r8.b1',
        vary=(
            'corr_co_acbch5.l8b1',
            'corr_co_acbyhs4.l8b1',
            'corr_co_acbyhs4.r8b1',
            'corr_co_acbyhs5.r8b1',
            'corr_co_acbcvs5.l8b1',
            'corr_co_acbyvs4.l8b1',
            'corr_co_acbyvs4.r8b1',
            'corr_co_acbyvs5.r8b1',
        ),
        targets=('ip8', 's.ds.r8.b1'),
    ),
}


@for_all_test_contexts
def test_orbit_correction(test_context):
    with open(test_data_folder /
              'hllhc14_no_errors_with_coupling_knobs/line_b1.json', 'r') as fid:
        dct_b1 = json.load(fid)

    input_line = xt.Line.from_dict(dct_b1)

    # Load line with knobs on correctors only
    mad = Madx(stdout=False)
    mad.call(str(test_data_folder /
                 'hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx'))
    mad.use(sequence='lhcb1')
    input_line_co_ref = xt.Line.from_madx_sequence(
        mad.sequence.lhcb1,
        deferred_expressions=True,
        expressions_for_element_types=('kicker', 'hkicker', 'vkicker'),
    )

    collider = xt.Environment(lines={'lhcb1': input_line.copy()})
    collider.import_line(line_name='lhcb1_co_ref', line=input_line_co_ref.copy())
    collider['lhcb1_co_ref'].particle_ref = collider['lhcb1'].particle_ref.copy()
    collider.build_trackers(_context=test_context)

    # Wipe out orbit correction from pymask
    tt_corr = collider.vars.get_table().rows['corr_acb.*']
    for kk in tt_corr.name:
        collider.vars[kk] = 0

    collider.vars['on_x1'] = 0
    collider.vars['on_x2'] = 0
    collider.vars['on_x5'] = 0
    collider.vars['on_x8'] = 0
    collider.vars['on_sep1'] = 0
    collider.vars['on_sep2'] = 0
    collider.vars['on_sep5'] = 0
    collider.vars['on_sep8'] = 0
    collider.vars['on_lhcb'] = 0
    collider.vars['on_alice'] = 0

    # Check that in both machines the orbit is flat at the ips
    for nn in ['lhcb1', 'lhcb1_co_ref']:
        tw = collider[nn].twiss(method='4d', zeta0=0, delta0=0)
        for ip in ['ip1', 'ip2', 'ip5', 'ip8']:
            xo.assert_allclose(tw['x', ip], 0, 1e-10)
            xo.assert_allclose(tw['px', ip], 0, 1e-10)
            xo.assert_allclose(tw['y', ip], 0, 1e-10)
            xo.assert_allclose(tw['py', ip], 0, 1e-10)

    # Check that the tune knobs work only on line and not on line_co_ref
    tw0 = collider['lhcb1'].twiss(method='4d', zeta0=0, delta0=0)
    tw0_co_ref = collider['lhcb1_co_ref'].twiss(method='4d', zeta0=0, delta0=0)
    collider['lhcb1'].vars['kqtf.b1'] = 1e-5
    collider['lhcb1_co_ref'].vars['kqtf.b1'] = 1e-5 # This should not change anything
    tw1 = collider['lhcb1'].twiss(method='4d', zeta0=0, delta0=0)
    tw1_co_ref = collider['lhcb1_co_ref'].twiss(method='4d', zeta0=0, delta0=0)
    assert tw1.qx != tw0.qx
    assert tw1_co_ref.qx == tw0_co_ref.qx

    # Add correction term to all dipole correctors
    collider.vars['on_corr_co'] = 1
    for kk in list(collider.vars.keys()):
        if kk.startswith('acb'):
            collider.vars['corr_co_'+kk] = 0
            collider.vars[kk] += (collider.vars['corr_co_'+kk]
                                * collider.vars['on_corr_co'])

    # Set some orbit knobs in both machines and switch on experimental magnets
    collider.vars['on_x1'] = 250
    collider.vars['on_x2'] = 250
    collider.vars['on_x5'] = 250
    collider.vars['on_x8'] = 250
    collider.vars['on_disp'] = 1
    collider.vars['on_lhcb'] = 1
    collider.vars['on_alice'] = 1

    # Introduce dip kick in all triplets (only in line)
    collider['lhcb1']['mqxfb.b2l1..11'].knl[0] = 1e-6
    collider['lhcb1']['mqxfb.b2l1..11'].ksl[0] = 1.5e-6
    collider['lhcb1']['mqxfb.b2r1..11'].knl[0] = 2e-6
    collider['lhcb1']['mqxfb.b2r1..11'].ksl[0] = 1e-6
    collider['lhcb1']['mqxb.b2l2..11'].knl[0] = 1e-6
    collider['lhcb1']['mqxb.b2l2..11'].ksl[0] = 1.5e-6
    collider['lhcb1']['mqxb.b2r2..11'].knl[0] = 2e-6
    collider['lhcb1']['mqxb.b2r2..11'].ksl[0] = 1e-6
    collider['lhcb1']['mqxfb.b2l5..11'].knl[0] = 1e-6
    collider['lhcb1']['mqxfb.b2l5..11'].ksl[0] = 1.5e-6
    collider['lhcb1']['mqxfb.b2r5..11'].knl[0] = 2e-6
    collider['lhcb1']['mqxfb.b2r5..11'].ksl[0] = 1e-6
    collider['lhcb1']['mqxb.b2l8..11'].knl[0] = 1e-6
    collider['lhcb1']['mqxb.b2l8..11'].ksl[0] = 1.5e-6
    collider['lhcb1']['mqxb.b2r8..11'].knl[0] = 2e-6
    collider['lhcb1']['mqxb.b2r8..11'].ksl[0] = 1e-6

    tw_before = collider.lhcb1.twiss()

    collider.lhcb1._xmask_correct_closed_orbit(
                             reference=collider.lhcb1_co_ref,
                             correction_config=correction_config)

    tw = collider.lhcb1.twiss()

    xo.assert_allclose(tw['px', 'ip1'], 250e-6, rtol=5e-5, atol=1e-8)
    xo.assert_allclose(tw['py', 'ip1'], 0, atol=1e-8)

    xo.assert_allclose(tw['px', 'ip5'], 0, atol=1e-8)
    xo.assert_allclose(tw['py', 'ip5'], 250e-6, atol=1e-8)

    assert tw['px', 'ip2'] > 1e-7 # effect of the spectrometer tilt
    assert tw['py', 'ip2'] > 255e-6 # effect of the spectrometer
    xo.assert_allclose(tw['px', 'bpmsw.1r2.b1'], 0, atol=1e-8)  # external angle
    xo.assert_allclose(tw['py', 'bpmsw.1r2.b1'], 250e-6, atol=1e-8) # external angle

    assert tw['px', 'ip8'] > 255e-6 # effect of the spectrometer
    assert tw['py', 'ip8'] > 1e-6 # effect of the spectrometer tilt
    xo.assert_allclose(tw['px', 'bpmsw.1r8.b1'], 250e-6, atol=1e-8) # external angle
    xo.assert_allclose(tw['py', 'bpmsw.1r8.b1'], 0, atol=1e-8) # external angle


    places_to_check = [
    'e.ds.r8.b1',
    'e.ds.r4.b1',
    's.ds.l2.b1',
    's.ds.l6.b1',
    'e.ds.l1.b1',
    'e.ds.l2.b1',
    'e.ds.l5.b1',
    'e.ds.l8.b1',
    's.ds.r1.b1',
    's.ds.r2.b1',
    's.ds.r5.b1',
    's.ds.r8.b1']

    for place in places_to_check:
        xo.assert_allclose(tw['x', place], 0, atol=1e-6)
        xo.assert_allclose(tw['px', place], 0, atol=1e-8)
        xo.assert_allclose(tw['y', place], 0, atol=1e-6)
        xo.assert_allclose(tw['py', place], 0, atol=1e-8)

    with xt._temp_knobs(collider, dict(on_corr_co=0, on_disp=0)):
        tw_ref = collider.lhcb1_co_ref.twiss()
