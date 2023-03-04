import json
import pathlib

import numpy as np
import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()

with open(test_data_folder /
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json', 'r') as fid:
    dct_b1 = json.load(fid)
input_line = xt.Line.from_dict(dct_b1)

# Load line with knobs on correctors only
from cpymad.madx import Madx
mad = Madx()
mad.call(str( test_data_folder /
            'hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx'))
mad.use(sequence='lhcb1')
input_line_co_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
    deferred_expressions=True,
    expressions_for_element_types=('kicker', 'hkicker', 'vkicker'))


@for_all_test_contexts
def test_multiline_and_orbit_correction(test_context):


    collider = xt.Multiline(
        lines={'lhcb1': input_line.copy(),
               'lhcb1_co_ref': input_line_co_ref.copy()})
    collider['lhcb1_co_ref'].particle_ref = collider['lhcb1'].particle_ref.copy()

    # Profit to test the dump and load
    collider = xt.Multiline.from_dict(collider.to_dict())
    collider.build_trackers(_context=test_context)

    # Wipe out orbit correction from pymask
    for kk in collider._var_sharing.data['var_values']:
        if kk.startswith('corr_acb'):
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
            assert np.isclose(tw[ip, 'x'], 0, 1e-10)
            assert np.isclose(tw[ip, 'px'], 0, 1e-10)
            assert np.isclose(tw[ip, 'y'], 0, 1e-10)
            assert np.isclose(tw[ip, 'py'], 0, 1e-10)

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
    for kk in list(collider.vars._owner.keys()):
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

    collider.lhcb1.correct_closed_orbit(
                             verbose=True,
                             reference=collider.lhcb1_co_ref,
                             correction_config=correction_config)

    tw = collider.lhcb1.twiss()

    assert np.isclose(tw['ip1', 'px'], 250e-6, atol=1e-8)
    assert np.isclose(tw['ip1', 'py'], 0, atol=1e-8)

    assert np.isclose(tw['ip5', 'px'], 0, atol=1e-8)
    assert np.isclose(tw['ip5', 'py'], 250e-6, atol=1e-8)

    assert tw['ip2', 'px'] > 1e-7 # effect of the spectrometer tilt
    assert tw['ip2', 'py'] > 255e-6 # effect of the spectrometer
    assert np.isclose(tw['bpmsw.1r2.b1', 'px'], 0, atol=1e-8)  # external angle
    assert np.isclose(tw['bpmsw.1r2.b1', 'py'], 250e-6, atol=1e-8) # external angle

    assert tw['ip8', 'px'] > 255e-6 # effect of the spectrometer
    assert tw['ip8', 'py'] > 1e-6 # effect of the spectrometer tilt
    assert np.isclose(tw['bpmsw.1r8.b1', 'px'], 250e-6, atol=1e-8) # external angle
    assert np.isclose(tw['bpmsw.1r8.b1', 'py'], 0, atol=1e-8) # external angle


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
        assert np.isclose(tw[place, 'x'], 0, atol=1e-6)
        assert np.isclose(tw[place, 'px'], 0, atol=1e-8)
        assert np.isclose(tw[place, 'y'], 0, atol=1e-6)
        assert np.isclose(tw[place, 'py'], 0, atol=1e-8)

    with xt.tracker._temp_knobs(collider, dict(on_corr_co=0, on_disp=0)):
        tw_ref = collider.lhcb1_co_ref.twiss()


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
