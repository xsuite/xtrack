import xtrack as xt
import numpy as np
from xobjects.test_helpers import for_all_test_contexts
import pytest

assert_allclose = np.testing.assert_allclose

def _make_line_no_expressions():
    bend = xt.Bend(
        k0=0.4, angle=0.3, length=1,
        shift_x=1e-3, shift_y=2e-3, shift_s=2e-3, rot_s_rad=0.2,
        k1=0.1,
        knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4])
    quad = xt.Quadrupole(k1=0.1, k1s=0.2,
        length=0.5, shift_x=2e-3, shift_y=1e-3, shift_s=-1e-3, rot_s_rad=0.1)
    sext = xt.Sextupole(k2=0.1, k2s=0.2,
        length=0.3, shift_x=3e-3, shift_y=4e-3, shift_s=-2e-3, rot_s_rad=0.3)
    octu = xt.Octupole(k3=0.1, k3s=0.2,
        length=0.4, shift_x=5e-3, shift_y=6e-3, shift_s=-1e-3, rot_s_rad=0.4)
    mult = xt.Multipole(knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4],
        length=0.4, shift_x=5e-3, shift_y=6e-3, shift_s=-1e-3, rot_s_rad=0.7,
        hxl=0.1)
    drift = xt.Drift(length=5.0)

    line = xt.Line(elements=[bend, quad, sext, octu, mult, drift, xt.Replica(parent_name='e0')])

    return line

def _make_line_with_expressions():
    bend = xt.Bend(length=999.)
    quad = xt.Quadrupole(length=999.)
    sext = xt.Sextupole(length=999.)
    octu = xt.Octupole(length=999.)
    mult = xt.Multipole(length=999, knl=[999.]*4, ksl=[999.]*4)
    drift = xt.Drift()

    line = xt.Line(elements=[bend, quad, sext, octu, mult, drift, xt.Replica(parent_name='e0')])

    line.vars['k0_bend'] = 999.
    line.vars['angle_bend'] = 999.
    line.vars['length_bend'] = 999.
    line.vars['shift_x_bend'] = 999.
    line.vars['shift_y_bend'] = 999.
    line.vars['shift_s_bend'] = 999.
    line.vars['rot_s_rad_bend'] = 999.
    line.vars['k1_bend'] = 999.
    line.vars['knl_bend_0'] = 999.
    line.vars['knl_bend_1'] = 999.
    line.vars['knl_bend_2'] = 999.
    line.vars['knl_bend_3'] = 999.
    line.vars['ksl_bend_0'] = 999.
    line.vars['ksl_bend_1'] = 999.
    line.vars['ksl_bend_2'] = 999.
    line.vars['ksl_bend_3'] = 999.

    line.vars['k1_quad'] = 999.
    line.vars['k1s_quad'] = 999.
    line.vars['length_quad'] = 999.
    line.vars['shift_x_quad'] = 999.
    line.vars['shift_y_quad'] = 999.
    line.vars['shift_s_quad'] = 999.
    line.vars['rot_s_rad_quad'] = 999.

    line.vars['k2_sext'] = 999.
    line.vars['k2s_sext'] = 999.
    line.vars['length_sext'] = 999.
    line.vars['shift_x_sext'] = 999.
    line.vars['shift_y_sext'] = 999.
    line.vars['shift_s_sext'] = 999.
    line.vars['rot_s_rad_sext'] = 999.

    line.vars['k3_octu'] = 999.
    line.vars['k3s_octu'] = 999.
    line.vars['length_octu'] = 999.
    line.vars['shift_x_octu'] = 999.
    line.vars['shift_y_octu'] = 999.
    line.vars['shift_s_octu'] = 999.
    line.vars['rot_s_rad_octu'] = 999.

    line.vars['knl_mult_0'] = 999.
    line.vars['knl_mult_1'] = 999.
    line.vars['knl_mult_2'] = 999.
    line.vars['knl_mult_3'] = 999.
    line.vars['ksl_mult_0'] = 999.
    line.vars['ksl_mult_1'] = 999.
    line.vars['ksl_mult_2'] = 999.
    line.vars['ksl_mult_3'] = 999.
    line.vars['length_mult'] = 999.
    line.vars['shift_x_mult'] = 999.
    line.vars['shift_y_mult'] = 999.
    line.vars['shift_s_mult'] = 999.
    line.vars['rot_s_rad_mult'] = 999.
    line.vars['hxl_mult'] = 999.

    line.vars['length_drift'] = 999.

    line.element_refs['e0'].k0 = line.vars['k0_bend']
    line.element_refs['e0'].angle = line.vars['angle_bend']
    line.element_refs['e0'].length = line.vars['length_bend']
    line.element_refs['e0'].shift_x = line.vars['shift_x_bend']
    line.element_refs['e0'].shift_y = line.vars['shift_y_bend']
    line.element_refs['e0'].shift_s = line.vars['shift_s_bend']
    line.element_refs['e0'].rot_s_rad = line.vars['rot_s_rad_bend']
    line.element_refs['e0'].k1 = line.vars['k1_bend']
    line.element_refs['e0'].knl[0] = line.vars['knl_bend_0']
    line.element_refs['e0'].knl[1] = line.vars['knl_bend_1']
    line.element_refs['e0'].knl[2] = line.vars['knl_bend_2']
    line.element_refs['e0'].knl[3] = line.vars['knl_bend_3']
    line.element_refs['e0'].ksl[0] = line.vars['ksl_bend_0']
    line.element_refs['e0'].ksl[1] = line.vars['ksl_bend_1']
    line.element_refs['e0'].ksl[2] = line.vars['ksl_bend_2']
    line.element_refs['e0'].ksl[3] = line.vars['ksl_bend_3']

    line.element_refs['e1'].k1 = line.vars['k1_quad']
    line.element_refs['e1'].k1s = line.vars['k1s_quad']
    line.element_refs['e1'].length = line.vars['length_quad']
    line.element_refs['e1'].shift_x = line.vars['shift_x_quad']
    line.element_refs['e1'].shift_y = line.vars['shift_y_quad']
    line.element_refs['e1'].shift_s = line.vars['shift_s_quad']
    line.element_refs['e1'].rot_s_rad = line.vars['rot_s_rad_quad']

    line.element_refs['e2'].k2 = line.vars['k2_sext']
    line.element_refs['e2'].k2s = line.vars['k2s_sext']
    line.element_refs['e2'].length = line.vars['length_sext']
    line.element_refs['e2'].shift_x = line.vars['shift_x_sext']
    line.element_refs['e2'].shift_y = line.vars['shift_y_sext']
    line.element_refs['e2'].shift_s = line.vars['shift_s_sext']
    line.element_refs['e2'].rot_s_rad = line.vars['rot_s_rad_sext']

    line.element_refs['e3'].k3 = line.vars['k3_octu']
    line.element_refs['e3'].k3s = line.vars['k3s_octu']
    line.element_refs['e3'].length = line.vars['length_octu']
    line.element_refs['e3'].shift_x = line.vars['shift_x_octu']
    line.element_refs['e3'].shift_y = line.vars['shift_y_octu']
    line.element_refs['e3'].shift_s = line.vars['shift_s_octu']
    line.element_refs['e3'].rot_s_rad = line.vars['rot_s_rad_octu']

    line.element_refs['e4'].knl[0] = line.vars['knl_mult_0']
    line.element_refs['e4'].knl[1] = line.vars['knl_mult_1']
    line.element_refs['e4'].knl[2] = line.vars['knl_mult_2']
    line.element_refs['e4'].knl[3] = line.vars['knl_mult_3']
    line.element_refs['e4'].ksl[0] = line.vars['ksl_mult_0']
    line.element_refs['e4'].ksl[1] = line.vars['ksl_mult_1']
    line.element_refs['e4'].ksl[2] = line.vars['ksl_mult_2']
    line.element_refs['e4'].ksl[3] = line.vars['ksl_mult_3']
    line.element_refs['e4'].length = line.vars['length_mult']
    line.element_refs['e4'].shift_x = line.vars['shift_x_mult']
    line.element_refs['e4'].shift_y = line.vars['shift_y_mult']
    line.element_refs['e4'].shift_s = line.vars['shift_s_mult']
    line.element_refs['e4'].rot_s_rad = line.vars['rot_s_rad_mult']
    line.element_refs['e4'].hxl = line.vars['hxl_mult']

    line.element_refs['e5'].length = line.vars['length_drift']

    return line

def _set_vars(line):
    line.vars['k0_bend'] = 0.4
    line.vars['angle_bend'] = 0.3
    line.vars['length_bend'] = 1
    line.vars['shift_x_bend'] = 1e-3
    line.vars['shift_y_bend'] = 2e-3
    line.vars['shift_s_bend'] = 2e-3
    line.vars['rot_s_rad_bend'] = 0.2
    line.vars['k1_bend'] = 0.1
    line.vars['knl_bend_0'] = 0.7
    line.vars['knl_bend_1'] = 0.8
    line.vars['knl_bend_2'] = 0.9
    line.vars['knl_bend_3'] = 1.0
    line.vars['ksl_bend_0'] = 0.1
    line.vars['ksl_bend_1'] = 0.2
    line.vars['ksl_bend_2'] = 0.3
    line.vars['ksl_bend_3'] = 0.4

    line.vars['k1_quad'] = 0.1
    line.vars['k1s_quad'] = 0.2
    line.vars['length_quad'] = 0.5
    line.vars['shift_x_quad'] = 2e-3
    line.vars['shift_y_quad'] = 1e-3
    line.vars['shift_s_quad'] = -1e-3
    line.vars['rot_s_rad_quad'] = 0.1

    line.vars['k2_sext'] = 0.1
    line.vars['k2s_sext'] = 0.2
    line.vars['length_sext'] = 0.3
    line.vars['shift_x_sext'] = 3e-3
    line.vars['shift_y_sext'] = 4e-3
    line.vars['shift_s_sext'] = -2e-3
    line.vars['rot_s_rad_sext'] = 0.3

    line.vars['k3_octu'] = 0.1
    line.vars['k3s_octu'] = 0.2
    line.vars['length_octu'] = 0.4
    line.vars['shift_x_octu'] = 5e-3
    line.vars['shift_y_octu'] = 6e-3
    line.vars['shift_s_octu'] = -1e-3
    line.vars['rot_s_rad_octu'] = 0.4

    line.vars['knl_mult_0'] = 0.7
    line.vars['knl_mult_1'] = 0.8
    line.vars['knl_mult_2'] = 0.9
    line.vars['knl_mult_3'] = 1.0
    line.vars['ksl_mult_0'] = 0.1
    line.vars['ksl_mult_1'] = 0.2
    line.vars['ksl_mult_2'] = 0.3
    line.vars['ksl_mult_3'] = 0.4
    line.vars['length_mult'] = 0.4
    line.vars['shift_x_mult'] = 5e-3
    line.vars['shift_y_mult'] = 6e-3
    line.vars['shift_s_mult'] = -1e-3
    line.vars['rot_s_rad_mult'] = 0.7
    line.vars['hxl_mult'] = 0.1

    line.vars['length_drift'] = 5.0


@for_all_test_contexts
@pytest.mark.parametrize(
    "check_expr",
    [False, True],
    ids=['no_expr', 'with_expr'],
)
@pytest.mark.parametrize(
    "use_copy",
    [False, True],
    ids=['no_copy', 'with_copy'],
)
def test_attr_replicas(test_context, check_expr, use_copy):

    if check_expr:
        line = _make_line_with_expressions()
    else:
        line = _make_line_no_expressions()

    line.build_tracker(_context=test_context)

    if use_copy:
        line = line.copy()
        line.build_tracker(_context=test_context)

    if check_expr:
        _set_vars(line)

    tt = line.get_table(attr=True)

    assert_allclose(tt['s', -1], 8.2, rtol=0, atol=1e-14)

    assert tt['element_type','e0'] == 'Bend'
    assert tt['isreplica', 'e0'] == False
    assert tt['parent_name', 'e0'] is None
    assert tt['isthick', 'e0'] == True
    assert tt['iscollective', 'e0'] == False
    assert_allclose(tt['s', 'e0'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0'], 1, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0'], 0.4 * 1 + 0.7, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0'], 0.1 * 1 + 0.8, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0'], 0.9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0'], 1.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0'], 0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e1'] == 'Quadrupole'
    assert tt['isreplica', 'e1'] == False
    assert tt['parent_name', 'e1'] is None
    assert tt['isthick', 'e1'] == True
    assert tt['iscollective', 'e1'] == False
    assert_allclose(tt['s', 'e1'], 1., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e1'], 0.5, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e1'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e1'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e1'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e1'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e1'], 0.1 * 0.5, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e1'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e1'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e1'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e1'], 0.2 * 0.5, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e1'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e1'], 0.0, rtol=0, atol=1e-14)

    assert tt['element_type','e2'] == 'Sextupole'
    assert tt['isreplica', 'e2'] == False
    assert tt['parent_name', 'e2'] is None
    assert tt['isthick', 'e2'] == True
    assert tt['iscollective', 'e2'] == False
    assert_allclose(tt['s', 'e2'], 1.5, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e2'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e2'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e2'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e2'], 3e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e2'], 4e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e2'], -2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e2'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e2'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e2'], 0.1 * 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e2'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e2'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e2'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e2'], 0.2 * 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e2'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e3'] == 'Octupole'
    assert tt['isreplica', 'e3'] == False
    assert tt['parent_name', 'e3'] is None
    assert tt['isthick', 'e3'] == True
    assert tt['iscollective', 'e3'] == False
    assert_allclose(tt['s', 'e3'], 1.8, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e3'], 0.4, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e3'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e3'], 0.4, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e3'], 5e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e3'], 6e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e3'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e3'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e3'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e3'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e3'], 0.1 * 0.4, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e3'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e3'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e3'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e3'], 0.2 * 0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e4'] == 'Multipole'
    assert tt['isreplica', 'e4'] == False
    assert tt['parent_name', 'e4'] is None
    assert tt['isthick', 'e4'] == False
    assert tt['iscollective', 'e4'] == False
    assert_allclose(tt['s', 'e4'], 2.2, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e4'], 0.4, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e4'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e4'], 0.7, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e4'], 5e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e4'], 6e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e4'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e4'], 0.7, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e4'], 0.8, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e4'], 0.9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e4'], 1.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e4'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e4'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e4'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e4'], 0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e5'] == 'Drift'
    assert tt['isreplica', 'e5'] == False
    assert tt['parent_name', 'e5'] is None
    assert tt['isthick', 'e5'] == True
    assert tt['iscollective', 'e5'] == False
    assert_allclose(tt['s', 'e5'], 2.2, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e5'], 5.0, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e5'], 0.0, rtol=0, atol=1e-14)

    assert tt['element_type','e6'] == 'Bend'
    assert tt['isreplica', 'e6'] == True
    assert tt['parent_name', 'e6'] == 'e0'
    assert tt['isthick', 'e6'] == True
    assert tt['iscollective', 'e6'] == False
    assert_allclose(tt['length', 'e6'], 1, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6'], 0.4 * 1 + 0.7, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6'], 0.1 * 1 + 0.8, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6'], 0.9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6'], 1.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6'], 0.4, rtol=0, atol=1e-14)


@for_all_test_contexts
@pytest.mark.parametrize(
    "check_expr",
    [False, True],
    ids=['no_expr', 'with_expr'],
)
@pytest.mark.parametrize(
    "use_copy",
    [False, True],
    ids=['no_copy', 'with_copy'],
)
def test_attr_thin_slicing(test_context, check_expr, use_copy):

    if check_expr:
        line = _make_line_with_expressions()
    else:
        line = _make_line_no_expressions()

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(2, mode='thin'))])

    line.build_tracker(_context=test_context)

    if use_copy:
        line = line.copy()
        line.build_tracker(_context=test_context)

    if check_expr:
        _set_vars(line)

    tt = line.get_table(attr=True)

    assert_allclose(tt['s', -1], 8.2, rtol=0, atol=1e-14)

    assert tt['element_type','e0..1'] == 'ThinSliceBend'
    assert tt['isreplica', 'e0..1'] == False
    assert tt['parent_name', 'e0..1'] == 'e0'
    assert tt['isthick', 'e0..1'] == False
    assert tt['iscollective', 'e0..1'] == False
    assert_allclose(tt['s', 'e0..1'], 2./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0..1'], 1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0..1'], 0.3/2, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0..1'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0..1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0..1'], 0.5 * (0.4 * 1 + 0.7), rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0..1'], 0.5 * (0.1 * 1 + 0.8), rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0..1'], 0.5 *  .9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0..1'], 0.5 * 1., rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0..1'], 0.5 * 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0..1'], 0.5 * 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0..1'], 0.5 *  0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0..1'], 0.5 *  0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e0..exit_map'] == 'ThinSliceBendExit'
    assert tt['isreplica', 'e0..exit_map'] == False
    assert tt['parent_name', 'e0..exit_map'] == 'e0'
    assert tt['isthick', 'e0..exit_map'] == False
    assert tt['iscollective', 'e0..exit_map'] == False
    assert_allclose(tt['s', 'e0..exit_map'], 1., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0..exit_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0..exit_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e0..entry_map'] == 'ThinSliceBendEntry'
    assert tt['isreplica', 'e0..entry_map'] == False
    assert tt['parent_name', 'e0..entry_map'] == 'e0'
    assert tt['isthick', 'e0..entry_map'] == False
    assert tt['iscollective', 'e0..entry_map'] == False
    assert_allclose(tt['s', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0..entry_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0..entry_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','drift_e0..1'] == 'DriftSliceBend'
    assert tt['isreplica', 'drift_e0..1'] == False
    assert tt['parent_name', 'drift_e0..1'] == 'e0'
    assert tt['isthick', 'drift_e0..1'] == True
    assert tt['iscollective', 'drift_e0..1'] == False
    assert_allclose(tt['s', 'drift_e0..1'], 1./3, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'drift_e0..1'], 1/3, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'drift_e0..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'drift_e0..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'drift_e0..1'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e1..1'] == 'ThinSliceQuadrupole'
    assert tt['isreplica', 'e1..1'] == False
    assert tt['parent_name', 'e1..1'] == 'e1'
    assert tt['isthick', 'e1..1'] == False
    assert tt['iscollective', 'e1..1'] == False
    assert_allclose(tt['s', 'e1..1'], 1 + .5*2./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e1..1'], 0.5*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e1..1'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e1..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e1..1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e1..1'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e1..1'], 0.1 * 0.5 / 2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e1..1'], 0.5 * 0.2 / 2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e1..1'], 0., rtol=0, atol=1e-14)

    assert tt['element_type','drift_e1..1'] == 'DriftSliceQuadrupole'
    assert tt['isreplica', 'drift_e1..1'] == False
    assert tt['parent_name', 'drift_e1..1'] == 'e1'
    assert tt['isthick', 'drift_e1..1'] == True
    assert tt['iscollective', 'drift_e1..1'] == False
    assert_allclose(tt['s', 'drift_e1..1'], 1 + 0.5 * 1./3, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'drift_e1..1'], 0.5 * 1/3, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'drift_e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'drift_e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'drift_e1..1'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e1_entry'] == 'Marker'
    assert tt['isreplica', 'e1_entry'] == False
    assert tt['parent_name', 'e1_entry'] is None
    assert tt['isthick', 'e1_entry'] == False
    assert tt['iscollective', 'e1_entry'] == False
    assert_allclose(tt['s', 'e1_entry'], 1, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e1_entry'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e1_entry'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e1_entry'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e2..1'] == 'ThinSliceSextupole'
    assert tt['isreplica', 'e2..1'] == False
    assert tt['parent_name', 'e2..1'] == 'e2'
    assert tt['isthick', 'e2..1'] == False
    assert tt['iscollective', 'e2..1'] == False
    assert_allclose(tt['s', 'e2..1'], 1.5 + .3*2./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e2..1'], 0.3*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e2..1'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e2..1'], 3e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e2..1'], 4e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e2..1'], -2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e2..1'], 0.1 * 0.3*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e2..1'], 0.2 * 0.3*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e2..1'], 0., rtol=0, atol=1e-14)

    assert tt['element_type','drift_e2..1'] == 'DriftSliceSextupole'
    assert tt['isreplica', 'drift_e2..1'] == False
    assert tt['parent_name', 'drift_e2..1'] == 'e2'
    assert tt['isthick', 'drift_e2..1'] == True
    assert tt['iscollective', 'drift_e2..1'] == False
    assert_allclose(tt['s', 'drift_e2..1'], 1.5 + .3*1./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'drift_e2..1'], 0.3*1/3, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'drift_e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'drift_e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'drift_e2..1'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e3..1'] == 'ThinSliceOctupole'
    assert tt['isreplica', 'e3..1'] == False
    assert tt['parent_name', 'e3..1'] == 'e3'
    assert tt['isthick', 'e3..1'] == False
    assert tt['iscollective', 'e3..1'] == False
    assert_allclose(tt['s', 'e3..1'], 1.8 + .4*2./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e3..1'], 0.4*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e3..1'], 0.4, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e3..1'], 5e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e3..1'], 6e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e3..1'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e3..1'], 0.1 * 0.4*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e3..1'], 0.2 * 0.4*1/2, rtol=0, atol=1e-14)

    assert tt['element_type','drift_e3..1'] == 'DriftSliceOctupole'
    assert tt['isreplica', 'drift_e3..1'] == False
    assert tt['parent_name', 'drift_e3..1'] == 'e3'
    assert tt['isthick', 'drift_e3..1'] == True
    assert tt['iscollective', 'drift_e3..1'] == False
    assert_allclose(tt['s', 'drift_e3..1'], 1.8 + .4*1./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'drift_e3..1'], 0.4*1/3, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'drift_e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'drift_e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'drift_e3..1'], 0, rtol=0, atol=1e-14)

    # Check e5 untouched
    assert tt['element_type','e5'] == 'Drift'
    assert tt['isreplica', 'e5'] == False
    assert tt['parent_name', 'e5'] is None
    assert tt['isthick', 'e5'] == True
    assert tt['iscollective', 'e5'] == False
    assert_allclose(tt['s', 'e5'], 2.2, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e5'], 5.0, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e5'], 0.0, rtol=0, atol=1e-14)

    assert tt['element_type','e6..1'] == 'ThinSliceBend'
    assert tt['isreplica', 'e6..1'] == False
    assert tt['parent_name', 'e6..1'] == 'e0'
    assert tt['isthick', 'e6..1'] == False
    assert tt['iscollective', 'e6..1'] == False
    assert_allclose(tt['s', 'e6..1'], 7.2 + 2./3., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e6..1'], 1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6..1'], 0.3/2, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6..1'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6..1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6..1'], 0.5 * (0.4 * 1 + 0.7), rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6..1'], 0.5 * (0.1 * 1 + 0.8), rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6..1'], 0.5 *  .9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6..1'], 0.5 * 1., rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6..1'], 0.5 * 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6..1'], 0.5 * 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6..1'], 0.5 *  0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6..1'], 0.5 *  0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e6..exit_map'] == 'ThinSliceBendExit'
    assert tt['isreplica', 'e6..exit_map'] == False
    assert tt['parent_name', 'e6..exit_map'] == 'e0'
    assert tt['isthick', 'e6..exit_map'] == False
    assert tt['iscollective', 'e6..exit_map'] == False
    assert_allclose(tt['s', 'e6..exit_map'], 7.2 + 1., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e6..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6..exit_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6..exit_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e6..entry_map'] == 'ThinSliceBendEntry'
    assert tt['isreplica', 'e6..entry_map'] == False
    assert tt['parent_name', 'e6..entry_map'] == 'e0'
    assert tt['isthick', 'e6..entry_map'] == False
    assert tt['iscollective', 'e6..entry_map'] == False
    assert_allclose(tt['s', 'e6..entry_map'], 7.2, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e6..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6..entry_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6..entry_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','drift_e6..1'] == 'DriftSliceBend'
    assert tt['isreplica', 'drift_e6..1'] == False
    assert tt['parent_name', 'drift_e6..1'] == 'e0'
    assert tt['isthick', 'drift_e6..1'] == True
    assert tt['iscollective', 'drift_e6..1'] == False
    assert_allclose(tt['s', 'drift_e6..1'], 7.2 + 1./3, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'drift_e6..1'], 1/3, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'drift_e6..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'drift_e6..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'drift_e6..1'], 0, rtol=0, atol=1e-14)


@for_all_test_contexts
@pytest.mark.parametrize(
    "check_expr",
    [False, True],
    ids=["no_expr", "with_expr"],
)
@pytest.mark.parametrize(
    "use_copy",
    [False, True],
    ids=["no_copy", "with_copy"],
)
def test_attr_thick_slicing(test_context, check_expr, use_copy):

    if check_expr:
        line = _make_line_with_expressions()
    else:
        line = _make_line_no_expressions()

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(2, mode='thick'))])

    line.build_tracker(_context=test_context)

    if use_copy:
        line = line.copy()
        line.build_tracker(_context=test_context)

    if check_expr:
        _set_vars(line)

    tt = line.get_table(attr=True)

    assert_allclose(tt['s', -1], 8.2, rtol=0, atol=1e-14)

    assert tt['element_type','e0..1'] == 'ThickSliceBend'
    assert tt['isreplica', 'e0..1'] == False
    assert tt['parent_name', 'e0..1'] == 'e0'
    assert tt['isthick', 'e0..1'] == True
    assert tt['iscollective', 'e0..1'] == False
    assert_allclose(tt['s', 'e0..1'], 1./2., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0..1'], 1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0..1'], 0.3/2, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0..1'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0..1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0..1'], 0.5 * (0.4 * 1 + 0.7), rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0..1'], 0.5 * (0.1 * 1 + 0.8), rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0..1'], 0.5 *  .9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0..1'], 0.5 * 1., rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0..1'], 0.5 * 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0..1'], 0.5 * 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0..1'], 0.5 *  0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0..1'], 0.5 *  0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e0..exit_map'] == 'ThinSliceBendExit'
    assert tt['isreplica', 'e0..exit_map'] == False
    assert tt['parent_name', 'e0..exit_map'] == 'e0'
    assert tt['isthick', 'e0..exit_map'] == False
    assert tt['iscollective', 'e0..exit_map'] == False
    assert_allclose(tt['s', 'e0..exit_map'], 1., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0..exit_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0..exit_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0..exit_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e0..entry_map'] == 'ThinSliceBendEntry'
    assert tt['isreplica', 'e0..entry_map'] == False
    assert tt['parent_name', 'e0..entry_map'] == 'e0'
    assert tt['isthick', 'e0..entry_map'] == False
    assert tt['iscollective', 'e0..entry_map'] == False
    assert_allclose(tt['s', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e0..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e0..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e0..entry_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e0..entry_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e0..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e0..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e0..entry_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e1..1'] == 'ThickSliceQuadrupole'
    assert tt['isreplica', 'e1..1'] == False
    assert tt['parent_name', 'e1..1'] == 'e1'
    assert tt['isthick', 'e1..1'] == True
    assert tt['iscollective', 'e1..1'] == False
    assert_allclose(tt['s', 'e1..1'], 1 + .5 / 2, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e1..1'], 0.5*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e1..1'], 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e1..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e1..1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e1..1'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e1..1'], 0.1 * 0.5 / 2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e1..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e1..1'], 0.5 * 0.2 / 2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e1..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e1..1'], 0., rtol=0, atol=1e-14)

    assert tt['element_type','e1_entry'] == 'Marker'
    assert tt['isreplica', 'e1_entry'] == False
    assert tt['parent_name', 'e1_entry'] is None
    assert tt['isthick', 'e1_entry'] == False
    assert tt['iscollective', 'e1_entry'] == False
    assert_allclose(tt['s', 'e1_entry'], 1, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e1_entry'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e1_entry'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e1_entry'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e1_entry'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e2..1'] == 'ThickSliceSextupole'
    assert tt['isreplica', 'e2..1'] == False
    assert tt['parent_name', 'e2..1'] == 'e2'
    assert tt['isthick', 'e2..1'] == True
    assert tt['iscollective', 'e2..1'] == False
    assert_allclose(tt['s', 'e2..1'], 1.5 + .3 / 2., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e2..1'], 0.3*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e2..1'], 0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e2..1'], 3e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e2..1'], 4e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e2..1'], -2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e2..1'], 0.1 * 0.3*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e2..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e2..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e2..1'], 0.2 * 0.3*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e2..1'], 0., rtol=0, atol=1e-14)

    assert tt['element_type','e3..1'] == 'ThickSliceOctupole'
    assert tt['isreplica', 'e3..1'] == False
    assert tt['parent_name', 'e3..1'] == 'e3'
    assert tt['isthick', 'e3..1'] == True
    assert tt['iscollective', 'e3..1'] == False
    assert_allclose(tt['s', 'e3..1'], 1.8 + .4 /2., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e3..1'], 0.4*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e3..1'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e3..1'], 0.4, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e3..1'], 5e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e3..1'], 6e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e3..1'], -1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e3..1'], 0.1 * 0.4*1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e3..1'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e3..1'], 0.2 * 0.4*1/2, rtol=0, atol=1e-14)

    # Check e5 untouched
    assert tt['element_type','e5'] == 'Drift'
    assert tt['isreplica', 'e5'] == False
    assert tt['parent_name', 'e5'] is None
    assert tt['isthick', 'e5'] == True
    assert tt['iscollective', 'e5'] == False
    assert_allclose(tt['s', 'e5'], 2.2, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e5'], 5.0, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e5'], 0.0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e5'], 0.0, rtol=0, atol=1e-14)

    assert tt['element_type','e6..1'] == 'ThickSliceBend'
    assert tt['isreplica', 'e6..1'] == False
    assert tt['parent_name', 'e6..1'] == 'e0'
    assert tt['isthick', 'e6..1'] == True
    assert tt['iscollective', 'e6..1'] == False
    assert_allclose(tt['s', 'e6..1'], 7.2 + 1./2., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e6..1'], 1/2, rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6..1'], 0.3/2, rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6..1'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6..1'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6..1'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6..1'], 0.5 * (0.4 * 1 + 0.7), rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6..1'], 0.5 * (0.1 * 1 + 0.8), rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6..1'], 0.5 *  .9, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6..1'], 0.5 * 1., rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6..1'], 0.5 * 0.1, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6..1'], 0.5 * 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6..1'], 0.5 *  0.3, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6..1'], 0.5 *  0.4, rtol=0, atol=1e-14)

    assert tt['element_type','e6..exit_map'] == 'ThinSliceBendExit'
    assert tt['isreplica', 'e6..exit_map'] == False
    assert tt['parent_name', 'e6..exit_map'] == 'e0'
    assert tt['isthick', 'e6..exit_map'] == False
    assert tt['iscollective', 'e6..exit_map'] == False
    assert_allclose(tt['s', 'e6..exit_map'], 7.2 + 1., rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e6..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6..exit_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6..exit_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6..exit_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6..exit_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6..exit_map'], 0, rtol=0, atol=1e-14)

    assert tt['element_type','e6..entry_map'] == 'ThinSliceBendEntry'
    assert tt['isreplica', 'e6..entry_map'] == False
    assert tt['parent_name', 'e6..entry_map'] == 'e0'
    assert tt['isthick', 'e6..entry_map'] == False
    assert tt['iscollective', 'e6..entry_map'] == False
    assert_allclose(tt['s', 'e6..entry_map'], 7.2 + 0, rtol=0, atol=1e-14)
    assert_allclose(tt['length', 'e6..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['angle_rad', 'e6..entry_map'], 0., rtol=0, atol=1e-14)
    assert_allclose(tt['rot_s_rad', 'e6..entry_map'], 0.2, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_x', 'e6..entry_map'], 1e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_y', 'e6..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['shift_s', 'e6..entry_map'], 2e-3, rtol=0, atol=1e-14)
    assert_allclose(tt['k0l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3l', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k0sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k1sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k2sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
    assert_allclose(tt['k3sl', 'e6..entry_map'], 0, rtol=0, atol=1e-14)
