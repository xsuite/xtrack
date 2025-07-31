import xtrack as xt
import numpy as np
import pytest

from xobjects.test_helpers import for_all_test_contexts
import xobjects as xo

assert_allclose = np.testing.assert_allclose

slice_mode = 'thin'
tilted = True
orientation = 'acw'
transform_to_actual_elements = True

if hasattr(np, 'trapezoid'): # numpy >= 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz

@for_all_test_contexts
@pytest.mark.parametrize(
    'slice_mode',
    [None, 'thin', 'thick'],
    ids=['no_slice', 'thin_slice', 'thick_slice'])
@pytest.mark.parametrize(
    'tilted',
    [False, True],
    ids=['not_tilted', 'tilted'])
@pytest.mark.parametrize(
    'orientation',
    ['cw', 'acw'],
    ids=['cw', 'acw'])
@pytest.mark.parametrize(
    'transform_to_actual_elements',
    [False, True],
    ids=['no_actual_elements', 'actual_elements'])
def test_survey_slicing(test_context, slice_mode, tilted, orientation,
                        transform_to_actual_elements):

    line = xt.Line(
        elements=[
            xt.Drift(),
            xt.Bend(),
            xt.Drift(),
            xt.Bend(),
            xt.Drift(),
            xt.Bend(),
            xt.Drift(),
            xt.Bend(),
        ]
    )
    line.build_tracker(_context=test_context)

    line.vars['l_drift'] = 999.
    line.vars['l_bend'] = 999.
    line.vars['h_bend'] = 999.
    line.vars['tilt_bend_deg'] = 999.


    line.element_refs['e0'].length = line.vars['l_drift']
    line.element_refs['e1'].length = line.vars['l_bend']
    line.element_refs['e2'].length = line.vars['l_drift']
    line.element_refs['e3'].length = line.vars['l_bend']
    line.element_refs['e4'].length = line.vars['l_drift']
    line.element_refs['e5'].length = line.vars['l_bend']
    line.element_refs['e6'].length = line.vars['l_drift']
    line.element_refs['e7'].length = line.vars['l_bend']

    line.element_refs['e1'].h = line.vars['h_bend']
    line.element_refs['e3'].h = line.vars['h_bend']
    line.element_refs['e5'].h = line.vars['h_bend']
    line.element_refs['e7'].h = line.vars['h_bend']

    line.element_refs['e1'].rot_s_rad = line.vars['tilt_bend_deg'] * np.pi / 180
    line.element_refs['e3'].rot_s_rad = line.vars['tilt_bend_deg'] * np.pi / 180
    line.element_refs['e5'].rot_s_rad = line.vars['tilt_bend_deg'] * np.pi / 180
    line.element_refs['e7'].rot_s_rad = line.vars['tilt_bend_deg'] * np.pi / 180

    if slice_mode is not None:
        line.slice_thick_elements(
            slicing_strategies=[xt.Strategy(xt.Teapot(20, mode=slice_mode))])
        line.build_tracker(_context=test_context)

    line.vars['l_drift'] = 1
    line.vars['l_bend'] = 1
    if orientation == 'cw':
        line.vars['h_bend'] = np.pi/2 / line.vars['l_bend']
    elif orientation == 'acw':
        line.vars['h_bend'] = -np.pi/2 / line.vars['l_bend']

    if tilted:
        line.vars['tilt_bend_deg'] = 90
    else:
        line.vars['tilt_bend_deg'] = 0

    if slice_mode == 'thin' and transform_to_actual_elements:
        line.discard_tracker()
        line._replace_with_equivalent_elements()
        line.build_tracker(_context=test_context)
        assert isinstance(line['e1..1'], xt.Multipole)

    sv = line.survey()
    assert_allclose(sv.Z[-1], 0, rtol=0, atol=1e-13)
    assert_allclose(sv.X[-1], 0, rtol=0, atol=1e-13)
    assert_allclose(sv.Y[-1], 0, rtol=0, atol=1e-13)
    assert_allclose(sv.s[-1], 8, rtol=0, atol=1e-13)


    if not tilted and orientation == 'acw':
        assert_allclose(np.abs(sv.Y), 0, rtol=0, atol=1e-14)
        assert_allclose(trapz(sv.X, sv.Z), -4.818 , rtol=0, # anti-clockwise
                        atol=(2e-3 if slice_mode is not None else 0.5))
    elif not tilted and orientation == 'cw':
        assert_allclose(np.abs(sv.Y), 0, rtol=0, atol=1e-14)
        assert_allclose(trapz(sv.X, sv.Z), 4.818 , rtol=0, # clockwise
                        atol=(2e-3 if slice_mode is not None else 0.5))
    elif tilted and orientation == 'acw':
        assert_allclose(np.abs(sv.X), 0, rtol=0, atol=1e-14)
        assert_allclose(trapz(sv.Y, sv.Z), -4.818 , rtol=0, # anti-clockwise
                        atol=(2e-3 if slice_mode is not None else 0.5))
    elif tilted and orientation == 'cw':
        assert_allclose(np.abs(sv.X), 0, rtol=0, atol=1e-14)
        assert_allclose(trapz(sv.Y, sv.Z), 4.818 , rtol=0, # clockwise
                        atol=(2e-3 if slice_mode is not None else 0.5))

def test_survey_with_ref_transformations():

    env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

    line = env.new_line(length=10, components=[
        env.new('r1', xt.YRotation, angle=30,  at=1),
        env.new('r2', xt.YRotation, angle=-30, at=2),
        env.new('r3', xt.YRotation, angle=-30, at=8),
        env.new('r4', xt.YRotation, angle=30,  at=9),

        env.new('rx1', xt.XRotation, angle=20,  at=3),
        env.new('rx2', xt.XRotation, angle=-20, at=4),
        env.new('rx3', xt.XRotation, angle=-20, at=6),
        env.new('rx4', xt.XRotation, angle=20,  at=7),

        env.new('rs1', xt.SRotation, angle=60.,  at=4.5),
        env.new('rs2', xt.SRotation, angle=-60, at=5.5),

        env.new('sxy1', xt.XYShift, dx=0.1, dy=0.2, at=4.8),
        env.new('sxy2', xt.XYShift, dx=-0.1, dy=-0.2, at=5.2),

        env.new('mid', xt.Marker, at=5.0),
        env.new('right', xt.Marker, at=9.5)

    ])

    line.config.XTRACK_GLOBAL_XY_LIMIT = None
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3, y=2e-3)

    sv_no_arg = line.survey()
    assert np.all(sv_no_arg.name == np.array([
        'drift_1', 'r1', 'drift_2', 'r2', 'drift_3', 'rx1', 'drift_4',
        'rx2', 'drift_5', 'rs1', 'drift_6', 'sxy1', 'drift_7', 'mid',
        'drift_8', 'sxy2', 'drift_9', 'rs2', 'drift_10', 'rx3', 'drift_11',
        'rx4', 'drift_12', 'r3', 'drift_13', 'r4', 'drift_14', 'right',
        'drift_15', '_end_point']))

    xo.assert_allclose(sv_no_arg.ref_shift_x, np.array([
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0.1,  0. ,  0. ,  0. , -0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]), atol=1e-14)

    xo.assert_allclose(sv_no_arg.ref_shift_y, np.array([
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0.2,  0. ,  0. ,  0. , -0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]), atol=1e-14)

    xo.assert_allclose(sv_no_arg.ref_rot_x_rad, np.array([
        0.        , -0.        ,  0.        ,  0.        ,  0.        ,
        0.34906585,  0.        , -0.34906585,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.        ,  0.        , -0.34906585,
        0.        ,  0.34906585,  0.        ,  0.        ,  0.        ,
    -0.        ,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

    xo.assert_allclose(sv_no_arg.ref_rot_y_rad, np.array([
        0.        , -0.52359878,  0.        ,  0.52359878,  0.        ,
        0.        ,  0.        , -0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.        ,  0.        , -0.        ,
        0.        ,  0.        ,  0.        ,  0.52359878,  0.        ,
    -0.52359878,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

    xo.assert_allclose(sv_no_arg.ref_rot_s_rad, np.array([
        0.        , -0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.        ,  0.        ,  1.04719755,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -1.04719755,  0.        , -0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    -0.        ,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

    xo.assert_allclose(sv_no_arg.drift_length, np.array([
        1. , 0. , 1. , 0. , 1. , 0. , 1. , 0. , 0.5, 0. , 0.3, 0. , 0.2,
        0. , 0.2, 0. , 0.3, 0. , 0.5, 0. , 1. , 0. , 1. , 0. , 1. , 0. ,
        0.5, 0. , 0.5, 0. ]), atol=1e-14)

    xo.assert_allclose(sv_no_arg.angle, np.zeros(30), atol=1e-14)
    xo.assert_allclose(sv_no_arg.rot_s_rad, np.zeros(30), atol=1e-14)

    xo.assert_allclose(
        sv_no_arg.s,
        np.array([ 0. ,  1. ,  1. ,  2. ,  2. ,  3. ,  3. ,  4. ,  4. ,  4.5,  4.5,
                4.8,  4.8,  5. ,  5. ,  5.2,  5.2,  5.5,  5.5,  6. ,  6. ,  7. ,
                7. ,  8. ,  8. ,  9. ,  9. ,  9.5,  9.5, 10. ]),
        atol=1e-14
    )

    p_no_arg = tw.x[:, None] * sv_no_arg.ex + tw.y[:, None] * sv_no_arg.ey + sv_no_arg.p0

    xo.assert_allclose(p_no_arg[:, 0], 1e-3, atol=1e-14)
    xo.assert_allclose(p_no_arg[:, 1], 2e-3, atol=1e-14)

    assert sv_no_arg.element0 == 0


    sv_mid_with_init = line.survey(element0='mid',
                            Z0=sv_no_arg['Z', 'mid'],
                            X0=sv_no_arg['X', 'mid'],
                            Y0=sv_no_arg['Y', 'mid'],
                            phi0=sv_no_arg['phi', 'mid'],
                            theta0=sv_no_arg['theta', 'mid'],
                            psi0=sv_no_arg['psi', 'mid'])

    sv_right_with_init = line.survey(element0='right',
                                Z0=sv_no_arg['Z', 'right'],
                                X0=sv_no_arg['X', 'right'],
                                Y0=sv_no_arg['Y', 'right'],
                                phi0=sv_no_arg['phi', 'right'],
                                theta0=sv_no_arg['theta', 'right'],
                                psi0=sv_no_arg['psi', 'right'])

    cols_to_check = [
        'X', 'Y', 'Z', 'theta', 'phi', 'psi', 's', 'drift_length', 'angle', 'rot_s_rad',
        'ref_shift_x', 'ref_shift_y', 'ref_rot_x_rad', 'ref_rot_y_rad', 'ref_rot_s_rad',
        'ex', 'ey', 'ez', 'p0', 'frame_matrix'
    ]

    assert sv_mid_with_init.element0 == 13
    assert sv_right_with_init.element0 == 27

    assert np.all(sv_no_arg.name == tw.name)

    for sv_test in sv_mid_with_init, sv_right_with_init:
        assert np.all(sv_test.name == sv_no_arg.name)
        for col in cols_to_check:
            xo.assert_allclose(sv_test[col], sv_no_arg[col], atol=1e-14)

    # Check with no starting from 0 in the middle
    sv_mid_no_init = line.survey(element0='mid')
    tw_init_at_mid = line.twiss4d(betx=1, bety=1, x=1e-3, y=2e-3,
                                init_at='mid')

    p_mid_no_init = tw_init_at_mid.x[:, None] * sv_mid_no_init.ex + \
                    tw_init_at_mid.y[:, None] * sv_mid_no_init.ey + sv_mid_no_init.p0

    xo.assert_allclose(p_mid_no_init[:, 0], 1e-3, atol=1e-14)
    xo.assert_allclose(p_mid_no_init[:, 1], 2e-3, atol=1e-14)

def test_survey_with_h_and_v_bends():

    env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

    line = env.new_line(length=10, components=[
        env.new('r1', xt.Bend, length=0.1, angle=np.deg2rad(30), k0_from_h=False, at=1),
        env.new('r2', xt.Bend, length=0.1, angle=-np.deg2rad(30), k0_from_h=False, at=2),
        env.new('r3', xt.Bend, length=0.1, angle=-np.deg2rad(30), k0_from_h=False, at=8),
        env.new('r4', xt.Bend, length=0.1, angle=np.deg2rad(30), k0_from_h=False, at=9),

        env.new('rx1', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=np.deg2rad(20), k0_from_h=False, at=3),
        env.new('rx2', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=-np.deg2rad(20), k0_from_h=False, at=4),
        env.new('rx3', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=-np.deg2rad(20), k0_from_h=False, at=6),
        env.new('rx4', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=np.deg2rad(20), k0_from_h=False, at=7),

        env.new('rs1', xt.SRotation, angle=60.,  at=4.5),
        env.new('rs2', xt.SRotation, angle=-60, at=5.5),

        env.new('sxy1', xt.XYShift, dx=0.1, dy=0.2, at=4.8),
        env.new('sxy2', xt.XYShift, dx=-0.1, dy=-0.2, at=5.2),

        env.new('mid', xt.Marker, at=5.0),
        env.new('right', xt.Marker, at=9.5)

    ])

    line.config.XTRACK_GLOBAL_XY_LIMIT = None
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3, y=2e-3)

    sv_no_arg = line.survey()

    assert np.all(sv_no_arg.name == np.array([
        'drift_1', 'r1', 'drift_2', 'r2', 'drift_3', 'rx1', 'drift_4',
        'rx2', 'drift_5', 'rs1', 'drift_6', 'sxy1', 'drift_7', 'mid',
        'drift_8', 'sxy2', 'drift_9', 'rs2', 'drift_10', 'rx3', 'drift_11',
        'rx4', 'drift_12', 'r3', 'drift_13', 'r4', 'drift_14', 'right',
        'drift_15', '_end_point']))

    xo.assert_allclose(sv_no_arg.ref_shift_x, np.array([
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0.1,  0. ,  0. ,  0. , -0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]), atol=1e-14)

    xo.assert_allclose(sv_no_arg.ref_shift_y, np.array([
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0.2,  0. ,  0. ,  0. , -0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]), atol=1e-14)

    xo.assert_allclose(sv_no_arg.ref_rot_x_rad, 0, atol=1e-14)
    xo.assert_allclose(sv_no_arg.ref_rot_y_rad, 0, atol=1e-14)

    xo.assert_allclose(sv_no_arg.ref_rot_s_rad, np.array([
        0.        , -0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.        ,  0.        ,  1.04719755,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -1.04719755,  0.        , -0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    -0.        ,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

    xo.assert_allclose(sv_no_arg.drift_length, np.array([
        0.95, 0.1 , 0.9 , 0.1 , 0.9 , 0.1 , 0.9 , 0.1 , 0.45, 0.  , 0.3 ,
        0.  , 0.2 , 0.  , 0.2 , 0.  , 0.3 , 0.  , 0.45, 0.1 , 0.9 , 0.1 ,
        0.9 , 0.1 , 0.9 , 0.1 , 0.45, 0.  , 0.5 , 0.   ]), atol=1e-14)

    xo.assert_allclose(sv_no_arg.angle, np.array(
        [ 0.        ,  0.52359878,  0.        , -0.52359878,  0.        ,
            0.34906585,  0.        , -0.34906585,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        , -0.34906585,
            0.        ,  0.34906585,  0.        , -0.52359878,  0.        ,
            0.52359878,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

    xo.assert_allclose(sv_no_arg.rot_s_rad, np.array([
        0.        , 0.        , 0.        , 0.        , 0.        ,
        1.57079633, 0.        , 1.57079633, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 1.57079633,
        0.        , 1.57079633, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.
    ]), atol=1e-8)

    xo.assert_allclose(
        sv_no_arg.s,
        np.array([ 0.  ,  0.95,  1.05,  1.95,  2.05,  2.95,  3.05,  3.95,  4.05,
            4.5 ,  4.5 ,  4.8 ,  4.8 ,  5.  ,  5.  ,  5.2 ,  5.2 ,  5.5 ,
            5.5 ,  5.95,  6.05,  6.95,  7.05,  7.95,  8.05,  8.95,  9.05,
            9.5 ,  9.5 , 10.   ]),   atol=1e-14
    )

    p_no_arg = tw.x[:, None] * sv_no_arg.ex + tw.y[:, None] * sv_no_arg.ey + sv_no_arg.p0

    xo.assert_allclose(p_no_arg[:, 0], 1e-3, atol=1e-14)
    xo.assert_allclose(p_no_arg[:, 1], 2e-3, atol=1e-14)

    assert sv_no_arg.element0 == 0

    sv_mid_with_init = line.survey(element0='mid',
                            Z0=sv_no_arg['Z', 'mid'],
                            X0=sv_no_arg['X', 'mid'],
                            Y0=sv_no_arg['Y', 'mid'],
                            phi0=sv_no_arg['phi', 'mid'],
                            theta0=sv_no_arg['theta', 'mid'],
                            psi0=sv_no_arg['psi', 'mid'])

    sv_right_with_init = line.survey(element0='right',
                                Z0=sv_no_arg['Z', 'right'],
                                X0=sv_no_arg['X', 'right'],
                                Y0=sv_no_arg['Y', 'right'],
                                phi0=sv_no_arg['phi', 'right'],
                                theta0=sv_no_arg['theta', 'right'],
                                psi0=sv_no_arg['psi', 'right'])

    cols_to_check = [
        'X', 'Y', 'Z', 'theta', 'phi', 'psi', 's', 'drift_length', 'angle',
        'ref_shift_x', 'ref_shift_y', 'ref_rot_x_rad', 'ref_rot_y_rad', 'ref_rot_s_rad',
        'ex', 'ey', 'ez', 'p0', 'frame_matrix'
    ]

    assert sv_mid_with_init.element0 == 13
    assert sv_right_with_init.element0 == 27

    assert np.all(sv_no_arg.name == tw.name)

    for sv_test in sv_mid_with_init, sv_right_with_init:
        assert np.all(sv_test.name == sv_no_arg.name)
        for col in cols_to_check:
            xo.assert_allclose(sv_test[col], sv_no_arg[col], atol=1e-14)

    # Check with no starting from 0 in the middle
    sv_mid_no_init = line.survey(element0='mid')
    tw_init_at_mid = line.twiss4d(betx=1, bety=1, x=1e-3, y=2e-3,
                                init_at='mid')

    p_mid_no_init = tw_init_at_mid.x[:, None] * sv_mid_no_init.ex + \
                    tw_init_at_mid.y[:, None] * sv_mid_no_init.ey + sv_mid_no_init.p0

    xo.assert_allclose(p_mid_no_init[:, 0], 1e-3, atol=1e-14)
    xo.assert_allclose(p_mid_no_init[:, 1], 2e-3, atol=1e-14)