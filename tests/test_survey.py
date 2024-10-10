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
