import xtrack as xt
import numpy as np

assert_allclose = np.testing.assert_allclose

slice_mode = 'thin'
tilted = True
orientation = 'acw'
transform_to_actual_elements = True

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
line.build_tracker()

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

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(20, mode=slice_mode))])
line.build_tracker()

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
    line.build_tracker()
    assert isinstance(line['e1..1'], xt.Multipole)

sv = line.survey()
assert_allclose(sv.Z[-1], 0, rtol=0, atol=1e-13)
assert_allclose(sv.X[-1], 0, rtol=0, atol=1e-13)
assert_allclose(sv.Y[-1], 0, rtol=0, atol=1e-13)
assert_allclose(sv.s[-1], 8, rtol=0, atol=1e-13)


if not tilted and orientation == 'acw':
    assert_allclose(np.abs(sv.Y), 0, rtol=0, atol=1e-14)
    assert_allclose(np.trapz(sv.X, sv.Z), -4.818 , rtol=0, atol=2e-3) # clockwise
elif not tilted and orientation == 'cw':
    assert_allclose(np.abs(sv.Y), 0, rtol=0, atol=1e-14)
    assert_allclose(np.trapz(sv.X, sv.Z), 4.818 , rtol=0, atol=2e-3)
elif tilted and orientation == 'acw':
    assert_allclose(np.abs(sv.X), 0, rtol=0, atol=1e-14)
    assert_allclose(np.trapz(sv.Y, sv.Z), -4.818 , rtol=0, atol=2e-3)
elif tilted and orientation == 'cw':
    assert_allclose(np.abs(sv.X), 0, rtol=0, atol=1e-14)
    assert_allclose(np.trapz(sv.Y, sv.Z), 4.818 , rtol=0, atol=2e-3)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X)
plt.plot(sv.Z, sv.Y)

plt.figure(2)
plt.plot(sv.s, sv.X)
plt.plot(sv.s, sv.Y)
plt.plot(sv.s, sv.Z)

plt.show()