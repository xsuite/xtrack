import xtrack as xt
import numpy as np

quad = xt.Quadrupole(k1=0.1, length=1,
            #    shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.2
            )

line = xt.Line(elements=[quad])

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(10000))])
line.build_tracker()
assert line['e0..995'].parent_name == 'e0'
assert line['e0..995']._parent is line['e0']
assert line['drift_e0..995'].parent_name == 'e0'
assert line['drift_e0..995']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

line.track(p_slice)
quad.track(p_ref)

assert_allclose = np.testing.assert_allclose
assert_allclose = np.testing.assert_allclose
assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

line.to_json('ttt.json')
line2 = xt.Line.from_json('ttt.json')
assert isinstance(line2['e0..995'], xt.ThinSliceQuadrupole)
assert line2['e0..995'].parent_name == 'e0'
assert line2['e0..995']._parent is None
assert line2['drift_e0..995'].parent_name == 'e0'
assert line2['drift_e0..995']._parent is None

line2.build_tracker()
assert isinstance(line2['e0..995'], xt.ThinSliceQuadrupole)
assert line2['e0..995'].parent_name == 'e0'
assert line2['e0..995']._parent is line2['e0']
assert isinstance(line2['drift_e0..995'], xt.DriftSliceQuadrupole)
assert line2['drift_e0..995'].parent_name == 'e0'
assert line2['drift_e0..995']._parent is line2['e0']

line.track(p_slice, backtrack=True)

assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

line.optimize_for_tracking()

if quad.shift_x !=0 or quad.shift_y != 0 or quad.rot_s_rad != 0:
    assert isinstance(line['e0..995'], xt.Multipole)
else:
    assert isinstance(line['e0..995'], xt.SimpleThinQuadrupole)
assert isinstance(line['drift_e0..995'], xt.Drift)

p_slice = p0.copy()
line.track(p_slice)

assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

line.track(p_slice, backtrack=True)

assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)