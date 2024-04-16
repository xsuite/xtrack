import xtrack as xt
import numpy as np

drift = xt.Drift(length=1)

line = xt.Line(elements=[drift])

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Uniform(5), element_type=xt.Drift)])
line.build_tracker()
assert line['drift_e0..0'].parent_name == 'e0'
assert line['drift_e0..0']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

line.track(p_slice)
drift.track(p_ref)

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
assert line2['drift_e0..0'].parent_name == 'e0'
assert line2['drift_e0..0']._parent is None

line2.build_tracker()
assert isinstance(line2['drift_e0..0'], xt.DriftSlice)
assert line2['drift_e0..0'].parent_name == 'e0'
assert line2['drift_e0..0']._parent is line2['e0']

line.track(p_slice, backtrack=True)

assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

line.optimize_for_tracking()

assert isinstance(line['drift_e0..0'], xt.Drift)

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