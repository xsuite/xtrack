import xtrack as xt
import numpy as np

bend = xt.Bend(k0=0.4, h=0.3, k1=0.1, length=1)

line = xt.Line(elements=[bend])
line.build_tracker() # Put everything in the same buffer
line.discard_tracker()

# Shallow copy
line_before_slicing = xt.Line.__new__(xt.Line)
line_before_slicing.__dict__.update(line.__dict__)
# Deep copy of element_names and comoupound_container
line_before_slicing.element_names = line.element_names.copy()
line_before_slicing.compound_container = line.compound_container.copy()

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(10000))])
line.build_tracker()
assert line['e0..995']._parent_name == 'e0'
assert line['e0..995']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

line.track(p_slice)
bend.track(p_ref)

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
assert line2['e0..995']._parent_name == 'e0'
assert line2['e0..995']._parent is None

line2.build_tracker()
assert isinstance(line2['e0..995'], xt.ThinSliceQuadrupole)
assert line2['e0..995']._parent_name == 'e0'
assert line2['e0..995']._parent is line2['e0']

