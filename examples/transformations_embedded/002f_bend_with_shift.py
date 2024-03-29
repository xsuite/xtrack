import xtrack as xt
import numpy as np

bend = xt.Bend(k0=0., h=0., k1=0.1, length=1)
bend.shift_x = 0.1

line = xt.Line(elements=[bend])

line.configure_bend_model(edge='linear', core='expanded')

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(10000))])
line.build_tracker()
line._line_before_slicing.build_tracker()
assert line['e0..995']._parent_name == 'e0'
assert line['e0..995']._parent is line['e0']
assert line['e0..entry_map']._parent_name == 'e0'
assert line['e0..entry_map']._parent is line['e0']
assert line['e0..exit_map']._parent_name == 'e0'
assert line['e0..exit_map']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=0., px=0., y=0., py=0., delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

line.track(p_slice)
line._line_before_slicing.track(p_ref)

assert_allclose = np.testing.assert_allclose
assert_allclose = np.testing.assert_allclose
assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)
