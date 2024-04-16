import xtrack as xt
import numpy as np

bend = xt.Bend(k0=0.4, h=0.3, k1=0.1, length=1)

line = xt.Line(elements=[bend])

line.configure_bend_model(edge='linear', core='expanded')

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(5, mode='thick'))])
line.build_tracker()
line._line_before_slicing.build_tracker()
assert line['e0..3'].parent_name == 'e0'
assert line['e0..3']._parent is line['e0']
assert line['e0..entry_map'].parent_name == 'e0'
assert line['e0..entry_map']._parent is line['e0']
assert line['e0..exit_map'].parent_name == 'e0'
assert line['e0..exit_map']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
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

line.to_json('ttt.json')
line2 = xt.Line.from_json('ttt.json')
assert isinstance(line2['e0..3'], xt.ThickSliceBend)
assert line2['e0..3'].parent_name == 'e0'
assert line2['e0..3']._parent is None
assert line2['e0..entry_map'].parent_name == 'e0'
assert line2['e0..entry_map']._parent is None
assert line2['e0..exit_map'].parent_name == 'e0'
assert line2['e0..exit_map']._parent is None

line2.build_tracker()
assert isinstance(line2['e0..3'], xt.ThickSliceBend)
assert line2['e0..3'].parent_name == 'e0'
assert line2['e0..3']._parent is line2['e0']
assert line2['e0..entry_map'].parent_name == 'e0'
assert line2['e0..entry_map']._parent is line2['e0']
assert line2['e0..exit_map'].parent_name == 'e0'
assert line2['e0..exit_map']._parent is line2['e0']

line.track(p_slice, backtrack=True)

assert (p_slice.state == 1).all()
assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)
