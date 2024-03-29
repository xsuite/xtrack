import xtrack as xt
import numpy as np

# bend = xt.Bend(k0=0.1, h=0.11, k1=0.1, length=1,
#                knl=[0, 0, 0.03, 0.4, 0.5],
#                ksl=[0, 0, -0.03, -0.2, -0.1]
# )
# bend.num_multipole_kicks = 10000
# atol=1e-7
# n_teapot = 10000

bend = xt.Bend(k0=0., h=0., k1=0, length=1,
               knl=[0, 0, 0.03, 0.4, 0.5],
               ksl=[0, 0, -0.03, -0.2, -0.1]
)
atol=1e-10
n_teapot = 1

bend.rot_s = 20.
bend.shift_x = 0.1
bend.shift_y = 0.2

line = xt.Line(elements=[bend])

line.configure_bend_model(edge='full', core='expanded')

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(n_teapot))])
line.build_tracker()
line._line_before_slicing.build_tracker()
assert line['e0..0']._parent_name == 'e0'
assert line['e0..0']._parent is line['e0']
assert line['e0..entry_map']._parent_name == 'e0'
assert line['e0..entry_map']._parent is line['e0']
assert line['e0..exit_map']._parent_name == 'e0'
assert line['e0..exit_map']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=1e-3, px=2.e-3, y=3.e-3, py=4.e-3, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

line.track(p_slice)
line._line_before_slicing.track(p_ref)

assert_allclose = np.testing.assert_allclose
assert_allclose = np.testing.assert_allclose
assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=atol)
assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=atol)
assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=atol)
assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=atol)
assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=atol)
assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=atol)


# Test properties

# Test backtracking