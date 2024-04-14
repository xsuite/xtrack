import xtrack as xt
import numpy as np

assert_allclose = np.testing.assert_allclose

k0 = 2.
length = 0.1
rot_s_rad = 0.2

bend = xt.Bend(k0=k0, length=length, rot_s_rad=rot_s_rad)

assert bend.shift_x == 0
assert bend.shift_y == 0
assert_allclose(bend._sin_rot_s, np.sin(rot_s_rad), rtol=0, atol=1e-14)
assert_allclose(bend._cos_rot_s, np.cos(rot_s_rad), rtol=0, atol=1e-14)

p = xt.Particles(x=0, p0c=1e12)
bend.track(p)

assert_allclose = np.testing.assert_allclose
assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3, atol=0)
assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3, atol=0)


# Change the shift
rot_s_rad = 0.3
bend.rot_s_rad = rot_s_rad
p = xt.Particles(x=0, p0c=1e12)
bend.track(p)
assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3, atol=0)
assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3, atol=0)

# Make a line
line = xt.Line(elements=[bend])

# Slice the line:
line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Uniform(3))])
line.build_tracker()

tt = line.get_table()
assert len(tt.rows['e0\.\..?']) == 3

p = xt.Particles(x=0, p0c=1e12)
line.track(p)

assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3, atol=0)
assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3, atol=0)

# Change the shift
rot_s_rad = 0.4
bend.rot_s_rad = rot_s_rad
p = xt.Particles(x=0, p0c=1e12)
line.track(p)

assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3, atol=0)
assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3, atol=0)