import numpy as np
import xtrack as xt

k2 = 3.
k2s = 5.
length = 0.4

line_thin = xt.Line(elements=[
    xt.Drift(length=length/2),
    xt.Multipole(knl=[0., 0., k2 * length],
                 ksl=[0., 0., k2s * length],
                 length=length),
    xt.Drift(length=length/2),
])
line_thin.build_tracker()

line_thick = xt.Line(elements=[
    xt.Sextupole(k2=k2, k2s=k2s, length=length),
])
line_thick.build_tracker()

p = xt.Particles(
    p0c=6500e9,
    x=[-3e-2, -2e-3, 0, 1e-3, 2e-3, 3e-2],
    px=[1e-6, 2e-6,  0, 2e-6, 1e-6, 1e-6],
    y=[-4e-2, -5e-3, 0, 5e-3, -4e-3, 4e-2],
    py=[2e-6, 4e-6,  0, 2e-6, 1e-6, 1e-6],
    delta=[1e-3, 2e-3, 0, -2e-3, -1e-3, -1e-3],
    zeta=[-5e-2, -6e-3, 0, 6e-3, 5e-3, 5e-2],
)

p_thin = p.copy()
p_thick = p.copy()

line_thin.track(p_thin)
line_thick.track(p_thick)

assert np.allclose(p_thin.x, p_thick.x, rtol=0, atol=1e-14)
assert np.allclose(p_thin.px, p_thick.px, rtol=0, atol=1e-14)
assert np.allclose(p_thin.y, p_thick.y, rtol=0, atol=1e-14)
assert np.allclose(p_thin.py, p_thick.py, rtol=0, atol=1e-14)
assert np.allclose(p_thin.delta, p_thick.delta, rtol=0, atol=1e-14)
assert np.allclose(p_thin.zeta, p_thick.zeta, rtol=0, atol=1e-14)

line_thin.track(p_thin, backtrack=True)
line_thick.track(p_thick, backtrack=True)