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
    p0c=6500e9, x=1e-3, px=1e-3, y=1e-3, py=1e-3, zeta=1e-3, delta=1e-3)

p_thin = p.copy()
p_thick = p.copy()

line_thin.track(p_thin)
line_thick.track(p_thick)

