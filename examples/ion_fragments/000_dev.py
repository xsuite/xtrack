import xtrack as xt

m = xt.Multipole(knl=[1,2, 3], hxl=0.1, hyl=0.2, length=0.3)
d = xt.Drift(length=1.0)

pref1 = xt.Particles(mass0=3.0, beta0=0.1, q0=2.)
pref2 = xt.Particles(mass0=2.0, beta0=0.2, q0=5.)

p1 = pref1.copy()
p2 = pref2.copy()

p1.delta = 0.1
p1.x = 0.1
p1.y = 0.2
