import xtrack as xt
import xobjects as xo

d1=xt.Drift(length=1)
d2=xt.Drift(length=1)
d3=xt.Drift(length=1)

d2.iscollective=True

line=xt.Line([d1,d2,d3])
line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
t = line.twiss4d(betx=1,bety=1,include_collective=True)

ddd1=xt.Drift(length=1)
ddd2=xt.Drift(length=1)
ddd3=xt.Drift(length=1)
line2=xt.Line([ddd1,ddd2,ddd3])
line2.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
t2 = line2.twiss4d(betx=1,bety=1,include_collective=True)

xo.assert_allclose(t.betx, t2.betx, atol=1e-12, rtol=0)

