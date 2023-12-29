import numpy as np

import xtrack as xt


fringe = xt.DipoleEdge(k=0.12, fint=100, hgap=0.035, model='full')

line = xt.Line(elements=[fringe])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, beta0=0.5)
line.build_tracker()

p0 = line.build_particles(px=0.5, py=0.001, y=0.01, delta=0.1)

p_ng = p0.copy()
p_ptc = p0.copy()

R_ng = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy())
line.track(p_ng)
line.config.XTRACK_FRINGE_FROM_PTC = True
R_ptc = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy())
line.track(p_ptc)

assert np.isclose(p_ng.x, p_ptc.x, rtol=0, atol=1e-10)
assert np.isclose(p_ng.px, p_ptc.px, rtol=0, atol=1e-12)
assert np.isclose(p_ng.y, p_ptc.y, rtol=0, atol=1e-12)
assert np.isclose(p_ng.py, p_ptc.py, rtol=0, atol=1e-12)
assert np.isclose(p_ng.delta, p_ptc.delta, rtol=0, atol=1e-12)
assert np.isclose(p_ng.s, p_ptc.s, rtol=0, atol=1e-12)
assert np.isclose(p_ng.zeta, p_ptc.zeta, rtol=0, atol=1e-10)