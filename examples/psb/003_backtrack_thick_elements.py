import numpy as np

import xtrack as xt

b = xt.Bend(k0=0.2, h=0.1, length=1.0)

line = xt.Line(elements=[b])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, beta0=0.5)
line.reset_s_at_end_turn = False
line.build_tracker()

p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                         zeta=0.05, delta=0.01)

p1 = p0.copy()
line.track(p1)

p2 = p1.copy()
line.track(p2, backtrack=True)

print('\nBend:')
for nn in 's x px y py zeta delta'.split():
    print(f"{nn}: {getattr(p0, nn)[0]:.6e} {getattr(p1, nn)[0]:.6e} {getattr(p2, nn)[0]:.6e}")

assert np.allclose(p2.s, p0.s, atol=1e-15, rtol=0)
assert np.allclose(p2.x, p0.x, atol=1e-15, rtol=0)
assert np.allclose(p2.px, p0.px, atol=1e-15, rtol=0)
assert np.allclose(p2.y, p0.y, atol=1e-15, rtol=0)
assert np.allclose(p2.py, p0.py, atol=1e-15, rtol=0)
assert np.allclose(p2.zeta, p0.zeta, atol=1e-15, rtol=0)
assert np.allclose(p2.delta, p0.delta, atol=1e-15, rtol=0)

p3 = p1.copy()
line.configure_bend_model(core='full')
line.track(p3, backtrack=True)

assert np.all(p3.state == -30)

p4 = p1.copy()
line.configure_bend_model(core='expanded')
b.num_multipole_kicks = 3
line.track(p4, backtrack=True)

assert np.all(p4.state == -31)

# Same for quadrupole
q = xt.Quadrupole(k1=0.2, length=1.0)
assert np.allclose(p2.s, p0.s, atol=1e-15, rtol=0)
assert np.allclose(p2.x, p0.x, atol=1e-15, rtol=0)
assert np.allclose(p2.px, p0.px, atol=1e-15, rtol=0)
assert np.allclose(p2.y, p0.y, atol=1e-15, rtol=0)
assert np.allclose(p2.py, p0.py, atol=1e-15, rtol=0)
assert np.allclose(p2.zeta, p0.zeta, atol=1e-15, rtol=0)
assert np.allclose(p2.delta, p0.delta, atol=1e-15, rtol=0)

line = xt.Line(elements=[q])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, beta0=0.5)
line.reset_s_at_end_turn = False
line.build_tracker()

p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                            zeta=0.05, delta=0.01)

p1 = p0.copy()
line.track(p1)

p2 = p1.copy()
line.track(p2, backtrack=True)

print('\nQuadrupole:')
for nn in 's x px y py zeta delta'.split():
    print(f"{nn}: {getattr(p0, nn)[0]:.6e} {getattr(p1, nn)[0]:.6e} {getattr(p2, nn)[0]:.6e}")

p4 = p1.copy()
q.num_multipole_kicks = 4
line.track(p4, backtrack=True)

assert np.all(p4.state == -31)

de = xt.DipoleEdge(e1=0.1, k=3, fint=0.3)
line = xt.Line(elements=[de])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, beta0=0.5)
line.reset_s_at_end_turn = False
line.build_tracker()

p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                            zeta=0.05, delta=0.01)

p1 = p0.copy()
line.track(p1)

assert np.all(p1.state == 1)

line.configure_bend_model(edge='full')
p2 = p1.copy()
line.track(p2, backtrack=True)

assert np.all(p2.state == -32)