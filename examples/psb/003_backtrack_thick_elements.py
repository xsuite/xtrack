import xtrack as xt
import xpart as xp

b = xt.Bend(k0=0.2, h=0.1, length=1.0)

line = xt.Line(elements=[b])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
line.reset_s_at_end_turn = False
line.build_tracker()

p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                         zeta=0.05, delta=0.01)

p1 = p0.copy()
line.track(p1)

p2 = p1.copy()
line.track(p2, backtrack=True)

for nn in 'x px y py zeta delta'.split():
    print(nn, getattr(p0, nn)[0], getattr(p1, nn)[0], getattr(p2, nn)[0])