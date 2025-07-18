import xtrack as xt


#1 with quadrupole fringe
#2 only dipole fringe

angle = 0.1
b2 = 5
b1 = 0

# dipole = xt.Bend(length=0, k0=b1, k1=b2, edge_exit_angle=angle, edge_exit_model='2')
# sextupole = xt.Multipole(knl = [0, 0, -3/2 * b2 * angle])
# line = xt.Line(elements=[dipole, sextupole])
# dipole_full = xt.Bend(length=0, k0=b1, k1=b2, edge_exit_angle=angle, edge_exit_model='1')
# line2= xt.Line(elements=[dipole_full])

dipole = xt.Bend(length=0, k0=b1, k1=b2, edge_entry_angle=angle, edge_entry_model='2')
sextupole = xt.Multipole(knl = [0, 0, -3/2 * b2 * angle])
line = xt.Line(elements=[sextupole, dipole])
dipole_full = xt.Bend(length=0, k0=b1, k1=b2, edge_entry_angle=angle, edge_entry_model='1')
line2= xt.Line(elements=[dipole_full])

x=0.1
px=0
y=0.1

p0 = xt.Particles(x=x,px=px,y=y)
p1 = p0.copy()

line.discard_tracker()
line.build_tracker(use_prebuilt_kernels=False)
line.track(p0)
print(p0.x, p0.px, p0.y, p0.py)


line2.discard_tracker()
line2.build_tracker(use_prebuilt_kernels=False)
line2.track(p1)
print(p1.x, p1.px, p1.y, p1.py)
