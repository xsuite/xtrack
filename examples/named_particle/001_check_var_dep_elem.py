import xtrack as xt

lhc = xt.Environment()
# lhc.particles['particle_ref/b1'] = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1)

lhc.new_line(name='b1', length=10, components=[
    lhc.new('q', 'Quadrupole', length=2, at=5)
])

lhc['a'] = 3
lhc['q'].k1 = 'a'

lhc['b'] = lhc.ref['q'].k1

lhc2 = xt.Line.from_dict(lhc.to_dict())
