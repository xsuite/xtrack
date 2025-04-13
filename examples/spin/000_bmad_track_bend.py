"""
To intall bmad:
  conda instakl -c conda-forge bmad
  pip install pytao
"""
from pytao import Tao

p0c = 5.0e+9 # eV
delta = 1e-3 # momentum deviation


input = f"""

bmad_com[spin_tracking_on] = T

parameter[geometry] = open
parameter[particle] = positron
parameter[p0c] = {p0c} ! eV

particle_start[x] = 0
particle_start[pz] = {delta} ! this is delta

particle_start[spin_x] = 1
particle_start[spin_y] = 0
particle_start[spin_z] = 0

beginning[beta_a]  =  10
beginning[alpha_a]=  0
beginning[beta_b] =   10
beginning[alpha_b] =   0
beginning[eta_x] =  0
beginning[etap_x] =0

b1: sbend, l=4.0, g=1/100.0 ! g is h in xtrack
dend: drift, l=10.0

myline: line = (b1, dend)

use, myline
"""

with open('lattice.bmad', 'w') as f:
    f.write(input)

tao = Tao('-lat lattice.bmad -noplot')

out = tao.orbit_at_s(s_offset=5)

px_bmad = out['px']

import xtrack as xt
env = xt.Environment()
line = env.new_line(components=[
  env.new('bb', xt.Bend, length=4.0, h=1/100.0, k0_from_h=True),
  env.new('observ', xt.Marker, at=5.0),
])

p = xt.Particles(p0c=p0c, delta=delta, mass0=xt.ELECTRON_MASS_EV)
line.track(p)

print('px_bmad', px_bmad)
print('px_xtrack', p.px[0])

spin_bmad = out['spin']
print('spin_bmad', spin_bmad)

