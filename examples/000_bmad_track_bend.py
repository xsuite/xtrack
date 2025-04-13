"""
To intall bmad:
  conda instakl -c conda-forge bmad
  pip install pytao
"""
from pytao import Tao

input = """

particle_start[x] = 0
particle_start[pz] = 1e-3 ! this is delta

beginning[beta_a]  =  10
beginning[alpha_a]=  0
beginning[beta_b] =   10
beginning[alpha_b] =   0
beginning[eta_x] =  0
beginning[etap_x] =0

parameter[geometry] = open
beginning[e_tot]  = 5.0e+09

b1: sbend, l=4.0, g=1/100.0
dend: drift, l=10.0

myline: line = (b1, dend)

use, myline
"""

with open('lattice.bmad', 'w') as f:
    f.write(input)

tao = Tao('-lat lattice.bmad -noplot')

out = tao.orbit_at_s(s_offset=5)

px_bmad = out.px