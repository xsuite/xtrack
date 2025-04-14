"""
To intall bmad:
  conda instakl -c conda-forge bmad
  pip install pytao
"""
from pytao import Tao
import numpy as np

p0c =  700e6# eV
delta = 0. # momentum deviation

spin_test = [1, 0, 0] # spin vector


input = f"""

bmad_com[spin_tracking_on] = T

parameter[geometry] = open
parameter[particle] = positron
parameter[p0c] = {p0c} ! eV

particle_start[x] = 0
particle_start[pz] = {delta} ! this is delta

particle_start[spin_x] = {spin_test[0]}
particle_start[spin_y] = {spin_test[1]}
particle_start[spin_z] = {spin_test[2]}

beginning[beta_a]  =  1
beginning[alpha_a]=  0
beginning[beta_b] =   1
beginning[alpha_b] =   0
beginning[eta_x] =  0
beginning[etap_x] =0

b1: sbend, l=4.0, g=1/100.0,  spin_fringe_on = T, fringe_a = no_end! g is h in xtrack
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
p0 = p.copy()
line.track(p)

print('px_bmad', px_bmad)
print('px_xtrack', p.px[0])

spin_bmad = out['spin']


from scipy.constants import c as clight
from scipy.constants import e as qe


brho = p.p0c[0] / clight / p.q0

By = line['bb'].k0 * brho # Sign to be checked!!!!!
B = np.asarray([0, By, 0, 0])

gamma = p.gamma0[0] * p.energy[0] / p.energy0[0]

# gyromagnetic anomaly
G_spin = 1.159652e-3
G_spin = G_spin
length = line['bb'].length
dzeta = p.zeta[0] - p0.zeta[0]

l_path = p.rvv[0] * (length - dzeta)

dyds=B[3]
Bpar=np.asarray([0,0,(B[2]+dyds*B[1])])
Bperp=np.asarray([B[0],B[1],-dyds*B[1]])
if np.sum(Bperp+Bpar)!=0:
    omega=(Bperp+Bpar)/np.sum(Bperp+Bpar)
else:
    omega=[0,0,0]
B_0=np.add(Bperp,Bpar)
phi=-((G_spin*gamma*B_0[0]+G_spin*gamma*B_0[1]+(1.+G_spin)*B_0[2])*l_path/brho)
sig_x=np.asarray([[0,1],
                  [1,0]])
sig_s=np.asarray([[0,complex(-1)],
                  [complex(1),0]])
sig_y=np.asarray([[1,0],
                  [0,-1]])
t0=np.cos(phi/2)
tx=omega[0]*np.sin(phi/2)
ty=omega[1]*np.sin(phi/2)
ts=omega[2]*np.sin(phi/2)
#    omega0=np.asarray([])
T=np.asarray([[complex(t0,ty),complex(ts,tx)],
              [complex(-ts,tx),complex(t0,-ty)]])
phi2=(2*np.arccos(np.trace(T)/2))
omega2=np.asarray([np.trace(np.dot(T,sig_x)),np.trace(np.dot(T,sig_y)),np.trace(np.dot(T,sig_s))])
M=np.asarray([[(t0**2+tx**2)-(ts**2+ty**2),2*(tx*ty+t0*ts)            ,2*(tx*ts+t0*ty)],
              [2*(tx*ty-t0*ts)            ,(t0**2+ty**2)-(tx**2+ts**2),2*(ts*ty+t0*tx)],
              [ 2*(tx*ts-t0*ty)           ,2*(ts*ty-t0*tx)            ,(t0**2+ts**2)-(tx**2+ty**2)]])

spin_out = M @ np.array(spin_test)
print('spin_bmad', np.array(spin_bmad))
print('spin_out', spin_out)
