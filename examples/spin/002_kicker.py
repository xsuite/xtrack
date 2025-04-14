"""
To intall bmad:
  conda install -c conda-forge bmad
  pip install pytao
"""
from pytao import Tao
import numpy as np
import xtrack as xt

import time

from scipy.constants import c as clight
from scipy.constants import e as qe

p0c =  0.7e9 #700e6# eV
p_ref = xt.Particles(p0c=p0c, delta=0, mass0=xt.ELECTRON_MASS_EV)

delta_vect = np.linspace(-0.01, 0.01, 11)
spin_zeta_bmad = []
spin_zeta_test = []
delta_bmad = []
for dd in delta_vect:

    print('delta', dd)

    delta = dd
    p = p_ref.copy()
    p.delta = delta

    gamma = p.energy[0] / p.energy0[0] * p_ref.gamma0[0]

    brho = p_ref.p0c[0] / clight / p_ref.q0
    length = 0.2

    By_T = 0.023349486663870645
    k0 = By_T / brho

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

    b1: sbend, l={length}, g={k0},  spin_fringe_on = T, fringe_at = no_end! g is h in xtrack
    ! b1: kicker, l={length}, hkick={k0 * length}, spin_fringe_on = F
    dend: drift, l=10.0

    myline: line = (b1, dend)

    use, myline
    """

    with open('lattice.bmad', 'w') as f:
        f.write(input)

    time.sleep(1)

    tao = Tao('-lat lattice.bmad -noplot')

    out = tao.orbit_at_s(s_offset=5)
    delta_bmad.append(out['pz'])



    # gyromagnetic anomaly
    G_spin = 1.15965218128e-3

    B = np.array([0, By_T, 0, 0])

    dyds=B[3]
    Bpar=np.asarray([0,0,(B[2]+dyds*B[1])])
    Bperp=np.asarray([B[0],B[1],-dyds*B[1]])
    if np.sum(Bperp+Bpar)!=0:
        omega=(Bperp+Bpar)/np.sum(Bperp+Bpar)
    else:
        omega=[0,0,0]
    B_0=np.add(Bperp,Bpar)
    # phi=-((G_spin*gamma*B_0[0]+G_spin*gamma*B_0[1]+(1.+G_spin)*B_0[2])*length/brho)
    phi=-((G_spin*gamma*B_0[1])*length/brho) # SPECIFIC FOR VERTICAL FIELD
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
    M=np.asarray([[(t0**2+tx**2)-(ts**2+ty**2),2*(tx*ty+t0*ts)            ,2*(tx*ts+t0*ty)],
                [2*(tx*ty-t0*ts)            ,(t0**2+ty**2)-(tx**2+ts**2),2*(ts*ty+t0*tx)],
                [ 2*(tx*ts-t0*ty)           ,2*(ts*ty-t0*tx)            ,(t0**2+ts**2)-(tx**2+ty**2)]])

    spin_out = M @ np.array(spin_test)
    print('spin_bmad', np.array(out['spin']))
    print('spin_out', spin_out)

    spin_zeta_bmad.append(out['spin'][2])
    spin_zeta_test.append(spin_out[2])

import matplotlib.pyplot as plt

plt.close('all')

plt.figure(1)
plt.plot(delta_vect, spin_zeta_bmad, '.-', label='bmad')
plt.plot(delta_vect, spin_zeta_test, 'x-', label='xtrack')

plt.xlabel('delta')
plt.ylabel('spin zeta')

plt.legend()
