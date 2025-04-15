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

def spin_rotation_matrix(Bx_T, By_T, Bz_T, length, p, G_spin):

    gamma = p.energy[0] / p.energy0[0] * p.gamma0[0]
    brho_ref = p.p0c[0] / clight / p.q0
    brho_part = brho_ref * p.rvv[0] * p.energy[0] / p.energy0[0]

    B_vec = np.array([Bx_T, By_T, Bz_T])

    delta_plus_1 = 1 + p.delta[0]
    beta = p.rvv[0] * p.beta0[0]
    kin_px = p.kin_px[0]
    kin_py = p.kin_py[0]
    beta_x = beta * kin_px / delta_plus_1
    beta_y = beta * kin_py / delta_plus_1
    beta_z = np.sqrt(1 - beta_x**2 - beta_y**2)

    beta_v = np.array([beta_x, beta_y, beta_z])

    i_v = beta_v / beta
    B_par = np.dot(B_vec, i_v) * i_v
    B_perp = B_vec - B_par

    # BMAD manual Eq. 24.2
    Omega_BMT = -1/brho_part * (
        (1 + G_spin*gamma) * B_perp + (1 + G_spin) * B_par)
    Omega_BMT_mod = np.sqrt(np.dot(Omega_BMT, Omega_BMT))

    omega = Omega_BMT / Omega_BMT_mod
    # omega = B_vec / np.linalg.norm(B_vec)

    phi = Omega_BMT_mod * length

    # B = np.array([Bx_T, By_T, Bz_T, 0])
    # dyds=B[3]
    # Bpar=np.asarray([0,0,(B[2]+dyds*B[1])])
    # Bperp=np.asarray([B[0],B[1],-dyds*B[1]])
    # if np.sum(Bperp+Bpar)!=0:
    #     omega=(Bperp+Bpar)/np.sum(Bperp+Bpar)
    # else:
    #     omega=[0,0,0]
    # B_0=np.add(Bperp,Bpar)

    # # This works for the corrector
    # phi=-(((G_spin*gamma + 1)*B_0[1])*length/brho_part) # SPECIFIC FOR VERTICAL FIELD

    # This works on momentum for the bend
    # phi=-(((G_spin*gamma)*B_0[1])*length/brho) # SPECIFIC FOR VERTICAL FIELD

    # From BMAD manual Eq. 24.21
    t0=np.cos(phi/2)
    tx=omega[0]*np.sin(phi/2)
    ty=omega[1]*np.sin(phi/2)
    ts=omega[2]*np.sin(phi/2)
    M=np.asarray([[(t0**2+tx**2)-(ts**2+ty**2),2*(tx*ty-t0*ts)          ,2*(tx*ts+t0*ty)],
                [2*(tx*ty+t0*ts)            ,(t0**2+ty**2)-(tx**2+ts**2),2*(ts*ty-t0*tx)],
                [2*(tx*ts-t0*ty)            ,2*(ts*ty+t0*tx)            ,(t0**2+ts**2)-(tx**2+ty**2)]])

    return M

def bmad_kicker(Bx_T, By_T, p0c, delta, length, spin_test, px=0, py=0):

    p_ref = xt.Particles(p0c=p0c, delta=0, mass0=xt.ELECTRON_MASS_EV)
    brho_ref = p_ref.p0c[0] / clight / p_ref.q0

    k0 = By_T / brho_ref
    k0s = Bx_T / brho_ref

    input = f"""

    bmad_com[spin_tracking_on] = T

    parameter[geometry] = open
    parameter[particle] = positron
    parameter[p0c] = {p0c} ! eV

    particle_start[x] = 0
    particle_start[y] = 0
    particle_start[px] = {px} ! this is px
    particle_start[py] = {py} ! this is py
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

    ! b1: sbend, l={length}, g={k0}! g is h in xtrack
    b1: kicker, l={length}, hkick={-k0 * length}, vkick={k0s * length}
    dend: drift, l=10.0

    b1[spin_tracking_method] = Symp_Lie_PTC
    b1[tracking_method] = Symp_Lie_PTC
    b1[SPIN_FRINGE_ON] = F

    myline: line = (b1, dend)

    use, myline
    """

    with open('lattice.bmad', 'w') as f:
        f.write(input)

    time.sleep(1)

    tao = Tao('-lat lattice.bmad -noplot')

    out = tao.orbit_at_s(s_offset=5)

    # ---------
    p = p_ref.copy()
    p.delta = delta
    p.px = px
    p.py = py
    M = spin_rotation_matrix(Bx_T=Bx_T, By_T=By_T, Bz_T=0, length=length,
                            p=p, G_spin=0.00115965218128)
    spin_test = M @ np.array(spin_test)

    out['spin_test'] = spin_test

    return out

By_T = 0.023349486663870645
Bx_T =0.01
p0c = 700e6
spin_test = [0, 0, 1] # spin vector
length = 0.2
delta = 1e-3

out_on_mom = bmad_kicker(Bx_T=Bx_T, By_T=By_T, p0c=p0c, delta=0,
                         length=length, spin_test=spin_test)
out_off_mom_p0c = bmad_kicker(Bx_T=Bx_T, By_T=By_T, p0c=p0c*(1 + delta), delta=0,
                              length=length, spin_test=spin_test)
out_off_mom_delta = bmad_kicker(Bx_T=Bx_T, By_T=By_T, p0c=p0c, delta=delta,
                                length=length, spin_test=spin_test)

delta_vect = np.linspace(-0.01, 0.01, 11)

spin_x_bmad = []
spin_x_test = []
spin_y_bmad = []
spin_y_test = []
spin_z_bmad = []
spin_z_test = []
for dd in delta_vect:
    print('dd', dd)
    out = bmad_kicker(Bx_T=Bx_T, By_T=By_T, p0c=p0c, delta=dd, length=length, spin_test=spin_test)
    spin_z_bmad.append(out['spin'][2])
    spin_z_test.append(out['spin_test'][2])
    spin_x_bmad.append(out['spin'][0])
    spin_x_test.append(out['spin_test'][0])
    spin_y_bmad.append(out['spin'][1])
    spin_y_test.append(out['spin_test'][1])

    print('spin_bmad', np.array(out['spin']))
    print('spin_test', np.array(out['spin_test']))

spin_z_bmad = np.array(spin_z_bmad)
spin_z_test = np.array(spin_z_test)
spin_x_bmad = np.array(spin_x_bmad)
spin_x_test = np.array(spin_x_test)
spin_y_bmad = np.array(spin_y_bmad)
spin_y_test = np.array(spin_y_test)


# Check vs px py
px_vect = np.linspace(-0.01, 0.01, 11)
py_vect = np.linspace(-0.02, 0.02, 11)

spin_x_angle_bmad = []
spin_x_angle_test = []
spin_y_angle_bmad = []
spin_y_angle_test = []
spin_z_angle_bmad = []
spin_z_angle_test = []
for px, py in zip(px_vect, py_vect):
    print('px', px)
    out = bmad_kicker(Bx_T=Bx_T, By_T=By_T, p0c=p0c, delta=0, length=length, spin_test=spin_test,
                        px=px, py=py)
    spin_z_angle_bmad.append(out['spin'][2])
    spin_z_angle_test.append(out['spin_test'][2])
    spin_x_angle_bmad.append(out['spin'][0])
    spin_x_angle_test.append(out['spin_test'][0])
    spin_y_angle_bmad.append(out['spin'][1])
    spin_y_angle_test.append(out['spin_test'][1])
    print('spin_bmad', np.array(out['spin']))
    print('spin_test', np.array(out['spin_test']))

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
plt.plot(delta_vect, spin_z_bmad, '.-', label='bmad')
plt.plot(delta_vect, spin_z_test, 'x-', label='xtrack')
plt.xlabel('delta')
plt.ylabel('spin z')
plt.legend()

plt.figure(2)
plt.plot(delta_vect, spin_x_bmad, '.-', label='bmad')
plt.plot(delta_vect, spin_x_test, 'x-', label='xtrack')
plt.xlabel('delta')
plt.ylabel('spin x')
plt.legend()

plt.figure(3)
plt.plot(delta_vect, spin_y_bmad, '.-', label='bmad')
plt.plot(delta_vect, spin_y_test, 'x-', label='xtrack')
plt.xlabel('delta')
plt.ylabel('spin y')
plt.legend()


plt.figure(11)
plt.plot(px_vect, spin_x_angle_bmad, '.-', label='bmad')
plt.plot(px_vect, spin_x_angle_test, 'x-', label='xtrack')
plt.xlabel('px')
plt.ylabel('spin x')
plt.legend()

plt.figure(12)
plt.plot(py_vect, spin_y_angle_bmad, '.-', label='bmad')
plt.plot(py_vect, spin_y_angle_test, 'x-', label='xtrack')
plt.xlabel('py')
plt.ylabel('spin y')
plt.legend()

plt.figure(13)
plt.plot(px_vect, spin_z_angle_bmad, '.-', label='bmad')
plt.plot(px_vect, spin_z_angle_test, 'x-', label='xtrack')
plt.xlabel('px')
plt.ylabel('spin z')
plt.legend()

plt.show()