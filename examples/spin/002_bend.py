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

from spin import spin_rotation_matrix

def bmad_bend(By_T, p0c, delta, length, spin_test, px=0, py=0):

    p_ref = xt.Particles(p0c=p0c, delta=0, mass0=xt.ELECTRON_MASS_EV)
    brho_ref = p_ref.p0c[0] / clight / p_ref.q0

    k0 = By_T / brho_ref

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

    b1: sbend, l={length}, g={k0}! g is h in xtrack
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

    p0 = p_ref.copy()
    p0.delta = delta
    p0.px = px
    p0.py = py
    p0.spin_x = spin_test[0]
    p0.spin_y = spin_test[1]
    p0.spin_z = spin_test[2]
    p0.anomalous_magnetic_moment = 0.00115965218128

    p = p0.copy()
    magnet = xt.Magnet(h=k0, k0=k0, length=length)
    magnet.track(p)
    spin_out = [p.spin_x[0], p.spin_y[0], p.spin_z[0]]

    p_python = p0.copy()
    M = spin_rotation_matrix(Bx_T=0, By_T=By_T, Bz_T=0, hx=k0, length=length,
                            p=p_python, G_spin=0.00115965218128)
    spin_out_python = M @ np.array(spin_test)

    out['spin_test'] = spin_out
    out['spin_test_python'] = spin_out_python

    return out

By_T = 0.023349486663870645
p0c = 700e6
spin_test = [0, 0, 1] # spin vector
length = 0.02
delta = 1e-3

out_on_mom = bmad_bend(By_T=By_T, p0c=p0c, delta=0,
                         length=length, spin_test=spin_test)
out_off_mom_p0c = bmad_bend(By_T=By_T, p0c=p0c*(1 + delta), delta=0,
                              length=length, spin_test=spin_test)
out_off_mom_delta = bmad_bend(By_T=By_T, p0c=p0c, delta=delta,
                                length=length, spin_test=spin_test)
out_off_mom_pxpy = bmad_bend(By_T=By_T, p0c=p0c, delta=0,
                                length=length, spin_test=spin_test,
                                px=1e-2, py=2e-3)

delta_vect = np.linspace(-0.01, 0.01, 5)

spin_x_bmad = []
spin_x_test = []
spin_y_bmad = []
spin_y_test = []
spin_z_bmad = []
spin_z_test = []
spin_x_test_python = []
spin_y_test_python = []
spin_z_test_python = []
for dd in delta_vect:
    print('dd', dd)
    out = bmad_bend(By_T=By_T, p0c=p0c, delta=dd, length=length, spin_test=spin_test)
    spin_z_bmad.append(out['spin'][2])
    spin_z_test.append(out['spin_test'][2])
    spin_x_bmad.append(out['spin'][0])
    spin_x_test.append(out['spin_test'][0])
    spin_y_bmad.append(out['spin'][1])
    spin_y_test.append(out['spin_test'][1])
    spin_z_test_python.append(out['spin_test_python'][2])
    spin_x_test_python.append(out['spin_test_python'][0])
    spin_y_test_python.append(out['spin_test_python'][1])

    print('spin_bmad', np.array(out['spin']))
    print('spin_test', np.array(out['spin_test']))

spin_z_bmad = np.array(spin_z_bmad)
spin_z_test = np.array(spin_z_test)
spin_x_bmad = np.array(spin_x_bmad)
spin_x_test = np.array(spin_x_test)
spin_y_bmad = np.array(spin_y_bmad)
spin_y_test = np.array(spin_y_test)
spin_x_test_python = np.array(spin_x_test_python)
spin_y_test_python = np.array(spin_y_test_python)
spin_z_test_python = np.array(spin_z_test_python)


# Check vs px py
px_vect = np.linspace(-0.03, 0.03, 11)
py_vect = np.linspace(-0.02, 0.02, 11)

spin_x_angle_bmad = []
spin_x_angle_test = []
spin_y_angle_bmad = []
spin_y_angle_test = []
spin_z_angle_bmad = []
spin_z_angle_test = []
for px, py in zip(px_vect, py_vect):
    print('px', px)
    out = bmad_bend(By_T=By_T, p0c=p0c, delta=1e-3, length=length, spin_test=spin_test,
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
plt.plot(delta_vect, spin_z_test_python, 'o-', label='xtrack python')
plt.xlabel('delta')
plt.ylabel('spin z')
plt.legend()

plt.figure(2)
plt.plot(delta_vect, spin_x_bmad, '.-', label='bmad')
plt.plot(delta_vect, spin_x_test, 'x-', label='xtrack')
plt.plot(delta_vect, spin_x_test_python, 'o-', label='xtrack python')
plt.xlabel('delta')
plt.ylabel('spin x')
plt.legend()

plt.figure(3)
plt.plot(delta_vect, spin_y_bmad, '.-', label='bmad')
plt.plot(delta_vect, spin_y_test, 'x-', label='xtrack')
plt.plot(delta_vect, spin_y_test_python, 'o-', label='xtrack python')
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
plt.plot(px_vect, spin_y_angle_bmad, '.-', label='bmad')
plt.plot(px_vect, spin_y_angle_test, 'x-', label='xtrack')
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
