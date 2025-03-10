import xtrack as xt
import numpy as np

from scipy.constants import hbar
from scipy.constants import electron_volt
from scipy.constants import c as clight

env = xt.load_madx_lattice('b075_2024.09.25.madx')
line = env.ring
line.particle_ref = xt.Particles(energy0=2.7e9, mass0=xt.ELECTRON_MASS_EV)
line.configure_bend_model(num_multipole_kicks=20)

line['vrf'] = 1.8e6
line['frf'] = 499.6e6
line['lagrf'] = 180.

line.insert(
    env.new('cav', 'Cavity', voltage='vrf', frequency='frf', lag='lagrf', at=0))

tt = line.get_table()
tw4d_thick = line.twiss4d()
tw6d_thick = line.twiss()

env['ring_thick'] = env.ring.copy(shallow=True)

line.discard_tracker()
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default
    xt.Strategy(slicing=xt.Teapot(20), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(8), element_type=xt.Quadrupole),
]
line.slice_thick_elements(slicing_strategies)

tw4d = line.twiss4d()
tw6d = line.twiss()

line.configure_radiation(model='mean')

tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

# import matplotlib.pyplot as plt
# plt.close('all')
# pl = tw_rad.plot(yl='y', yr='dy')
# pl.xlim(tw_rad['s', 's.wig'] - 10, tw_rad['s', 'e.wig'] + 10)
# plt.show()

# from synchrotron_integrals import SynchrotronIntegral as synint
# integrals = synint(line)

tw = tw_rad

angle_rad = tw['angle_rad']
rot_s_rad = tw['rot_s_rad']
x = tw['x']
y = tw['y']
kin_px = tw['kin_px']
kin_py = tw['kin_py']
delta = tw['delta']
length = tw['length']

betx = tw['betx']             # Twiss beta function x
alfx = tw['alfx']             # Twiss alpha x
gamx = tw['gamx']             # Twiss gamma x
bety = tw['bety']             # Twiss beta function y
alfy = tw['alfy']             # Twiss alpha y
gamy = tw['gamy']             # Twiss gamma y
dx = tw['dx']                 # Dispersion x
dy = tw['dy']                 # Dispersion y
dpx = tw['dpx']               # Dispersion px
dpy = tw['dpy']               # Dispersion py

mass0 = line.particle_ref.mass0
r0 = line.particle_ref.get_classical_particle_radius0()
gamma0 = tw.gamma0

dxprime = dpx * (1 - delta) - kin_px
dyprime = dpy * (1 - delta) - kin_py

# Curvature of the reference trajectory
mask = length != 0
kappa0_x = 0 * angle_rad
kappa0_y = 0 * angle_rad
kappa0_x[mask] = angle_rad[mask] * np.cos(rot_s_rad[mask]) / length[mask]
kappa0_y[mask] = angle_rad[mask] * np.sin(rot_s_rad[mask]) / length[mask]
kappa0 = np.sqrt(kappa0_x**2 + kappa0_y**2)

# Field index
fieldindex = 0 * angle_rad
k1 = 0 * angle_rad
k1[mask] = tw.k1l[mask] / length[mask]
mask_k0 = kappa0 > 0
fieldindex[mask_k0] = -1. / kappa0[mask_k0]**2 * k1[mask_k0]

# Compute x', y', x'', y''
ps = np.sqrt((1 + delta)**2 - kin_px**2 - kin_py**2)
xp = kin_px / ps
yp = kin_py / ps
xp_ele = xp * 0
yp_ele = yp * 0
xp_ele[:-1] = (xp[:-1] + xp[1:]) / 2
yp_ele[:-1] = (yp[:-1] + yp[1:]) / 2

mask_length = length != 0
xpp_ele = xp_ele * 0
ypp_ele = yp_ele * 0
xpp_ele[mask_length] = np.diff(xp, append=0)[mask_length] / length[mask_length]
ypp_ele[mask_length] = np.diff(yp, append=0)[mask_length] / length[mask_length]

# Curvature of the particle trajectory
hhh = 1 + kappa0_x * x + kappa0_y * y
hprime = kappa0_x * xp_ele + kappa0_y * yp_ele
mask1 = xpp_ele**2 + hhh**2 != 0
mask2 = xpp_ele**2 + hhh**2 != 0
kappa_x = (-(hhh * (xpp_ele - hhh * kappa0_x) - 2 * hprime * xp_ele)[mask1]
           / (xp_ele**2 + hhh**2)[mask1]**(3/2))
kappa_y = (-(hhh * (ypp_ele - hhh * kappa0_y) - 2 * hprime * yp_ele)[mask2]
           / (yp_ele**2 + hhh**2)[mask2]**(3/2))

# Curly H
Hx_rad = gamx * dx**2 + 2*alfx * dx * dxprime + betx * dxprime**2
Hy_rad = gamy * dy**2 + 2*alfy * dy * dyprime + bety * dyprime**2

# Integrands
i1x_integrand = kappa_x * dx
i1y_integrand = kappa_y * dy

i2x_integrand = kappa_x * kappa_x
i2y_integrand = kappa_y * kappa_y

i3x_integrand = kappa_x * kappa_x * kappa_x
i3y_integrand = kappa_y * kappa_y * kappa_y

i4x_integrand = kappa_x * kappa_x * kappa_x * dx * (1 - 2 * fieldindex)
i4y_integrand = kappa_y * kappa_y * kappa_y * dy * (1 - 2 * fieldindex)

i5x_integrand = np.abs(kappa_x*kappa_x*kappa_x) * Hx_rad
i5y_integrand = np.abs(kappa_y*kappa_y*kappa_y) * Hy_rad

# Integrate
i1x = np.sum(i1x_integrand * length)
i1y = np.sum(i1y_integrand * length)
i2x = np.sum(i2x_integrand * length)
i2y = np.sum(i2y_integrand * length)
i3x = np.sum(i3x_integrand * length)
i3y = np.sum(i3y_integrand * length)
i4x = np.sum(i4x_integrand * length)
i4y = np.sum(i4y_integrand * length)
i5x = np.sum(i5x_integrand * length)
i5y = np.sum(i5y_integrand * length)

eq_gemitt_x = (55/(32 * 3**(1/2)) * hbar / electron_volt * clight
               / mass0 * gamma0**2 * i5x / (i2x + i2y - i4x))
eq_gemitt_y = (55/(32 * 3**(1/2)) * hbar / electron_volt * clight
               / mass0 * gamma0**2 * i5y / (i2x + i2y - i4y))

damping_constant_x_s = r0/3 * gamma0**3 * clight/tw.circumference * (i2x + i2y - i4x)
damping_constant_y_s = r0/3 * gamma0**3 * clight/tw.circumference * (i2x + i2y - i4y)
damping_constant_zeta_s = r0/3 * gamma0**3 * clight/tw.circumference * (2 * (i2x + i2y) + i4x + i4y)

tw_integrals = line.twiss(radiation_integrals=True)



tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

print('ex rad int:', tw_integrals.rad_int_eq_gemitt_x)
print('ex Chao:   ', tw_rad.eq_gemitt_x)
print('ey rad int:', tw_integrals.rad_int_eq_gemitt_y)
print('ey Chao:   ', tw_rad.eq_gemitt_y)

print('damping rate x [s^-1] rad int:   ', tw_integrals.rad_int_damping_constant_x_s)
print('damping rate x [s^-1] eigenval:  ', tw_rad.damping_constants_s[0])
print('damping rate y [s^-1] rad int:   ', tw_integrals.rad_int_damping_constant_y_s)
print('damping rate y [s^-1] eigenval:  ', tw_rad.damping_constants_s[1])
print('damping rate z [s^-1] rad int:   ', tw_integrals.rad_int_damping_constant_zeta_s)
print('damping rate z [s^-1] eigenval:  ', tw_rad.damping_constants_s[2])