import numpy as np
from pathlib import Path

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0 as eps0

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


ctx = xo.ContextCpu()

boris_knl_description = xo.Kernel(
    c_name='boris_step',
    args=[
        xo.Arg(xo.Int64,   name='N_sub_steps'),
        xo.Arg(xo.Float64, name='Dtt'),
        xo.Arg(xo.Float64, name='B_field', pointer=True),
        xo.Arg(xo.Float64, name='B_skew', pointer=True),
        xo.Arg(xo.Float64, name='xn1', pointer=True),
        xo.Arg(xo.Float64, name='yn1', pointer=True),
        xo.Arg(xo.Float64, name='zn1', pointer=True),
        xo.Arg(xo.Float64, name='vxn1', pointer=True),
        xo.Arg(xo.Float64, name='vyn1', pointer=True),
        xo.Arg(xo.Float64, name='vzn1', pointer=True),
        xo.Arg(xo.Float64, name='Ex_n', pointer=True),
        xo.Arg(xo.Float64, name='Ey_n', pointer=True),
        xo.Arg(xo.Float64, name='Bx_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='By_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='Bz_n_custom', pointer=True),
        xo.Arg(xo.Int64,   name='custom_B'),
        xo.Arg(xo.Int64,   name='N_mp'),
        xo.Arg(xo.Int64,   name='N_multipoles'),
        xo.Arg(xo.Float64, name='charge'),
        xo.Arg(xo.Float64, name='mass', pointer=True),
    ],
)

ctx.add_kernels(
    kernels={'boris': boris_knl_description},
    sources=[xt._pkg_root / '_temp/boris_and_solenoid_map/boris.h'],
)

# tag = 'case0'
# orbit_lims = -5, 5
# p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
#                  energy0=45.6e9,
#                  x=1e-3, px=0,
#                  y=1e-3, py=0,
#                  delta=0)

# tag = 'case1'
# orbit_lims = -100, 100
# p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
#                  energy0=45.6e9,
#                  x=-170e-3, px=15e-3,
#                  y=10e-3, py=-5e-3,
#                  delta=0)

# tag = 'case2'
# orbit_lims = -100, 100
# delta = -0.1
# p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
#                  energy0=45.6e9,
#                  x=-170e-3, px=15e-3 * (1+delta),
#                  y=10e-3, py=-5e-3 * (1+delta),
#                  delta=delta)

# tag = 'case3'
# orbit_lims = -100, 100
# delta = -0.99
# p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
#                  energy0=45.6e9,
#                  x=-170e-3, px=15e-3 * (1+delta),
#                  y=10e-3, py=-5e-3 * (1+delta),
#                  delta=delta)

# tag = 'case4'
# orbit_lims = -100, 100
# delta = -0.999
# p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
#                  energy0=45.6e9,
#                  x=-170e-3, px=15e-3 * (1+delta),
#                  y=10e-3, py=-5e-3 * (1+delta),
#                  delta=delta)

tag = 'case5'
orbit_lims = -100, 100
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                 energy0=45.6e9,
                 x=-150e-3, px=15e-3,
                 y=30e-3, py=-5e-3,
                 delta=0)

p = p0.copy()

z_sol_center = 10
sf = SolenoidField(L=4, a=0.3, B0=1., z0=z_sol_center)

dt = 1e-10
n_steps = 1500

x_log = []
y_log = []
z_log = []
px_log = []
py_log = []
pp_log = []
pow_log = []
pow_x_log = []
pow_y_log = []
Bx_log = []
By_log = []
Bz_log = []

for ii in range(n_steps):

    x = p.x.copy()
    y = p.y.copy()
    z = p.s.copy()

    Bx, By, Bz = sf.get_field(x, y, z)

    gamma = p.energy / p.mass0
    mass0_kg = p.mass0 * qe / clight**2
    charge0_coulomb = p.q0 * qe

    p0c_J = p.p0c * qe

    Pxc_J = p.px * p0c_J
    Pyc_J = p.py * p0c_J
    Pzc_J = np.sqrt((p0c_J*(1 + p.delta))**2 - Pxc_J**2 - Pyc_J**2)

    vx = Pxc_J / clight / (gamma * mass0_kg) # m/s
    vy = Pyc_J / clight / (gamma * mass0_kg) # m/s
    vz = Pzc_J / clight / (gamma * mass0_kg) # m/s

    vx_before = vx.copy()
    vy_before = vy.copy()

    ctx.kernels.boris(
            N_sub_steps=1,
            Dtt=dt,
            B_field=np.array([0.]),
            B_skew=np.array([0.]),
            xn1=x,
            yn1=y,
            zn1=z,
            vxn1=vx,
            vyn1=vy,
            vzn1=vz,
            Ex_n=0 * x,
            Ey_n=0 * x,
            Bx_n_custom=Bx,
            By_n_custom=By,
            Bz_n_custom=Bz,
            custom_B=1,
            N_mp=len(x),
            N_multipoles=0,
            charge=charge0_coulomb,
            mass=mass0_kg * gamma,
    )

    p.x = x
    p.y = y
    p.s = z
    p.px = mass0_kg * gamma * vx * clight / p0c_J
    p.py = mass0_kg * gamma * vy * clight / p0c_J
    pz = mass0_kg * gamma * vz * clight / p0c_J
    pp = np.sqrt(p.px**2 + p.py**2 + pz**2)

    beta_x_before = vx_before / clight
    beta_y_before = vy_before / clight

    beta_x_after = vx / clight
    beta_y_after = vy / clight

    beta_x_dot = (beta_x_after - beta_x_before) / dt
    beta_y_dot = (beta_y_after - beta_y_before) / dt

    bet_dot_square = beta_x_dot**2 + beta_y_dot**2

    # From Hofmann, "The physics of synchrontron radiation" Eq 3.7 and below
    pow = 2 * qe**2 * bet_dot_square * gamma**4 / (12 * np.pi * eps0 * clight)

    x_log.append(p.x.copy())
    y_log.append(p.y.copy())
    z_log.append(p.s.copy())
    px_log.append(p.px.copy())
    py_log.append(p.py.copy())
    pp_log.append(pp)
    pow_log.append(pow)
    Bx_log.append(Bx)
    By_log.append(By)
    Bz_log.append(Bz)

x_log = np.array(x_log)
y_log = np.array(y_log)
z_log = np.array(z_log)
px_log = np.array(px_log)
py_log = np.array(py_log)
pp_log = np.array(pp_log)
Bx_log = np.array(Bx_log)
By_log = np.array(By_log)
Bz_log = np.array(Bz_log)

pow_log = np.array(pow_log)

dE_ds_boris_J = 0 * pow_log

dE_ds_boris_J[:-1]= pow_log[:-1] * dt / np.diff(z_log, axis=0)
dE_ds_boris_eV = dE_ds_boris_J / qe

z_axis = np.linspace(0, 30, 1001)
Bz_axis = sf.get_field(0 * z_axis, 0 * z_axis, z_axis)[2]

P0_J = p.p0c[0] * qe / clight
brho = P0_J / qe / p.q0

#ks = 0.5 * (Bz_axis[:-1] + Bz_axis[1:]) / brho
ks = Bz_axis[:-1] / brho

line = xt.Line(elements=[xt.Solenoid(length=z_axis[1]-z_axis[0], ks=ks[ii])
                            for ii in range(len(z_axis)-1)])
line.build_tracker()
line.configure_radiation(model='mean')

p_xt = p0.copy()
line.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track

p_xt = p0.copy()
line.configure_radiation(model=None)
line.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
mon_no_rad = line.record_last_track

Bz_mid = 0.5 * (Bz_axis[:-1] + Bz_axis[1:])
Bz_mon = 0 * Bz_axis
Bz_mon[1:] = Bz_mid

# Wolsky Eq. 3.114
Ax = -0.5 * Bz_mon * mon.y
Ay =  0.5 * Bz_mon * mon.x

# Wolsky Eq. 2.74
ax_ref = Ax * p0.q0 * qe / P0_J
ay_ref = Ay * p0.q0 * qe / P0_J

px_mech = mon.px - ax_ref
py_mech = mon.py - ay_ref
pz_mech = np.sqrt((1 + mon.delta)**2 - px_mech**2 - py_mech**2)

xp = px_mech / pz_mech
yp = py_mech / pz_mech

dx_ds = np.diff(mon.x, axis=1) / np.diff(mon.s, axis=1)
dy_ds = np.diff(mon.y, axis=1) / np.diff(mon.s, axis=1)

dE_ds = -np.diff(mon.ptau, axis=1)/np.diff(mon.s, axis=1) * p_xt.energy0[0]

emitted_dpx = -(np.diff(mon.px, axis=1) - np.diff(mon_no_rad.px, axis=1))
emitted_dpy = -(np.diff(mon.py, axis=1) - np.diff(mon_no_rad.py, axis=1))
emitted_dp = -(np.diff(mon.delta, axis=1) - np.diff(mon_no_rad.delta, axis=1))

z_check = sf.z0 + sf.L * np.linspace(-2, 2, 1001)

for i_part in range(z_log.shape[1]):

    this_s_boris = 0.5 * (z_log[:-1, i_part] + z_log[1:, i_part])
    dx_ds_boris = np.diff(x_log[:, i_part]) / np.diff(z_log[:, i_part])
    dy_ds_boris = np.diff(y_log[:, i_part]) / np.diff(z_log[:, i_part])

    s_xsuite = 0.5 * (mon.s[i_part, :-1] + mon.s[i_part, 1:])
    dx_ds_xsuite = np.diff(mon.x[i_part, :]) / np.diff(mon.s[i_part, :])
    dy_ds_xsuite = np.diff(mon.y[i_part, :]) / np.diff(mon.s[i_part, :])
    dE_ds_xsuite = dE_ds[i_part, :]

    dx_ds_xsuite_check = np.interp(z_check, s_xsuite, dx_ds_xsuite)
    dy_ds_xsuite_check = np.interp(z_check, s_xsuite, dy_ds_xsuite)
    dE_ds_xsuite_check = np.interp(z_check, s_xsuite, dE_ds_xsuite)

    dx_ds_boris_check = np.interp(z_check, this_s_boris, dx_ds_boris)
    dy_ds_boris_check = np.interp(z_check, this_s_boris, dy_ds_boris)
    dE_ds_boris_check = np.interp(z_check, z_log[:, i_part], dE_ds_boris_eV[:, i_part])

    this_emitted_dpx = emitted_dpx[i_part, :]
    this_emitted_dpy = emitted_dpy[i_part, :]
    this_dE_ds = dE_ds[i_part, :]
    this_dx_ds = dx_ds[i_part, :]
    this_dy_ds = dy_ds[i_part, :]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(z_log, x_log, label='Boris')
plt.plot(mon.s.T, mon.x.T, '--', label='xsuite')
plt.ylabel('x [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(z_axis, Bz_axis)
plt.ylabel(r'$B_{z}$ [T]')
plt.xlabel('z [m]')

plt.figure(2)
dx_ds_log = np.diff(x_log, axis=0) / np.diff(z_log, axis=0)
dy_ds_log = np.diff(y_log, axis=0) / np.diff(z_log, axis=0)
ax1 = plt.subplot(2, 1, 1)
plt.plot(0.5 * (z_log[:-1, :] + z_log[1:, :]), dx_ds_log, label='Boris')
plt.plot(mon.s.T, xp.T, label="x'", color='C1', linestyle='-')
plt.plot(mon.s[:, :-1].T, dx_ds.T, '--', color='C2',
         label=r"$\Delta x / \Delta s$")
plt.ylabel(r"$x'$")
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(0.5 * (z_log[:-1] + z_log[1:]), dy_ds_log, label='Boris')
plt.plot(mon.s.T, yp.T, label="y'", color='C1', linestyle='-')
plt.plot(mon.s[:, :-1].T, dy_ds.T, '--', color='C2',
         label=r"$\Delta y / \Delta s$")


# Compare ax and ay
plt.figure(3)
plt.plot(mon.s.T, ax_ref.T, label="ax_ref", color='C0', linestyle='-')
plt.plot(mon.s.T, ay_ref.T, label="ay_ref", color='C1', linestyle='-')
plt.plot(mon.s.T, mon.ax.T, label="ax", color='C2', linestyle='--')
plt.plot(mon.s.T, mon.ay.T, label="ay", color='C3', linestyle='--')

plt.figure(4)
plt.plot(mon.s[:, :-1].T, dE_ds.T * 1e-2 * 1e-3, '.-', label='dE/ds')
plt.plot(z_log, dE_ds_boris_eV * 1e-2 * 1e-3, 'x-', label='dE/ds Boris')

plt.figure(5)

ax1 = plt.subplot(2, 1, 1)
plt.plot(mon.s[:, :-1].T, emitted_dpx.T, '-', label='dpx')
plt.plot(mon.s[:, :-1].T, dE_ds.T * dx_ds.T*np.diff(mon.s, 1).T/p.p0c[0], '--')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(mon.s[:, :-1].T, emitted_dpy.T, '-', label='dpy')
plt.plot(mon.s[:, :-1].T, dE_ds.T * dy_ds.T*np.diff(mon.s, 1).T/p.p0c[0], '--')

i_part_plot = 0

plt.figure(100, figsize=(6.4*1.1, 4.8*1.7))


sp1 = plt.subplot(5, 1, 1)
# plt.plot(z_axis - z_sol_center, Bz_axis)
plt.plot(0.5 * (z_log[:-1, i_part] + z_log[1:, i_part]) - z_sol_center, Bz_log[:-1, i_part])
plt.ylabel(r'$B_{z}$ [T]')

sp2 = plt.subplot(5, 1, 2, sharex=sp1)
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e3 * mon.x[i_part_plot, :])
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e3 * mon.y[i_part_plot, :])
plt.axhline(0, color='grey', alpha=0.6, linestyle=':')
plt.ylim(*orbit_lims)
plt.ylabel('x, y [mm]')

px = mon.px[i_part_plot, :]
kin_px = mon.kin_px[i_part_plot, :]
kin_px_boris = px_log[:, i_part_plot]

py = mon.py[i_part_plot, :]
kin_py = mon.kin_py[i_part_plot, :]
kin_py_boris = py_log[:, i_part_plot]

sp3 = plt.subplot(5, 1, 3, sharex=sp1)
plt.plot(0.5 * (z_log[:-1, i_part] + z_log[1:, i_part]) - z_sol_center, Bx_log[:-1, i_part])
plt.plot(0.5 * (z_log[:-1, i_part] + z_log[1:, i_part]) - z_sol_center, By_log[:-1, i_part])
plt.ylabel(r'$B_{x,y}$ [T]')
plt.legend([r'$B_x$', r'$B_y$'], loc='upper left')

sp4 = plt.subplot(5, 1, 4, sharex=sp1)
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (px - px[0]),
         label=r'$p_x$ (xsuite)')
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (kin_px - kin_px[0]),
         label=r"$p_x^\text{kin}$ (xsuite)")
plt.plot(z_log[:, i_part_plot] - z_sol_center, 1e6 * (kin_px_boris - kin_px[0]),
         label=r"$p_x^\text{kin}$ (Boris)", linestyle='--')
plt.legend(fontsize='medium', loc='upper left')
plt.ylabel(r"$\Delta p_x$ [$10^{-6}$]")

sp5 = plt.subplot(5, 1, 5, sharex=sp1)
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (py - py[0]),
         label=r'$p_y$ (xsuite)')
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (kin_py - kin_py[0]),
         label=r"$p_y^\text{kin}$ (xsuite)")
plt.plot(z_log[:, i_part_plot] - z_sol_center, 1e6 * (kin_py_boris - kin_py[0]),
         label=r"$p_y^\text{kin}$ (Boris)", linestyle='--')
plt.legend(fontsize='medium', loc='upper left')
plt.ylabel(r"$\Delta p_y$ [$10^{-6}$]")

plt.xlim(-6, 6)
plt.xlabel('s [m]')

plt.subplots_adjust(top=.91, bottom=.06, hspace=.3, left=.14)
plt.suptitle(r'$\Delta s_\text{slices}$ = ' f'{np.diff(z_log, axis=0).mean():.2f}, '
             r'$E_0$ = ' f'{p.energy0[0]*1e-9:.1f} GeV, '
             r'$\delta$ = ' f'{p.delta[0]:.3f} '
             r'($E$ = ' f'{p.energy[0]*1e-9:.3f} GeV) \n'
             r"Initial $x'$ = " f"{mon.kin_xprime[i_part_plot, 0]*1e3:.2f} mrad, "
             r"initial $y'$ = " f"{mon.kin_yprime[i_part_plot, 0]*1e3:.2f} mrad")

plt.savefig(f'benchmark_{tag}.png', dpi=200)


plt.figure(101)
plt.plot(mon.s[i_part_plot, :-1] - z_sol_center, dE_ds[i_part_plot, :]* 1e-2,
            label='xsuite')
plt.plot(z_log[:, i_part_plot] - z_sol_center, dE_ds_boris_eV *1e-2,
         label="Boris", linestyle='--')
plt.ylabel(r"$dE/ds$ [eV/cm]")
plt.legend(fontsize='medium', loc='upper left')
plt.xlim(-6, 6)
plt.xlabel('s [m]')
plt.suptitle(r'$\Delta s_\text{slices}$ = ' f'{np.diff(z_log, axis=0).mean():.2f}, '
             r'$E_0$ = ' f'{p.energy0[0]*1e-9:.1f} GeV, '
             r'$\delta$ = ' f'{p.delta[0]:.3f} '
             r'($E$ = ' f'{p.energy[0]*1e-9:.3f} GeV) \n'
             r"Initial $x'$ = " f"{mon.kin_xprime[i_part_plot, 0]*1e3:.2f} mrad, "
             r"initial $y'$ = " f"{mon.kin_yprime[i_part_plot, 0]*1e3:.2f} mrad")

plt.show()