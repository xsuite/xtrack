import numpy as np
from pathlib import Path

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0 as eps0

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

ctx = xo.ContextCpu()

# p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
#                  energy0=45.6e9,
#                  x=-140e-3, px=10e-3,
#                  y=1e-3, py=0,
#                  delta=0)
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                 energy0=45.6e9,
                 x=1e-3, px=1e-3,
                 y=1e-3, py=1e-3,
                 delta=0)


p = p0.copy()

z_sol_center = 15
sf = SolenoidField(L=4, a=0.3, B0=2, z0=z_sol_center)

z_axis = np.linspace(0, 30, 200)
Bz_axis = sf.get_field(0 * z_axis, 0 * z_axis, z_axis)[2]

z_fine = np.linspace(0, 30, 1000)
Bz_fine = sf.get_field(0 * z_fine, 0 * z_fine, z_fine)[2]

P0_J = p.p0c[0] * qe / clight
brho = P0_J / qe / p.q0

#ks = 0.5 * (Bz_axis[:-1] + Bz_axis[1:]) / brho
ks = Bz_axis[:-1] / brho

line = xt.Line(elements=[xt.Solenoid(length=z_axis[1]-z_axis[0], ks=ks[ii])
                            for ii in range(len(z_axis)-1)])
line.cut_at_s(np.linspace(0, line.get_length(), 1000))
line.build_tracker()

p_xt = p0.copy()
line.configure_radiation(model=None)
line.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track

Bz_mid = 0.5 * (Bz_axis[:-1] + Bz_axis[1:])
Bz_mon = 0 * Bz_axis
Bz_mon[1:] = Bz_mid


import matplotlib.pyplot as plt
plt.close('all')

toplot = ['canonical', 'potential', 'prime']
toplot = ['canonical']

i_part_plot = 0
dz = z_axis[1] - z_axis[0]

plt.figure(100, figsize=(6.4, 4.8*1.6))

sp1 = plt.subplot(4, 1, 1)
plt.bar(z_axis - dz/2 - z_sol_center, Bz_axis, width=z_axis[1]-z_axis[0],
        alpha=0.5, align='edge', linewidth=1, edgecolor='C0')
plt.plot(z_fine- z_sol_center, Bz_fine)
plt.ylabel(r'$B_{s}$ [T]')

sp2 = plt.subplot(4, 1, 2, sharex=sp1)
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e3 * mon.x[i_part_plot, :], label='x')
plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e3 * mon.y[i_part_plot, :], label='y')
plt.ylim(-5, 30)
plt.grid()
plt.ylabel('[mm]')
plt.legend(fontsize='medium', loc='upper left')

px = mon.px[i_part_plot, :]
kin_px = mon.kin_px[i_part_plot, :]
kin_xprime = mon.kin_xprime[i_part_plot, :]

py = mon.py[i_part_plot, :]
kin_py = mon.kin_py[i_part_plot, :]
kin_yprime = mon.kin_yprime[i_part_plot, :]

sp3 = plt.subplot(4, 1, 3, sharex=sp1)
if 'canonical' in toplot:
    plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (px - px[0]),
             label=r'$p_x$')
if 'potential' in toplot:
    plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * mon.ax[i_part_plot, :],
             label=r'$a_x$')
if 'prime' in toplot:
    plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (kin_xprime - kin_xprime[0]),
            label=r"x'", color='C2', linewidth=2)
plt.grid()
plt.legend(fontsize='medium', loc='upper left')
plt.ylabel(r"[$10^{-6}$]")
plt.ylim(-150, 150)

sp4 = plt.subplot(4, 1, 4, sharex=sp1)
if 'canonical' in toplot:
    plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (py - py[0]),
             label=r'$p_y$')
if 'potential' in toplot:
    plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * mon.ay[i_part_plot, :],
             label=r'$a_y$')
if 'prime' in toplot:
    plt.plot(mon.s[i_part_plot, :] - z_sol_center, 1e6 * (kin_yprime - kin_yprime[0]),
         label=r"$y'$", color='C2', linewidth=2)
plt.grid()
plt.legend(fontsize='medium', loc='upper left')
plt.ylabel(r"[$10^{-6}$]")
plt.ylim(-150, 150)

plt.xlim(-5, 5)
plt.xlabel('s [m]')
plt.subplots_adjust(top=.95, bottom=.06, hspace=.3)

plt.savefig(f'solenoid_tracking_{"_".join(toplot)}.png', dpi=200)


plt.show()

