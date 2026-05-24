import xtrack as xt
import numpy as np

env = xt.load('fccee_z_lcc_solenoid.json')

# work on a copy of the line for slicing
line = env.fccee_p_ring.copy(shallow=True)

line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

# Slice the IP regions
line.cycle('end_ds_start_straight_ipa')
ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
tt = line.get_table()
for ip_name in ip_names:
    s_cut_right = np.arange(tt['s', ip_name] + 2.4, tt['s', ip_name] + 11, 0.2)
    line.cut_at_s(s_cut_right)
    s_cut_left = np.arange(tt['s', ip_name] - 11, tt['s', ip_name] - 2.4, 0.2)
    line.cut_at_s(s_cut_left)

# Turn off exprimental solenoids
line['on_sol_ipa'] = 0
line['on_sol_ipd'] = 0
line['on_sol_ipg'] = 0
line['on_sol_ipj'] = 0

# Turn off corrections
# (compens. solenoids, doublet tilts, orbit correction, optics correction)
line['on_sol_corr_ipa'] = 0
line['on_sol_corr_ipd'] = 0
line['on_sol_corr_ipg'] = 0
line['on_sol_corr_ipj'] = 0

tw_off = line.twiss6d(strengths=True)

# Turn on exprimental solenoids
line['on_sol_ipa'] = 1
line['on_sol_ipd'] = 1
line['on_sol_ipg'] = 1
line['on_sol_ipj'] = 1

# Turn on corrections
# (compens. solenoids, doublet tilts, orbit correction, optics correction)
line['on_sol_corr_ipa'] = 1
line['on_sol_corr_ipd'] = 1
line['on_sol_corr_ipg'] = 1
line['on_sol_corr_ipj'] = 1

tw = line.twiss6d(strengths=True, polarization_analysis=True)

two = line.twiss(betx=tw_off.betx[0], bety=tw_off.bety[0])

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()
tw_rad = line.twiss6d(strengths=True)

tw['bs'] = tw.ks * line.particle_ref.rigidity0[0]
for ip_name in tw.rows['ip.*'].name:
    tw['bs', ip_name] = np.nan # to avoid seeing zero at ips

import matplotlib.pyplot as plt
plt.close('all')

ip_plot = 'ipg'
tw_off.zero_at(ip_plot)
tw.zero_at(ip_plot)

fig1 = plt.figure(figsize=(6.4, 4.8 * 1.8))
ax1 = fig1.add_subplot(5,1,1)
plty = tw_off.plot(ax=ax1)

ax2 = fig1.add_subplot(5,1,2, sharex=ax1)
ax2.plot(tw.s, tw.bs)
ax2.set_ylabel(r'$B_s$ [T]')
ax2.grid(True)

ax3 = fig1.add_subplot(5,1,3, sharex=ax1)
ax3.plot(tw.s, tw.y * 1e3)
ax3.set_ylabel('y [mm]')
ax3.set_ylim(-0.2, 0.2)
ax3.grid(True)

ax4 = fig1.add_subplot(5,1,4, sharex=ax1)
ax4.plot(tw.s, tw.dy * 1e3)
ax4.set_ylabel(r'$D_y$ [mm]')
ax4.set_ylim(-0.2, 0.2)
ax4.grid(True)

ax5 = fig1.add_subplot(5,1,5, sharex=ax1)
ax5.plot(tw.s, tw.betx2)
ax5.plot(tw.s, tw.bety1)
ax5.set_ylabel(r'$\beta_{x2,y1}$')
ax5.grid(True)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')

fig1.subplots_adjust(hspace=.25, top=0.95, bottom=0.06, left=0.14)

ax5.set_xlim(-20, 20)

plt.show()