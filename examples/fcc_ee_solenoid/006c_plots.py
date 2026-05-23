import xtrack as xt
import numpy as np

env = xt.load('fccee_z_lcc_solenoid.json')

# work on a copy of the line for slicing
line = env.fccee_p_ring.copy(shallow=True)

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

import matplotlib.pyplot as plt
plt.close('all')

ip_plot = 'ipg'
tw_off.zero_at(ip_plot)
tw.zero_at(ip_plot)

fig1 = plt.figure(figsize=(6.4, 4.8 * 1.6))
ax1 = fig1.add_subplot(4,1,1)
plty = tw_off.plot(ax=ax1)

ax2 = fig1.add_subplot(4,1,2, sharex=ax1)
ax2.plot(tw.s, tw.y * 1e3)
ax2.set_ylabel('y [mm]')

ax3 = fig1.add_subplot(4,1,3, sharex=ax1)
ax3.plot(tw.s, tw.dy * 1e3)
ax3.set_ylabel('Dy [mm]')

ax3 = fig1.add_subplot(4,1,4, sharex=ax1)
ax3.plot(tw.s, tw.betx2)
ax3.plot(tw.s, tw.bety1)
ax3.set_ylabel(r'$\beta_{x2,y1}$')

plt.show()