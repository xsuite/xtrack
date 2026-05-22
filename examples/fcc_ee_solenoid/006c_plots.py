import xtrack as xt

env = xt.load('fccee_z_lcc_solenoid.json')
line = env.fccee_p_ring

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

tw = line.twiss6d(strengths=True)

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