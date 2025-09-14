import xtrack as xt
import matplotlib.pyplot as plt

# Load LEP lattice
line = xt.load('../../test_data/lep/lep_sol.json')

# Set solenoids and compensation bumps off
line['on_solenoids'] = 0
line['on_spin_bumps'] = 0
line['on_coupling_corrections'] = 0

# Twiss with spin and polarization calculation
tw_no_sol = line.twiss4d(spin=True, polarization=True)

# Inspect equilibrium polarization
tw_no_sol.spin_polarization_eq # is 0.92376

# Enable solenoids and corresponding coupling corrections
line['on_solenoids'] = 1
line['on_spin_bumps'] = 0
line['on_coupling_corrections'] = 1

# Twiss with spin and polarization calculation
tw_sol = line.twiss4d(spin=True, polarization=True)

# Inspect equilibrium polarization
tw_sol.spin_polarization_eq # is 0.018617

# Enable also spin bumps
line['on_solenoids'] = 1
line['on_spin_bumps'] = 1
line['on_coupling_corrections'] = 1

# Twiss with spin and polarization calculation
tw = line.twiss4d(spin=True, polarization=True)

# Inspect equilibrium polarization
tw.spin_polarization_eq # is 0.89160

# Plot spin closed solution (n_0)
plt.figure(figsize=(6.4, 4.8*1.3))
ax1 = plt.subplot(3, 1, 1)
tw.plot('y', ax=ax1)
ax2 = plt.subplot(3, 1, 2)
tw.plot('spin_x spin_z', ax=ax2)
plt.ylabel(r'$n_{0,x}, n_{0,z}$')
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
tw.plot('spin_y', ax=ax3)
plt.ylabel(r'$n_{0,y}$')
plt.ylim(0.998, 1.001)

plt.show()