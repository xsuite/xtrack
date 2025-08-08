import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 500

line = xt.load('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128

energy = np.linspace(45.5e9, 45.7e9, 100)

spin_tunes = []
for ee in energy:
    line.particle_ref.p0c = ee
    spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]
    print(f'Energy: {ee/1e9:.2f} GeV, Spin tune: {spin_tune:.2f}')
    spin_tunes.append(spin_tune)

import matplotlib.pyplot as plt
plt.plot(energy/1e9, spin_tunes)
plt.xlabel('Energy [GeV]')
plt.ylabel('Spin tune')
plt.title('Spin tune vs Energy')

plt.grid()
plt.show()