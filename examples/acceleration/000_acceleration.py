import pathlib
import json
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

############
# Settings #
############

Delta_p0c = 450e9/10*23e-6 # ramp rate 450GeV/10s

fname_line = ('../../test_data/sps_w_spacecharge/'
                  'line_no_spacecharge_and_particle.json')

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])

context = xo.ContextCpu()
buffer = context.new_buffer()

energy_increase = xt.ReferenceEnergyIncrease(_buffer=buffer,
                                             Delta_p0c=Delta_p0c)
line.append_element(energy_increase, 'energy_increase')

tracker = xt.Tracker(_buffer=buffer, line=line)

particles = xp.Particles(_context=context, p0c=26e9,
                         zeta=np.linspace(-1, 1, 40))

tracker.track(particles, num_turns=500, turn_by_turn_monitor=True)

rec = tracker.record_last_track
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
for ii in range(rec.x.shape[0]):
    mask = rec.state[ii, :]>0
    plt.plot(rec.zeta[ii, mask], rec.delta[ii, mask])

# Quick check for stable phase
from scipy.constants import c as clight
# Assume only first cavity is active
frequency = line.get_elements_of_type(xt.Cavity)[0][0].frequency
voltage = line.get_elements_of_type(xt.Cavity)[0][0].voltage

# Assuming proton and beta=1
stable_z = np.arcsin(Delta_p0c/voltage)/frequency/2/np.pi*clight

plt.axvline(x=stable_z, color='black', linestyle='--')
plt.grid(linestyle='--')
plt.xlabel('z [m]')
plt.ylabel(r'$\Delta p / p_0$')

plt.show()
