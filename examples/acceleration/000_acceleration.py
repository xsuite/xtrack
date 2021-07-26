import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xpart as xp
import xtrack as xt

############
# Settings #
############

fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_no_spacecharge_and_particle.json')

with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xl.Line.from_dict(input_data['line'])

context = xo.ContextCpu()
buffer = context.new_buffer()

energy_increase = xt.ReferenceEnergyIncrease(_buffer=buffer,
                                             Delta_p0c=450e9/10*23e-6)
sequence.append_element(energy_increase, 'energy_increase')

tracker = xt.Tracker(_buffer=buffer, sequence=sequence)

particles = xt.Particles(_context=context, p0c=26e9,
                         zeta=np.linspace(-1, 1, 40))

tracker.track(particles, num_turns=500, turn_by_turn_monitor=True)

rec = tracker.record_last_track
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
for ii in range(rec.x.shape[0]):
    mask = rec.state[ii, :]>0
    plt.plot(rec.zeta[ii, mask], rec.delta[ii, mask])

plt.show()
