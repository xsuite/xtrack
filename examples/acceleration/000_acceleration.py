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

tracker = xt.Tracker(_context=context, sequence=sequence)

particles = xt.Particles(_context=context, p0c=26e9,
                         delta=np.linspace(0, 0.4e-2, 20))

tracker.track(particles, num_turns=500, turn_by_turn_monitor=True)

rec = tracker.record_last_track
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
for ii in range(rec.x.shape[0]):
    plt.plot(rec.zeta[ii, :], rec.delta[ii, :])

plt.show()
