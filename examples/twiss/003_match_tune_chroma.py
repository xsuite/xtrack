import json
import time
import numpy as np
import xtrack as xt
import xpart as xp

with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker=line.build_tracker()

tw0 = tracker.twiss()


def error(knob_values):
    tracker.vars['kqtf.b1'] = knob_values[0]
    tracker.vars['kqtd.b1'] = knob_values[1]
    tw = tracker.twiss()
    return np.array([tw['qx'] - 62.315, tw['qy'] - 60.325])

from scipy.optimize import fsolve
#time fsolve
t1 = time.time()
fsolve(error, x0=[tracker.vars['kqtf.b1']._value, tracker.vars['kqtd.b1']._value])
t2 = time.time()
print('Time fsolve: ', t2-t1)