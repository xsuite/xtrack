import json
import time
from functools import partial
import numpy as np
import xtrack as xt
import xpart as xp

with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker=line.build_tracker()

tw0 = tracker.twiss()

def error(knob_values, vary, targets, tracker):
    for kk, vv in zip(vary, knob_values):
        tracker.vars[kk] = vv
    tw = tracker.twiss()
    res = []
    for tt in targets:
        if isinstance(tt[0], str):
            res.append(tw[tt[0]] - tt[1])
        else:
            res.append(tt[0](tw) - tt[1])
    return np.array(res)

def match(tracker, vary, targets):
    _err = partial(error, vary=vary, targets=targets, tracker=tracker)
    x0 = [tracker.vars[vv]._value for vv in vary]
    (res, infodict, ier, mesg) = fsolve(_err, x0=x0, full_output=True)
    for kk, vv in zip(vary, res):
        tracker.vars[kk] = vv
    fsolve_info = {
            'res': res, 'info': infodict, 'ier': ier, 'mesg': mesg}
    return fsolve_info

print('\nInitial twiss parameters')
tw_before = tracker.twiss()
print(f"Qx = {tw_before['qx']:.5f} Qy = {tw_before['qy']:.5f} "
      f"Q'x = {tw_before['dqx']:.5f} Q'y = {tw_before['dqy']:.5f}")

from scipy.optimize import fsolve
#time fsolve
t1 = time.time()
match(tracker, vary= ['kqtf.b1', 'kqtd.b1','ksf.b1', 'ksd.b1'],
    targets = [
        ('qx', 62.315),
        (lambda tw: tw['qx'] - tw['qy'], 1.99), # equivalent to ('qy', 60.325)
        ('dqx', 10.0),
        ('dqy', 12.0),])
t2 = time.time()

print('\nTime fsolve: ', t2-t1)

tw_final = tracker.twiss()
print('\nFinal twiss parameters')
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
      f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")