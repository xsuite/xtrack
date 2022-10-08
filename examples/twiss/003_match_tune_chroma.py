import json
import time
import xtrack as xt
import xpart as xp

with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker=line.build_tracker()

print('\nInitial twiss parameters')
tw_before = tracker.twiss()
print(f"Qx = {tw_before['qx']:.5f} Qy = {tw_before['qy']:.5f} "
      f"Q'x = {tw_before['dqx']:.5f} Q'y = {tw_before['dqy']:.5f}")

t1 = time.time()
tracker.match(vary=['kqtf.b1', 'kqtd.b1','ksf.b1', 'ksd.b1'],
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