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

print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

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
print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

t1 = time.time()
tracker.match(vary=['kqtf.b1', 'kqtd.b1','ksf.b1', 'ksd.b1'],
    targets = [
        ('qx', 62.27),
        ('qy', 60.28),
        ('dqx', -5.0),
        ('dqy', -7.0),])
t2 = time.time()
print('\nTime fsolve: ', t2-t1)

tw_final = tracker.twiss()
print('\nFinal twiss parameters')
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
      f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")
print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

# 4d matching
for ee in line.elements:
    if isinstance(ee, xt.Cavity):
        ee.voltage = 0.0

t1 = time.time()
tracker.match(method='4d', # <-- 4d matching
    vary=['kqtf.b1', 'kqtd.b1','ksf.b1', 'ksd.b1'],
    targets = [
        ('qx', 62.29),
        ('qy', 60.31),
        ('dqx', 6.0),
        ('dqy', 4.0),])
t2 = time.time()
print('\nTime fsolve: ', t2-t1)

tw_final = tracker.twiss(method='4d')
print('\nFinal twiss parameters')
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
      f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")
print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
print(f"ksd.b1 = {line.vars['ksd.b1']._value}")