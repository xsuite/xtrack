import json
import time
import xtrack as xt
import xpart as xp

###################################
# Load a line and build a tracker #
####################################

with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])
line.build_tracker()

#########
# Twiss #
#########

print('\nInitial twiss parameters')                                             #!skip-doc
tw_before = line.twiss()

print(f"Qx = {tw_before['qx']:.5f} Qy = {tw_before['qy']:.5f} ")
print(f"Q'x = {tw_before['dqx']:.5f} Q'y = {tw_before['dqy']:.5f}")

# Initial value of the knobs correcting tunes an chromaticities
print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")


t1 = time.time()                                                                #!skip-doc
line.match(
    #solver='fsolve',
    solver='jacobian',
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('ksf.b1', step=1e-8),
        xt.Vary('ksd.b1', step=1e-8),
    ],
    targets = [
        xt.Target('qx', 62.315, tol=1e-4),
        xt.Target('qy', 60.325, tol=1e-4),
        xt.Target('dqx', 10.0, tol=0.05),
        xt.Target('dqy', 12.0, tol=0.05)]
    )
t2 = time.time()                                                                #!skip-doc
print('\nTime match: ', t2 - t1)

tw_after = line.twiss()

print('\nFinal twiss parameters')                                               #!skip-doc
print(f"Qx = {tw_after['qx']:.5f} Qy = {tw_after['qy']:.5f} ")
print(f"Q'x = {tw_after['dqx']:.5f} Q'y = {tw_after['dqy']:.5f}")

# Initial value of the knobs correcting tunes an chromaticities
print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")