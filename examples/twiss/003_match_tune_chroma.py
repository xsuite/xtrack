import json
import time
import xtrack as xt

###################################
# Load a line and build a tracker #
####################################

with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xt.Particles.from_dict(dct['particle'])
line.build_tracker()

#########
# Twiss #
#########

print('\nInitial twiss parameters')                                             #!skip-doc
tw_before = line.twiss()

####################################################
# Tunes, chromaticities, and knobs before matching #
####################################################

# Initial tune and chromaticity values
print(f"Qx = {tw_before['qx']:.5f} Qy = {tw_before['qy']:.5f} "
      f"Q'x = {tw_before['dqx']:.5f} Q'y = {tw_before['dqy']:.5f}")

# Initial value of the knobs correcting tunes an chromaticities
print(f"kqtf.b1 = {line['kqtf.b1']}")
print(f"kqtd.b1 = {line['kqtd.b1']}")
print(f"ksf.b1 = {line['ksf.b1']}")
print(f"ksd.b1 = {line['ksd.b1']}")

#####################################################
# Match tunes and chromaticities to assigned values #
#####################################################

t1 = time.time()                                                                #!skip-doc
line.match(
    vary=[
        xt.Vary('kqtf.b1', step=1e-6),
        xt.Vary('kqtd.b1', step=1e-6),
        xt.Vary('ksf.b1', step=1e-5),
        xt.Vary('ksd.b1', step=1e-5),
    ],
    targets = [
        xt.Target('qx', 62.315, tol=1e-4),
        xt.Target('qy', 60.325, tol=1e-4),
        xt.Target('dqx', 10.0, tol=0.05),
        xt.Target('dqy', 12.0, tol=0.05)])
t2 = time.time()                                                                #!skip-doc
print('\nTime match: ', t2-t1)                                                 #!skip-doc

###################################################
# Tunes, chromaticities, and knobs after matching #
###################################################

tw_final = line.twiss()
print('\nFinal twiss parameters')                                               #!skip-doc
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
      f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")
print(f"kqtf.b1 = {line['kqtf.b1']}")
print(f"kqtd.b1 = {line['kqtd.b1']}")
print(f"ksf.b1 = {line['ksf.b1']}")
print(f"ksd.b1 = {line['ksd.b1']}")

#####################################
# Match with specific twiss options #
#####################################

# Any argument accepted by the twiss method can be passed to the match method
# For example, to match the tunes and chromaticities using the '4d' method:

t1 = time.time()                                                                #!skip-doc
line.match(method='4d', # <-- 4d matching
    vary=[
        xt.Vary('kqtf.b1', step=1e-6),
        xt.Vary('kqtd.b1', step=1e-6),
        xt.Vary('ksf.b1', step=1e-5),
        xt.Vary('ksd.b1', step=1e-5),
    ],
    targets = [
        xt.Target('qx', 62.29, tol=1e-4),
        xt.Target('qy', 60.31, tol=1e-4),
        xt.Target('dqx', 6.0, tol=0.05),
        xt.Target('dqy', 4.0, tol=0.05)],
    )
t2 = time.time()                                                                #!skip-doc
print('\nTime 4d match: ', t2-t1)                                               #!skip-doc

tw_final = line.twiss(method='4d')                                              #!skip-doc
print('\nFinal twiss parameters')                                               #!skip-doc
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "                   #!skip-doc
      f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")               #!skip-doc
print(f"kqtf.b1 = {line['kqtf.b1']}")                               #!skip-doc
print(f"kqtd.b1 = {line['kqtd.b1']}")                               #!skip-doc
print(f"ksf.b1 = {line['ksf.b1']}")                                 #!skip-doc
print(f"ksd.b1 = {line['ksd.b1']}")                                 #!skip-doc

##############################################
# Match custom function of the twiss results #
##############################################

# The match method can also be used to match any user-defined function of
# the twiss results. For example, to match the difference between the tunes,
# instead of the vertical tune:

t1 = time.time()                                                                #!skip-doc
line.match(
    vary=[
        xt.Vary('kqtf.b1', step=1e-6),
        xt.Vary('kqtd.b1', step=1e-6),
        xt.Vary('ksf.b1', step=1e-5),
        xt.Vary('ksd.b1', step=1e-5),
    ],
    targets = [
        xt.Target('qx', 62.27, tol=1e-4),
        xt.Target(lambda tw: tw['qx'] - tw['qy'], 1.98, tol=1e-4), # equivalent to ('qy', 60.325)
        xt.Target('dqx', 6.0, tol=0.05),
        xt.Target('dqy', 4.0, tol=0.05),])
t2 = time.time()                                                                #!skip-doc
print('\nTime match with function: ', t2-t1)                                    #!skip-doc

print('\nFinal twiss parameters')                                               #!skip-doc
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "                   #!skip-doc
      f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")               #!skip-doc
print(f"kqtf.b1 = {line['kqtf.b1']}")                               #!skip-doc
print(f"kqtd.b1 = {line['kqtd.b1']}")                               #!skip-doc
print(f"ksf.b1 = {line['ksf.b1']}")                                 #!skip-doc
print(f"ksd.b1 = {line['ksd.b1']}")                                 #!skip-doc

#!end-doc-part

