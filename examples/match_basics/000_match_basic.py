import json
import time
import xtrack as xt
import xpart as xp

# Load a line and build a tracker
with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)
line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])
line.build_tracker()

# Tunes, chromaticities before matching
tw_before = line.twiss()
print('\nInitial twiss parameters')                                             #!skip-doc
print(f"Qx = {tw_before['qx']:.5f} Qy = {tw_before['qy']:.5f} "
      f"Q'x = {tw_before['dqx']:.5f} Q'y = {tw_before['dqy']:.5f}")

# Match tunes and chromaticities to assigned values
opt = line.match(
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8),
        xt.Vary('ksf.b1', step=1e-8),
        xt.Vary('ksd.b1', step=1e-8),
    ],
    targets = [
        xt.Target('qx', 62.315, tol=1e-4),
        xt.Target('qy', 60.325, tol=1e-4),
        xt.Target('dqx', 10.0, tol=0.01),
        xt.Target('dqy', 12.0, tol=0.01)])

# Tunes, chromaticities after matching
tw_final = line.twiss()
print('\nFinal twiss parameters')                                               #!skip-doc
print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
      f"Q'x = {tw_final['dqx']:.2f} Q'y = {tw_final['dqy']:.2f}")
