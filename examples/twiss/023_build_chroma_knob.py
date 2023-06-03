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
line.twiss_default['method'] = '4d'
line.twiss_default['freeze_longitudinal'] = True
line.build_tracker()

#########
# Twiss #
#########

vary=[ xt.Vary('ksf.b1', step=1e-8),  xt.Vary('ksd.b1', step=1e-8)]
line.match(
    vary=vary,
    targets = [xt.Target('dqx', 2.0, tol=0.001),
               xt.Target('dqy', 2.0, tol=0.001)])

# record values of the knobs
values_before = [line.vars[vv.name]._value for vv in vary]
knob_name = 'dqx.b1'
knob_before = 2.0
line.match(
    vary=vary,
    targets = [xt.Target('dqx', 3.0, tol=0.001),
               xt.Target('dqy', 2.0, tol=0.001)])
knob_after = 3.0
values_after = [line.vars[vv.name]._value for vv in vary]

line.vars[knob_name] = knob_after
for ii, vv in enumerate(vary):
    line.vars[vv.name] += ((values_before[ii] - values_after[ii])
                            * (line.vars[knob_name] - knob_after)
                            / (knob_before - knob_after))








