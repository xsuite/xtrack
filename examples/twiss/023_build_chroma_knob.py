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
knob_name = 'dqx.b1'
knob_value_start = 2.0

# TODO: Remember to handle the limits!

vary_aux = []
for vv in vary:
    line.vars[vv.name + '_from_' + knob_name] = 0
    line.vars[vv.name] += line.vars[vv.name + '_from_' + knob_name]
    vary_aux.append(xt.Vary(vv.name + '_from_' + knob_name, step=vv.step))
line.match(
    vary=vary_aux,
    targets = [xt.Target('dqx', 3.0, tol=0.001),
               xt.Target('dqy', 'preserve', tol=0.001)])
knob_value_end = 3.0
line.vars[knob_name] = knob_value_end

for ii, vv in enumerate(vary_aux):
    line.vars[vv.name] = (line.vars[vv.name]._value
                          * (line.vars[knob_name] - knob_value_start)
                          / (knob_value_end - knob_value_start))

line.vars[knob_name] = knob_value_start









