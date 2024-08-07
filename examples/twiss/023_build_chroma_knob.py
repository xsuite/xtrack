import json

import numpy as np

import xtrack as xt

###################################
# Load a line and build a tracker #
####################################

with open('../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json') as f:
    dct = json.load(f)

line = xt.Line.from_dict(dct['line'])
line.particle_ref = xt.Particles.from_dict(dct['particle'])
line.twiss_default['method'] = '4d'
line.twiss_default['freeze_longitudinal'] = True
line.build_tracker()

vary=[ xt.Vary('ksf.b1', step=1e-8),  xt.Vary('ksd.b1', step=1e-8)]
line.match(
    vary=vary,
    targets = [xt.Target('dqx', 2.0, tol=1e-6),
               xt.Target('dqy', 2.0, tol=1e-6)])

tw0 = line.twiss()
line.match_knob('dqx.b1',
            knob_value_start=2.0,
            knob_value_end=3.0,
            vary=[ xt.Vary('ksf.b1', step=1e-8), xt.Vary('ksd.b1', step=1e-8)],
            targets=[
                xt.Target('dqx', 3.0, tol=1e-6),
                xt.Target('dqy', tw0, tol=1e-6)])

line.match_knob('dqy.b1',
            knob_value_start=2.0,
            knob_value_end=3.0,
            vary=[ xt.Vary('ksf.b1', step=1e-8), xt.Vary('ksd.b1', step=1e-8)],
            targets=[
                xt.Target('dqx', tw0, tol=1e-6),
                xt.Target('dqy', 3.0, tol=1e-6)])

tw = line.twiss()
assert np.isclose(tw.dqx, 2.0, atol=1e-6)
assert np.isclose(tw.dqy, 2.0, atol=1e-6)

line.vars['dqx.b1'] = 6.0
line.vars['dqy.b1'] = 7.0

tw = line.twiss()
assert np.isclose(tw.dqx, 6.0, atol=1e-4)
assert np.isclose(tw.dqy, 7.0, atol=1e-4)








