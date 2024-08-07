import time
import numpy as np

import xtrack as xt
from xtrack.slicing import Teapot, Strategy

line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.twiss_default['method'] = '4d'
line.build_tracker()

line_thick = line.copy()
line_thick.build_tracker()

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default
    Strategy(slicing=Teapot(4), element_type=xt.Bend),
    Strategy(slicing=Teapot(20), element_type=xt.Quadrupole),
    Strategy(slicing=Teapot(2), name=r'^mb\..*'),
    Strategy(slicing=Teapot(5), name=r'^mq\..*'),
    Strategy(slicing=Teapot(2), name=r'^mqt.*'),
    Strategy(slicing=Teapot(60), name=r'^mqx.*'),
]

line.discard_tracker()
t1 = time.time()
line.slice_thick_elements(slicing_strategies=slicing_strategies)
t2 = time.time()
print('\nTime slice thick: ', t2-t1)
line.build_tracker()

tw = line.twiss()
tw_thick = line_thick.twiss()

beta_beat_x_at_ips = [tw['betx', f'ip{nn}'] / tw_thick['betx', f'ip{nn}'] - 1
                        for nn in range(1, 9)]
beta_beat_y_at_ips = [tw['bety', f'ip{nn}'] / tw_thick['bety', f'ip{nn}'] - 1
                        for nn in range(1, 9)]

assert np.allclose(beta_beat_x_at_ips, 0, atol=3e-3)
assert np.allclose(beta_beat_y_at_ips, 0, atol=3e-3)

# Checks on orbit knobs
assert np.isclose(tw_thick['px', 'ip1'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw_thick['py', 'ip1'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw_thick['px', 'ip5'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw_thick['py', 'ip5'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw['px', 'ip1'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw['py', 'ip1'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw['px', 'ip5'], 0, rtol=0, atol=1e-7)
assert np.isclose(tw['py', 'ip5'], 0, rtol=0, atol=1e-7)

line.vars['on_x1'] = 50
line.vars['on_x5'] = 60
line_thick.vars['on_x1'] = 50
line_thick.vars['on_x5'] = 60

tw = line.twiss()
tw_thick = line_thick.twiss()

assert np.isclose(tw_thick['px', 'ip1'], 50e-6, rtol=0, atol=5e-7)
assert np.isclose(tw_thick['py', 'ip1'], 0, rtol=0, atol=5e-7)
assert np.isclose(tw_thick['px', 'ip5'], 0, rtol=0, atol=5e-7)
assert np.isclose(tw_thick['py', 'ip5'], 60e-6, rtol=0, atol=5e-7)
assert np.isclose(tw['px', 'ip1'], 50e-6, rtol=0, atol=5e-7)
assert np.isclose(tw['py', 'ip1'], 0, rtol=0, atol=5e-7)
assert np.isclose(tw['px', 'ip5'], 0, rtol=0, atol=5e-7)
assert np.isclose(tw['py', 'ip5'], 60e-6, rtol=0, atol=5e-7)

t1 = time.time()
opt_thick = line_thick.match(
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('ksf.b1', step=1e-8),
        xt.Vary('ksd.b1', step=1e-8),
    ],
    targets = [
        xt.Target('qx', 62.27, tol=1e-4),
        xt.Target('qy', 60.29, tol=1e-4),
        xt.Target('dqx', 10.0, tol=0.05),
        xt.Target('dqy', 12.0, tol=0.05)])
t2 = time.time()
print('\nTime match thick: ', t2-t1)

opt_thin = line.match(
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('ksf.b1', step=1e-8),
        xt.Vary('ksd.b1', step=1e-8),
    ],
    targets = [
        xt.Target('qx', 62.27, tol=1e-4),
        xt.Target('qy', 60.29, tol=1e-4),
        xt.Target('dqx', 10.0, tol=0.05),
        xt.Target('dqy', 12.0, tol=0.05)])
t2 = time.time()
print('\nTime match thin: ', t2-t1)

line.to_json('lhc_thin.json')

tw = line.twiss()
tw_thick = line_thick.twiss()

assert np.isclose(tw_thick.qx, 62.27, rtol=0, atol=1e-4)
assert np.isclose(tw_thick.qy, 60.29, rtol=0, atol=1e-4)
assert np.isclose(tw_thick.dqx, 10.0, rtol=0, atol=0.05)
assert np.isclose(tw_thick.dqy, 12.0, rtol=0, atol=0.05)
assert np.isclose(tw.qx, 62.27, rtol=0, atol=1e-4)
assert np.isclose(tw.qy, 60.29, rtol=0, atol=1e-4)
assert np.isclose(tw.dqx, 10.0, rtol=0, atol=0.05)
assert np.isclose(tw.dqy, 12.0, rtol=0, atol=0.05)

assert np.isclose(line.vars['kqtf.b1']._value, line_thick.vars['kqtf.b1']._value,
                    rtol=0.03, atol=0)
assert np.isclose(line.vars['kqtd.b1']._value, line_thick.vars['kqtd.b1']._value,
                    rtol=0.03, atol=0)
assert np.isclose(line.vars['ksf.b1']._value, line_thick.vars['ksf.b1']._value,
                    rtol=0.01, atol=0)
assert np.isclose(line.vars['ksd.b1']._value, line_thick.vars['ksd.b1']._value,
                    rtol=0.01, atol=0)
