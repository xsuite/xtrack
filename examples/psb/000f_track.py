import numpy as np
import xtrack as xt

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line.build_tracker()

line.vars['on_chicane_k0'] = 1
line.vars['on_chicane_k2'] = 1
line.vars['on_chicane_beta_corr'] = 1
line.vars['on_chicane_tune_corr'] = 1

# Install monitor at foil
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=6000, num_particles=1)
line.discard_tracker()
line.insert_element(index='bi1.tstr1l1', element=monitor, name='monitor_at_foil')
line.build_tracker()

p = line.build_particles(x=0, px=0, y=0, py=0, delta=0, zeta=0)

line.enable_time_dependent_vars = True
line.dt_update_time_dependent_vars = 3e-6

print('Tracking...')
line.track(p, num_turns=6000, time=True)
print(f'Done in {line.time_last_track:.4} s')

import numpy as np
assert_allclose = np.testing.assert_allclose
assert_allclose(monitor.x[0, 0], -0.045936, rtol=0, atol=1e-5)
assert_allclose(monitor.x[0, 300], -0.04522354, rtol=0, atol=1e-5)
assert_allclose(monitor.x[0, 2500], -0.02256763, rtol=0, atol=1e-5)
assert_allclose(monitor.x[0, 4500], -0.00143883, rtol=0, atol=1e-5)
assert_allclose(monitor.x[0, 5500], 0.0, rtol=0, atol=1e-5)


import matplotlib.pyplot as plt

plt.close('all')
plt.figure(1)
plt.plot(monitor.x.T)

plt.show()