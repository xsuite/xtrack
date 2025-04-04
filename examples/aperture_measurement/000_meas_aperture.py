import xtrack as xt
import numpy as np

line = xt.Line.from_json('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')

tw0 = line.twiss4d()

# x_grid = np.arange(-0.1, 0.1, 1e-3)
# y_grid = np.arange(-0.05, 0.05, 1e-3)
# XX, YY = np.meshgrid(x_grid, y_grid)
# x_probe = XX.flatten()
# y_probe = YY.flatten()

x_probe = np.linspace(-0.1, 0.1, 100)
y_probe = 0

p = line.build_particles(x=x_probe, y=y_probe)

line.freeze_longitudinal()
line.freeze_vars(['x', 'px', 'y', 'py'])
line.config.XSUITE_RESTORE_LOSS = True

line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track

mask_lost = p.state < 1

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(figsize=(10, 5))
plt.plot(p.x[mask_lost], p.y[mask_lost], '.' , markersize=1)

plt.figure()
plt.pcolormesh(mon.state)

plt.show()
