import xtrack as xt
import numpy as np

line = xt.Line.from_json('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')

tw0 = line.twiss4d()

x_probe = np.arange(-0.1, 0.1, 1e-3)
y_probe = np.arange(-0.05, 0.05, 1e-3)

XX, YY = np.meshgrid(x_probe, y_probe)

p = line.build_particles(x=XX.flatten(), y=YY.flatten())

line.freeze_longitudinal()
line.freeze_vars(['x', 'px', 'y', 'py'])



mask_lost = p.state < 1

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(p.x[mask_lost], p.y[mask_lost], '.' , markersize=1)