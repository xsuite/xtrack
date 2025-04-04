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

diff_loss = np.diff(mon.state, axis=0)
mean_x = 0.5*(mon.x[:-1, :] + mon.x[1:, :])
zeros = mean_x == 0
x_aper_low_mat = np.where(diff_loss>0, mean_x, zeros)
x_aper_low = x_aper_low_mat.sum(axis=0)
x_aper_high_mat = np.where(diff_loss<0, mean_x, zeros)
x_aper_high = x_aper_high_mat.sum(axis=0)

s_aper = mon.s[0, :]

mask_interp_low = x_aper_low != 0
x_aper_low_interp = np.interp(s_aper,
                        s_aper[mask_interp_low], x_aper_low[mask_interp_low])
mask_interp_high = x_aper_high != 0
x_aper_high_interp = np.interp(s_aper,
                        s_aper[mask_interp_high], x_aper_high[mask_interp_high])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.plot(s_aper, x_aper_low_interp)
plt.plot(s_aper, x_aper_high_interp)
plt.show()

