import numpy as np

import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.build_tracker()

mode = 'polar'
r_range = (0.1, 6)
theta_range = (0.2, np.pi/2-0.05)
nr=10
ntheta=10

mode = 'action'
J_x_range = (0.1**1, 6**2)
J_y_range = (0.1**2, 6**2)
nJ_x = 10
nJ_y = 10


nemitt_x = 1e-6
nemitt_y = 1e-6
num_turns = 256
n = 2**18

if mode == 'polar':
    x_norm, y_norm, r, theta = xp.generate_2D_polar_grid(
        r_range=r_range, theta_range=theta_range, nr=nr, ntheta=ntheta)
elif mode == 'action':
    Jx_grid = np.linspace(*J_x_range, nJ_x)
    Jy_grid = np.linspace(*J_y_range, nJ_y)
    JJx, JJy = np.meshgrid(Jx_grid, Jy_grid)
    Jx = JJx.flatten()
    Jy = JJy.flatten()
    x_norm = np.sqrt(Jx)
    y_norm = np.sqrt(Jy)

x_norm2d = x_norm.reshape((nr, ntheta))
y_norm2d = y_norm.reshape((nr, ntheta))

particles = line.build_particles(
    x_norm=x_norm, y_norm=y_norm, nemitt_x=nemitt_x, nemitt_y=nemitt_y)

line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)

assert np.all(particles.state == 1)
mon = line.record_last_track


fft_x = np.fft.rfft(mon.x - np.atleast_2d(np.mean(mon.x, axis=1)).T, n=n, axis=1)
fft_y = np.fft.rfft(mon.y - np.atleast_2d(np.mean(mon.y, axis=1)).T, n=n, axis=1)

freq_axis = np.fft.rfftfreq(n)

# fft_x[:, freq_axis < 0.01] = 0
# fft_y[:, freq_axis < 0.01] = 0

qx = freq_axis[np.argmax(np.abs(fft_x), axis=1)]
qy = freq_axis[np.argmax(np.abs(fft_y), axis=1)]

qx_2d = qx.reshape((nr, ntheta))
qy_2d = qy.reshape((nr, ntheta))