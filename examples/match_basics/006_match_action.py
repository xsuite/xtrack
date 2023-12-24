import numpy as np
import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

line.vars['kof.a23b1'] = 1e3
line.vars['kof.a34b1'] = 1e3
line.vars['kof.a67b1'] = 1e3
line.vars['kof.a78b1'] = 1e3

line.vars['kod.a23b1'] = 1e3
line.vars['kod.a34b1'] = 1e3
line.vars['kod.a67b1'] = 1e3
line.vars['kod.a78b1'] = 1e3

a1_sigmas = .1
a2_sigmas = .2
num_turns = 256

Jx_1 = a1_sigmas**2 * nemitt_x / 2
Jx_2 = a2_sigmas**2 * nemitt_x / 2
Jy_1 = a1_sigmas**2 * nemitt_y / 2
Jy_2 = a2_sigmas**2 * nemitt_y / 2

particles = line.build_particles(
                    method='4d',
                    zeta=0, delta=0,
                    x_norm=[a1_sigmas, a2_sigmas, a1_sigmas],
                    y_norm=[a1_sigmas, a1_sigmas, a2_sigmas],
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y)

line.track(particles, num_turns=num_turns, time=True, turn_by_turn_monitor=True)

assert np.all(particles.state > 0)
# self.x_norm_2d = np.sqrt(2 * self.Jx_2d / nemitt_x)
# self.y_norm_2d = np.sqrt(2 * self.Jy_2d / nemitt_y)

n_fft = 2**18
freq_axis = np.fft.rfftfreq(n_fft)

mon = line.record_last_track
fft_x = np.fft.rfft(mon.x, n=n_fft)
fft_y = np.fft.rfft(mon.y, n=n_fft)

qx = freq_axis[np.argmax(np.abs(fft_x), axis=1)]
qy = freq_axis[np.argmax(np.abs(fft_y), axis=1)]

Dq = np.array([[qx[1] - qx[0], qx[2] - qx[0]],
               [qy[1] - qy[0], qy[2] - qy[0]]])

DJ = np.array([[Jx_2 - Jx_1 , 0            ],
                [0          , Jy_2 - Jy_1 ]])

DD = Dq @ np.linalg.inv(DJ)

# alpha_xx = (qx[1] - qx[0]) / (Jx_2 - Jx_1)
# alpha_yy = (qy[3] - qy[2]) / (Jy_2 - Jy_1)
# alpha_xy = (qx[3] - qx[2]) / (Jy_2 - Jy_1)
# alpha_yx = (qy[1] - qy[0]) / (Jx_2 - Jx_1)


