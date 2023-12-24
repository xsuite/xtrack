import numpy as np
import xtrack as xt
import NAFFlib as pnf

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

k_oct_arr = np.arange(-250, 250, 1e1)
alpha_xx_arr = np.empty_like(k_oct_arr)
alpha_yy_arr = np.empty_like(k_oct_arr)
alpha_xy_arr = np.empty_like(k_oct_arr)
alpha_yx_arr = np.empty_like(k_oct_arr)

for k_oct in k_oct_arr:

    print(f'k_oct = {k_oct}', end='\r', flush=True)

    line.vars['kof.a23b1'] = k_oct
    line.vars['kof.a34b1'] = k_oct
    line.vars['kof.a67b1'] = k_oct
    line.vars['kof.a78b1'] = k_oct

    line.vars['kod.a23b1'] = k_oct
    line.vars['kod.a34b1'] = k_oct
    line.vars['kod.a67b1'] = k_oct
    line.vars['kod.a78b1'] = k_oct

    a0_sigmas = .01
    a1_sigmas = 0.1
    a2_sigmas = 0.15
    num_turns = 256

    Jx_0 = a0_sigmas**2 * nemitt_x / 2
    Jx_1 = a1_sigmas**2 * nemitt_x / 2
    Jx_2 = a2_sigmas**2 * nemitt_x / 2
    Jy_0 = a0_sigmas**2 * nemitt_y / 2
    Jy_1 = a1_sigmas**2 * nemitt_y / 2
    Jy_2 = a2_sigmas**2 * nemitt_y / 2

    particles = line.build_particles(
                        method='4d',
                        zeta=0, delta=0,
                        x_norm=[a1_sigmas, a2_sigmas, a0_sigmas, a0_sigmas],
                        y_norm=[a0_sigmas, a0_sigmas, a1_sigmas, a2_sigmas],
                        nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    line.track(particles, num_turns=num_turns, time=True, turn_by_turn_monitor=True)

    assert np.all(particles.state > 0)

    n_fft = 2**20
    freq_axis = np.fft.rfftfreq(n_fft)

    mon = line.record_last_track
    fft_x = np.fft.rfft(mon.x, n=n_fft)
    fft_y = np.fft.rfft(mon.y, n=n_fft)

    qx = freq_axis[np.argmax(np.abs(fft_x), axis=1)]
    qy = freq_axis[np.argmax(np.abs(fft_y), axis=1)]

    for ii in range(len(qx)):
        qx[ii] = pnf.get_tune(mon.x[ii, :])
        qy[ii] = pnf.get_tune(mon.y[ii, :])

    alpha_xx = (qx[1] - qx[0]) / (Jx_2 - Jx_1)
    alpha_yy = (qy[3] - qy[2]) / (Jy_2 - Jy_1)
    alpha_xy = (qx[3] - qx[2]) / (Jy_2 - Jy_1)
    alpha_yx = (qy[1] - qy[0]) / (Jx_2 - Jx_1)

    alpha_xx_arr[k_oct_arr==k_oct] = alpha_xx
    alpha_yy_arr[k_oct_arr==k_oct] = alpha_yy
    alpha_xy_arr[k_oct_arr==k_oct] = alpha_xy
    alpha_yx_arr[k_oct_arr==k_oct] = alpha_yx

