import numpy as np
import xtrack as xt

def measure_aperture(line,
            dx=1e-3, dy=1e-3, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1)):
    x_test = np.arange(x_range[0], x_range[1], dx)
    y_test = np.arange(y_range[0], y_range[1], dy)

    n_x = len(x_test)

    x_probe = np.concatenate([x_test, 0*y_test])
    y_probe = np.concatenate([0*x_test, y_test])

    p = line.build_particles(x=x_probe, y=y_probe)

    with xt.line._preserve_config(line):
        line.freeze_longitudinal()
        line.freeze_vars(['x', 'px', 'y', 'py'])
        line.config.XSUITE_RESTORE_LOSS = True

        line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
        mon = line.record_last_track

    x_h_aper = mon.x[:n_x, :]
    s_h_aper = mon.s[:n_x, :]
    state_h_aper = mon.state[:n_x, :]
    state_h_aper[:, :-1] = state_h_aper[:, 1:] # due to the way they are logged

    mean_x = 0.5*(x_h_aper[:-1, :] + x_h_aper[1:, :])
    diff_loss_h = np.diff(state_h_aper, axis=0)
    zeros = mean_x * 0
    x_aper_low_mat = np.where(diff_loss_h>0, mean_x, zeros)
    x_aper_low_discrete = x_aper_low_mat.sum(axis=0)
    x_aper_high_mat = np.where(diff_loss_h<0, mean_x, zeros)
    x_aper_high_discrete = x_aper_high_mat.sum(axis=0)

    y_v_aper = mon.y[n_x:, :]
    state_v_aper = mon.state[n_x:, :]
    state_v_aper[:, :-1] = state_v_aper[:, 1:] # due to the way they are logged

    mean_y = 0.5*(y_v_aper[:-1, :] + y_v_aper[1:, :])
    diff_loss_v = np.diff(state_v_aper, axis=0)
    zeros = mean_y * 0
    y_aper_low_mat = np.where(diff_loss_v>0, mean_y, zeros)
    y_aper_low_discrete = y_aper_low_mat.sum(axis=0)
    y_aper_high_mat = np.where(diff_loss_v<0, mean_y, zeros)
    y_aper_high_discrete = y_aper_high_mat.sum(axis=0)

    s_aper = s_h_aper[0, :]

    mask_interp_low_h = x_aper_low_discrete != 0
    x_aper_low = np.interp(s_aper,
                            s_aper[mask_interp_low_h], x_aper_low_discrete[mask_interp_low_h])
    mask_interp_high_h = x_aper_high_discrete != 0
    x_aper_high = np.interp(s_aper,
                            s_aper[mask_interp_high_h], x_aper_high_discrete[mask_interp_high_h])
    x_aper_low_discrete[~mask_interp_low_h] = np.nan
    x_aper_high_discrete[~mask_interp_high_h] = np.nan

    mask_interp_low_v = y_aper_low_discrete != 0
    y_aper_low = np.interp(s_aper,
                            s_aper[mask_interp_low_v], y_aper_low_discrete[mask_interp_low_v])
    mask_interp_high_v = y_aper_high_discrete != 0
    y_aper_high = np.interp(s_aper,
                            s_aper[mask_interp_high_v], y_aper_high_discrete[mask_interp_high_v])
    y_aper_low_discrete[~mask_interp_low_v] = np.nan
    y_aper_high_discrete[~mask_interp_high_v] = np.nan

    # I force the values, for the case in which there are multiple apertures
    # at the same location
    x_aper_low[mask_interp_low_h] = x_aper_low_discrete[mask_interp_low_h]
    x_aper_high[mask_interp_low_h] = x_aper_high_discrete[mask_interp_low_h]
    y_aper_low[mask_interp_high_h] = y_aper_low_discrete[mask_interp_high_h]
    y_aper_high[mask_interp_high_h] = y_aper_high_discrete[mask_interp_high_h]

    # Force nan at end_point
    x_aper_low_discrete[-1] = np.nan
    x_aper_high_discrete[-1] = np.nan
    y_aper_low_discrete[-1] = np.nan
    y_aper_high_discrete[-1] = np.nan

    out = xt.Table({
        'name': np.array(list(line.element_names) + ['_end_point']),
        's': s_aper,
        'x_aper_low': x_aper_low,
        'x_aper_high': x_aper_high,
        'x_aper_low_discrete': x_aper_low_discrete,
        'x_aper_high_discrete': x_aper_high_discrete,
        'y_aper_low': y_aper_low,
        'y_aper_high': y_aper_high,
        'y_aper_low_discrete': y_aper_low_discrete,
        'y_aper_high_discrete': y_aper_high_discrete,
    })

    return out