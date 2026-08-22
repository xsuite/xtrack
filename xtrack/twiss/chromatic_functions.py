# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .. import linear_normal_form as lnf
from .optics_propagation import _propagate_twiss_from_init
from .twiss_init import TwissInit

if hasattr(np, 'trapezoid'): # numpy >= 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz


def _get_chromatic_functions(
        twiss_config, on_momentum_twiss_res, tw_chrom_res=None):

    if twiss_config['only_markers']:
        raise NotImplementedError('only_markers not supported anymore')

    if tw_chrom_res is None:
        tw_chrom_res = []
        delta_chrom = twiss_config['delta_chrom']
        for momentum_offset in [-delta_chrom, delta_chrom]:
            if twiss_config['periodic']:
                tw_init_chrom = _build_periodic_off_momentum_init(
                    twiss_config, on_momentum_twiss_res, momentum_offset)
            else:
                tw_init_chrom = _build_open_off_momentum_init(
                    twiss_config, momentum_offset)

            tw_chrom_res.append(
                _propagate_twiss_from_init(
                    line=twiss_config['line'],
                    init=tw_init_chrom,
                    start=twiss_config['start'],
                    end=twiss_config['end'],
                    nemitt_x=twiss_config['nemitt_x'],
                    nemitt_y=twiss_config['nemitt_y'],
                    step_W_sigma=twiss_config['step_W_sigma'],
                    delta_disp=twiss_config['delta_disp'],
                    use_full_inverse=twiss_config['use_full_inverse'],
                    hide_thin_groups=twiss_config['hide_thin_groups'],
                    only_markers=twiss_config['only_markers'],
                    continue_if_lost=False,
                    keep_tracking_data=False,
                    keep_initial_particles=False,
                    initial_particles=None,
                    ebe_monitor=None,
                )
            )

    ddelta_local = tw_chrom_res[1].delta - tw_chrom_res[0].delta

    dmux = (tw_chrom_res[1].mux - tw_chrom_res[0].mux)/ddelta_local
    dmuy = (tw_chrom_res[1].muy - tw_chrom_res[0].muy)/ddelta_local

    dbetx = (tw_chrom_res[1].betx - tw_chrom_res[0].betx)/ddelta_local
    dbety = (tw_chrom_res[1].bety - tw_chrom_res[0].bety)/ddelta_local
    dalfx = (tw_chrom_res[1].alfx - tw_chrom_res[0].alfx)/ddelta_local
    dalfy = (tw_chrom_res[1].alfy - tw_chrom_res[0].alfy)/ddelta_local
    betx = (tw_chrom_res[1].betx + tw_chrom_res[0].betx)/2
    bety = (tw_chrom_res[1].bety + tw_chrom_res[0].bety)/2
    alfx = (tw_chrom_res[1].alfx + tw_chrom_res[0].alfx)/2
    alfy = (tw_chrom_res[1].alfy + tw_chrom_res[0].alfy)/2

    # See MAD8 physics manual section 6.3
    bx_chrom = dbetx / betx
    by_chrom = dbety / bety
    ax_chrom = dalfx - dbetx * alfx / betx
    ay_chrom = dalfy - dbety * alfy / bety

    wx_chrom = np.sqrt(ax_chrom**2 + bx_chrom**2)
    wy_chrom = np.sqrt(ay_chrom**2 + by_chrom**2)

    # Could be addede if needed (note that mad-x unwraps and devide by 2pi)
    # phix_chrom = np.arctan2(ax_chrom, bx_chrom)
    # phiy_chrom = np.arctan2(ay_chrom, by_chrom)

    dqx = dmux[-1]
    dqy = dmuy[-1]

    dzeta = (tw_chrom_res[1].zeta - tw_chrom_res[0].zeta)/ddelta_local
    dzeta -= dzeta[0]
    dzeta = np.array(dzeta)

    cols_chrom = {'dmux': dmux, 'dmuy': dmuy, 'dzeta': dzeta,
                  'bx_chrom': bx_chrom, 'by_chrom': by_chrom,
                  'ax_chrom': ax_chrom, 'ay_chrom': ay_chrom,
                  'wx_chrom': wx_chrom, 'wy_chrom': wy_chrom,
                  }
    scalars_chrom = {'dqx': dqx, 'dqy': dqy}

    if on_momentum_twiss_res is not None:

        tw_plus = tw_chrom_res[1]
        tw_minus = tw_chrom_res[0]
        tw_center = on_momentum_twiss_res

        if tw_center.s[-1] == 0:
            # line has zero length, so we cannot integrate.
            # We just take the mean of the delta values
            delta_plus_mean = np.mean(tw_plus.delta)
            delta_minus_mean = np.mean(tw_minus.delta)
            delta_center_mean = np.mean(tw_center.delta)
        else:
            delta_plus_mean = trapz(tw_plus.delta, tw_plus.s) / tw_plus.s[-1]
            delta_minus_mean = trapz(tw_minus.delta, tw_minus.s) / tw_minus.s[-1]
            delta_center_mean = trapz(tw_center.delta, tw_center.s) / tw_center.s[-1]

        dqx_plus = (tw_plus.mux[-1] - tw_center.mux[-1]) / (delta_plus_mean - delta_center_mean)
        dqx_minus = (tw_center.mux[-1] - tw_minus.mux[-1]) / (delta_center_mean - delta_minus_mean)
        dqy_plus = (tw_plus.muy[-1] - tw_center.muy[-1]) / (delta_plus_mean - delta_center_mean)
        dqy_minus = (tw_center.muy[-1] - tw_minus.muy[-1]) / (delta_center_mean - delta_minus_mean)

        delta_dqxy_plus = 0.5 * (delta_plus_mean + delta_center_mean)
        delta_dqxy_minus = 0.5 * (delta_center_mean + delta_minus_mean)
        ddqx = (dqx_plus - dqx_minus) / (delta_dqxy_plus - delta_dqxy_minus)
        ddqy = (dqy_plus - dqy_minus) / (delta_dqxy_plus - delta_dqxy_minus)

        delta_dxdy_plus = 0.5 * (tw_plus.delta + tw_center.delta)
        delta_dxdy_minus = 0.5 * (tw_center.delta + tw_minus.delta)

        dx_plus = (tw_plus.x - tw_center.x) / (tw_plus.delta - tw_center.delta)
        dpx_plus = (tw_plus.px - tw_center.px) / (tw_plus.delta - tw_center.delta)
        dy_plus = (tw_plus.y - tw_center.y) / (tw_plus.delta - tw_center.delta)
        dpy_plus = (tw_plus.py - tw_center.py) / (tw_plus.delta - tw_center.delta)

        dx_minus = (tw_center.x - tw_minus.x) / (tw_center.delta - tw_minus.delta)
        dpx_minus = (tw_center.px - tw_minus.px) / (tw_center.delta - tw_minus.delta)
        dy_minus = (tw_center.y - tw_minus.y) / (tw_center.delta - tw_minus.delta)
        dpy_minus = (tw_center.py - tw_minus.py) / (tw_center.delta - tw_minus.delta)

        ddx = (dx_plus - dx_minus) / (delta_dxdy_plus - delta_dxdy_minus)
        ddpx = (dpx_plus - dpx_minus) / (delta_dxdy_plus - delta_dxdy_minus)
        ddy = (dy_plus - dy_minus) / (delta_dxdy_plus - delta_dxdy_minus)
        ddpy = (dpy_plus - dpy_minus) / (delta_dxdy_plus - delta_dxdy_minus)

        cols_chrom.update({'ddx': ddx, 'ddpx': ddpx,
                           'ddy': ddy, 'ddpy': ddpy})
        scalars_chrom.update({'ddqx': ddqx, 'ddqy': ddqy})

    return cols_chrom, scalars_chrom


def _build_periodic_off_momentum_init(
        twiss_config, on_momentum_twiss_res, momentum_offset):

    import xpart

    line = twiss_config['line']
    method = twiss_config['method']
    twiss_init = twiss_config['init'].copy()

    slip_factor = on_momentum_twiss_res.slip_factor_dzeta_ddelta
    zeta_shift = momentum_offset * slip_factor
    part_guess = xpart.build_particles(
        _context=line._context,
        x_norm=0,
        zeta=twiss_init.zeta,
        delta=twiss_init.delta + momentum_offset,
        particle_on_co=on_momentum_twiss_res.particle_on_co.copy(),
        nemitt_x=twiss_config['nemitt_x'],
        nemitt_y=twiss_config['nemitt_y'],
        W_matrix=twiss_init.W_matrix,
        include_collective=twiss_config['include_collective'])

    delta0 = twiss_config['delta0']
    if method == '4d':
        delta0 = (delta0 + momentum_offset
                  if delta0 is not None else momentum_offset)
    twiss_init.particle_on_co = line.find_closed_orbit(
        delta0=delta0,
        zeta0=twiss_config['zeta0'],
        zeta_shift=-(zeta_shift if method == '6d' else 0),
        co_guess=part_guess,
        start=twiss_config['start'],
        end=twiss_config['end'],
        num_turns=twiss_config['num_turns'],
        symmetrize=False,
        include_collective=twiss_config['include_collective'],
    )
    R_matrix = line.get_R_matrix(
        particle_on_co=twiss_init.particle_on_co.copy(),
        start=twiss_config['start'],
        end=twiss_config['end'],
        num_turns=twiss_config['num_turns'],
        steps=twiss_config['steps_R_matrix'],
        symmetrize=False,
        include_collective=twiss_config['include_collective'],
    )['R_matrix']
    W_matrix, _, _, _ = lnf.get_linear_normal_form(
        R_matrix,
        only_4d_block=(method == '4d'),
        responsiveness_tol=twiss_config['matrix_responsiveness_tol'],
        stability_tol=twiss_config['matrix_stability_tol'],
        symplectify=twiss_config['symplectify'])
    twiss_init.W_matrix = W_matrix
    return twiss_init


def _build_open_off_momentum_init(twiss_config, momentum_offset):

    line = twiss_config['line']
    init = twiss_config['init']
    twiss_init = init.copy()

    alfx = init.alfx
    betx = init.betx
    alfy = init.alfy
    bety = init.bety
    dx = init.dx
    dy = init.dy
    dpx = init.dpx
    dpy = init.dpy
    ddx = init.ddx
    ddpx = init.ddpx
    ddy = init.ddy
    ddpy = init.ddpy
    ax_chrom = init.ax_chrom
    bx_chrom = init.bx_chrom
    ay_chrom = init.ay_chrom
    by_chrom = init.by_chrom

    dbetx_dpzeta = bx_chrom * betx
    dbety_dpzeta = by_chrom * bety
    dalfx_dpzeta = ax_chrom + bx_chrom * alfx
    dalfy_dpzeta = ay_chrom + by_chrom * alfy

    twiss_init.particle_on_co.x += (
        dx * momentum_offset + 1 / 2 * ddx * momentum_offset**2)
    twiss_init.particle_on_co.px += (
        dpx * momentum_offset + 1 / 2 * ddpx * momentum_offset**2)
    twiss_init.particle_on_co.y += (
        dy * momentum_offset + 1 / 2 * ddy * momentum_offset**2)
    twiss_init.particle_on_co.py += (
        dpy * momentum_offset + 1 / 2 * ddpy * momentum_offset**2)
    twiss_init.particle_on_co.delta += momentum_offset

    auxiliary_init = TwissInit(
        alfx=alfx + dalfx_dpzeta * momentum_offset,
        betx=betx + dbetx_dpzeta * momentum_offset,
        alfy=alfy + dalfy_dpzeta * momentum_offset,
        bety=bety + dbety_dpzeta * momentum_offset,
        dx=dx + ddx * momentum_offset,
        dpx=dpx + ddpx * momentum_offset,
        dy=dy + ddy * momentum_offset,
        dpy=dpy + ddpy * momentum_offset)
    auxiliary_init._finish_initialization(
        line, element_name=init.element_name)
    twiss_init.W_matrix = auxiliary_init.W_matrix
    return twiss_init
