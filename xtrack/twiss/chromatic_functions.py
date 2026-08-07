# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .. import linear_normal_form as lnf
from .open_twiss import _twiss_open
from .twiss_init import TwissInit

if hasattr(np, 'trapezoid'): # numpy >= 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz


def _get_chromatic_functions(line, init, delta_chrom,
                    delta0, zeta0,
                    steps_R_matrix,
                    matrix_responsiveness_tol, matrix_stability_tol, symplectify,
                    method='6d', use_full_inverse=False,
                    nemitt_x=None, nemitt_y=None,
                    step_W_sigma=1e-3, delta_disp=1e-3, zeta_disp=1e-3,
                    on_momentum_twiss_res=None,
                    start=None, end=None, num_turns=None,
                    hide_thin_groups=False,
                    only_markers=False,
                    periodic=False,
                    periodic_mode=None,
                    include_collective=False,
                    tw_chrom_res=None
                    ):

    if only_markers:
        raise NotImplementedError('only_markers not supported anymore')

    if tw_chrom_res is None:
        tw_chrom_res = []
        for dd in [-delta_chrom, delta_chrom]:
            tw_init_chrom = init.copy()

            if periodic:
                slip_factor_dzeta_ddelta = on_momentum_twiss_res.slip_factor_dzeta_ddelta
                dzeta = dd * slip_factor_dzeta_ddelta
                import xpart
                part_guess = xpart.build_particles(
                    _context=line._context,
                    x_norm=0,
                    zeta=tw_init_chrom.zeta,
                    delta=tw_init_chrom.delta + dd,
                    particle_on_co=on_momentum_twiss_res.particle_on_co.copy(),
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                    W_matrix=tw_init_chrom.W_matrix,
                    include_collective=include_collective)

                dd0=delta0
                if method == '4d':
                    dd0 = delta0 + dd if delta0 is not None else dd
                part_chrom = line.find_closed_orbit(
                    delta0=dd0,
                    zeta0=zeta0,
                    zeta_shift=-(dzeta if method == '6d' else 0),
                    co_guess=part_guess,
                    start=start, end=end, num_turns=num_turns,
                    symmetrize=False,
                    include_collective=include_collective,
                    )
                tw_init_chrom.particle_on_co = part_chrom
                RR_chrom = line.get_R_matrix(
                                            particle_on_co=tw_init_chrom.particle_on_co.copy(),
                                            start=start, end=end, num_turns=num_turns,
                                            steps=steps_R_matrix,
                                            symmetrize=False,
                                            include_collective=include_collective,
                                            )['R_matrix']
                (WW_chrom, _, _, _) = lnf.get_linear_normal_form(RR_chrom,
                                        only_4d_block=(method == '4d'),
                                        responsiveness_tol=matrix_responsiveness_tol,
                                        stability_tol=matrix_stability_tol,
                                        symplectify=symplectify)
                tw_init_chrom.W_matrix = WW_chrom
            else:
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

                tw_init_chrom.particle_on_co.x += dx * dd + 1/2 * ddx * dd**2
                tw_init_chrom.particle_on_co.px += dpx * dd + 1/2 * ddpx * dd**2
                tw_init_chrom.particle_on_co.y += dy * dd + 1/2 * ddy * dd**2
                tw_init_chrom.particle_on_co.py += dpy * dd + 1/2 * ddpy * dd**2
                tw_init_chrom.particle_on_co.delta += dd

                twinit_aux = TwissInit(
                    alfx=alfx + dalfx_dpzeta * dd,
                    betx=betx + dbetx_dpzeta * dd,
                    alfy=alfy + dalfy_dpzeta * dd,
                    bety=bety + dbety_dpzeta * dd,
                    dx=dx + ddx * dd,
                    dpx=dpx + ddpx * dd,
                    dy=dy + ddy * dd,
                    dpy=dpy + ddpy * dd)
                twinit_aux._complete(line, element_name=init.element_name)
                tw_init_chrom.W_matrix = twinit_aux.W_matrix

            tw_chrom_res.append(
                _twiss_open(
                    line=line,
                    init=tw_init_chrom,
                    start=start, end=end,
                    nemitt_x=nemitt_x,
                    nemitt_y=nemitt_y,
                    step_W_sigma=step_W_sigma,
                    delta_disp=delta_disp,
                    use_full_inverse=use_full_inverse,
                    hide_thin_groups=hide_thin_groups,
                    only_markers=only_markers,
                    _continue_if_lost=False,
                    _keep_tracking_data=False,
                    _keep_initial_particles=False,
                    _initial_particles=None,
                    _ebe_monitor=None,
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



        # mux = on_momentum_twiss_res.mux
        # muy = on_momentum_twiss_res.muy
        # x = on_momentum_twiss_res.x
        # px = on_momentum_twiss_res.px
        # y = on_momentum_twiss_res.y
        # py = on_momentum_twiss_res.py
        # ddqx = (tw_chrom_res[1].mux[-1] - 2 * mux[-1] + tw_chrom_res[0].mux[-1]
        #         ) / delta_chrom**2
        # ddqy = (tw_chrom_res[1].muy[-1] - 2 * muy[-1] + tw_chrom_res[0].muy[-1]
        #         ) / delta_chrom**2
        # ddx = (tw_chrom_res[1].x - 2 * x + tw_chrom_res[0].x) / delta_chrom**2
        # ddpx = (tw_chrom_res[1].px - 2 * px + tw_chrom_res[0].px) / delta_chrom**2
        # ddy = (tw_chrom_res[1].y - 2 * y + tw_chrom_res[0].y) / delta_chrom**2
        # ddpy = (tw_chrom_res[1].py - 2 * py + tw_chrom_res[0].py) / delta_chrom**2

        cols_chrom.update({'ddx': ddx, 'ddpx': ddpx,
                           'ddy': ddy, 'ddpy': ddpy})
        scalars_chrom.update({'ddqx': ddqx, 'ddqy': ddqy})

    return cols_chrom, scalars_chrom
