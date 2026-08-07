# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xobjects as xo
from scipy.constants import c as clight

if hasattr(np, 'trapezoid'): # numpy >= 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz


def _add_ring_quantities(line, twiss_res, method):

    s_vect = twiss_res['s']
    line_length = line.tracker._tracker_data_base.line_length
    part_on_co = twiss_res['particle_on_co']
    W_matrix = twiss_res['W_matrix']

    beta0 = part_on_co._xobject.beta0[0]
    gamma0 = part_on_co._xobject.gamma0[0]
    t_rev0 = line_length/clight/beta0
    bets0 = W_matrix[0, 4, 4]**2 + W_matrix[0, 4, 5]**2

    # compute slip factor

    if method == '6d':
        RR = twiss_res['R_matrix']
        dz_test = 1e-3 # All linear, so the value does not matter
        xx = np.linalg.solve(RR - np.eye(6), np.array([0,0,0,0,dz_test,0]))
        delta_test = xx[5]
    elif method == '4d':
        RR = twiss_res['R_matrix'].copy()
        solve_mat = RR - np.eye(6)
        solve_mat[4, :] = np.array([0,0,0,0,1,0]) # dummy
        solve_mat[5, :] = np.array([0,0,0,0,0,1]) # delta
        delta_test = 1e-3 # All linear, so the value does not matter
        xx = np.linalg.solve(solve_mat, np.array([0,0,0,0,0,delta_test]))
        # measure slippage on original matrix
        xx_out = twiss_res['R_matrix'] @ xx
        dz_test = xx_out[4] - xx[4]

    slip_factor_dzeta_ddelta = dz_test / delta_test

    if line_length > 0:
        slip_factor = -slip_factor_dzeta_ddelta / line_length
        momentum_compaction_factor = (slip_factor + 1/gamma0**2)
    else:
        slip_factor = np.nan
        momentum_compaction_factor = np.nan

    if slip_factor_dzeta_ddelta > 0: # below transition
        bets0 = -bets0

    twiss_res._data.update({
        'bets0': bets0,
        'line_length': line_length,
        'circumference': line_length,  # deprecated
        'T_rev0': t_rev0, # deprecated
        't_rev0': t_rev0,
        'particle_on_co':part_on_co.copy(_context=xo.context_default),
        'gamma0': gamma0,
        'beta0': beta0,
        'p0c': part_on_co._xobject.p0c[0],
        'slip_factor': slip_factor,
        'momentum_compaction_factor': momentum_compaction_factor,
        'slip_factor_dz_ddelta': slip_factor_dzeta_ddelta, # deprecated
        'slip_factor_dzeta_ddelta': slip_factor_dzeta_ddelta,
    })

    if hasattr(part_on_co, '_fsolve_info'):
        twiss_res.particle_on_co._fsolve_info = part_on_co._fsolve_info
    else:
        twiss_res.particle_on_co._fsolve_info = None

    if 'mux' in twiss_res._data: # Lattice functions are available
        mux = twiss_res['mux']
        muy = twiss_res['muy']

        # Coupling
        # from Y. Luo et al., "Possible phase loop for the global betatron decoupling",
        #  C-A/AP/#174, https://www.agsrhichome.bnl.gov//AP/ap_notes/ap_note_174.pdf
        w11 = W_matrix[:, 0, 0]
        w13 = W_matrix[:, 0, 2]
        w14 = W_matrix[:, 0, 3]
        w31 = W_matrix[:, 2, 0]
        w32 = W_matrix[:, 2, 1]
        w33 = W_matrix[:, 2, 2]

        c_r1 = np.sqrt(w31**2 + w32**2) / w11
        c_r2 = np.sqrt(w13**2 + w14**2) / w33
        c_phi1 = np.arctan2(w32, w31)
        c_phi2 = np.arctan2(w14, w13)

        # Coupling (https://arxiv.org/pdf/2005.02753.pdf)
        # R. Jones, Measuring Tune, Chromaticity and Coupling,
        # Proceedings of the 2018 CERN-Accelerator-School
        cmin_arr = (2 * np.sqrt(c_r1*c_r2) *
                    np.abs(np.mod(mux[-1], 1) - np.mod(muy[-1], 1))
                    /(1 + c_r1 * c_r2))
        if line_length > 0:
            c_minus = trapz(cmin_arr, s_vect)/(line_length)
        else:
            c_minus = np.mean(cmin_arr)

        c_minus_cplx = c_minus * np.exp(1j * c_phi1)
        c_minus_re = np.real(c_minus_cplx)
        c_minus_im = np.imag(c_minus_cplx)
        c_minus_local = cmin_arr * np.exp(1j * c_phi1)

        qs = np.abs(twiss_res['muzeta'][-1])

        # Scalars
        twiss_res._data.update({
            'qx': mux[-1], 'qy': muy[-1], 'qs': qs,
            'c_minus': c_minus,
            'c_minus_re_0': c_minus_re[0], 'c_minus_im_0': c_minus_im[0],
            'c_minus_local': c_minus_local,
        })

        # Coupling columns
        twiss_res['c_minus_re'] = c_minus_re
        twiss_res['c_minus_im'] = c_minus_im
        twiss_res['c_r1'] = c_r1
        twiss_res['c_r2'] = c_r2
        twiss_res['c_phi1'] = c_phi1
        twiss_res['c_phi2'] = c_phi2
