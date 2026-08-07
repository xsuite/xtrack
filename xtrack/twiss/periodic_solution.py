# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .. import linear_normal_form as lnf
from .element_indexing import _str_to_index
from .twiss_init import TwissInit


def _find_periodic_solution(line, particle_on_co, particle_ref, method,
                            co_search_settings, continue_on_closed_orbit_error,
                            delta0, zeta0,
                            zeta_shift,
                            steps_R_matrix, W_matrix,
                            R_matrix, co_guess,
                            delta_disp, symplectify,
                            matrix_responsiveness_tol,
                            matrix_stability_tol,
                            nemitt_x, nemitt_y, step_W_sigma,
                            start=None, end=None,
                            num_turns=1,
                            co_search_at=None,
                            search_for_t_rev=False,
                            spin=None,
                            num_turns_search_t_rev=1,
                            compute_R_element_by_element=False,
                            only_markers=False,
                            only_orbit=False,
                            periodic_mode='periodic',
                            include_collective=False,
                            factor_adapt_steps=0.3
                            ):

    eigenvalues = None
    Rot = None
    RR_ebe = None

    assert periodic_mode in ['periodic', 'periodic_symmetric']

    if periodic_mode == 'periodic_symmetric':
        raise ValueError('``periodic_symmetric`` not supported anymore')

    if start is not None or end is not None:
        assert start is not None and end is not None, (
            'start and end must be both None or both not None')

    if start is not None:
        assert _str_to_index(line, start) <= _str_to_index(line, end)

    if method == '4d' and delta0 is None:
        delta0 = 0

    if method == '6d' and delta0 is not None:
        raise ValueError('delta0 should be None when method is "6d"')

    if method == '6d' and zeta0 is not None:
        raise ValueError('zeta0 should be None when method is "6d"')

    if periodic_mode == 'periodic_symmetric':
        raise ValueError('``periodic_symmetric`` not supported anymore')
        assert R_matrix is None, 'R_matrix must be None for ``periodic_symmetric``'
        assert W_matrix is None, 'W_matrix must be None for ``periodic_symmetric``'

    if particle_on_co is not None:
        part_on_co = particle_on_co
    else:
        if search_for_t_rev:
            assert method == '6d', 'search_for_t_rev possible when ``method`` is "6d"'
        part_on_co = line.find_closed_orbit(
                                co_guess=co_guess,
                                particle_ref=particle_ref,
                                co_search_settings=co_search_settings,
                                continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                                delta0=delta0,
                                zeta0=zeta0,
                                zeta_shift=zeta_shift,
                                start=start,
                                end=end,
                                num_turns=num_turns,
                                co_search_at=co_search_at,
                                search_for_t_rev=search_for_t_rev,
                                spin=spin,
                                num_turns_search_t_rev=num_turns_search_t_rev,
                                symmetrize=False,
                                include_collective=include_collective
                                )
    if only_orbit:
        W_matrix = np.eye(6)


    if W_matrix is not None:
        W = W_matrix
        RR = None
    else:
        if R_matrix is not None:
            RR = R_matrix
            lnf._assert_matrix_responsiveness(RR, matrix_responsiveness_tol,
                                                only_4d=(method == '4d'))
            W, _, Rot, eigenvalues = lnf.get_linear_normal_form(
                        RR, only_4d_block=(method == '4d'),
                        symplectify=symplectify,
                        responsiveness_tol=matrix_responsiveness_tol,
                        stability_tol=matrix_stability_tol)
        else:
            steps_R_matrix['adapted'] = False
            for iter in range(2):
                RR_out = line.get_R_matrix(
                    steps=steps_R_matrix,
                    particle_on_co=part_on_co,
                    start=start,
                    end=end,
                    num_turns=num_turns,
                    element_by_element=compute_R_element_by_element,
                    only_markers=only_markers,
                    symmetrize=False,
                    include_collective=include_collective
                    )
                RR = RR_out['R_matrix']
                RR_ebe = RR_out['R_matrix_ebe']

                if matrix_responsiveness_tol is not None:
                    lnf._assert_matrix_responsiveness(RR,
                        matrix_responsiveness_tol, only_4d=(method == '4d'))

                W, _, Rot, eigenvalues = lnf.get_linear_normal_form(
                            RR, only_4d_block=(method == '4d'),
                            symplectify=symplectify,
                            responsiveness_tol=None,
                            stability_tol=None)

                # Estimate beam size (betatron part)
                gemitt_x = nemitt_x/part_on_co._xobject.beta0[0]/part_on_co._xobject.gamma0[0]
                gemitt_y = nemitt_y/part_on_co._xobject.beta0[0]/part_on_co._xobject.gamma0[0]
                betx_at_start = W[0, 0]**2 + W[0, 1]**2
                bety_at_start = W[2, 2]**2 + W[2, 3]**2
                gamx_at_start = W[1, 0]**2 + W[1, 1]**2
                gamy_at_start = W[3, 2]**2 + W[3, 3]**2
                sigma_x_start = np.sqrt(betx_at_start * gemitt_x)
                sigma_y_start = np.sqrt(bety_at_start * gemitt_y)
                sigma_px_start = np.sqrt(gamx_at_start * gemitt_x)
                sigma_py_start = np.sqrt(gamy_at_start * gemitt_y)

                if ((steps_R_matrix['dx'] < factor_adapt_steps * sigma_x_start)
                    and (steps_R_matrix['dy'] < factor_adapt_steps * sigma_y_start)
                    and (steps_R_matrix['dpx'] < factor_adapt_steps * sigma_px_start)
                    and (steps_R_matrix['dpy'] < factor_adapt_steps * sigma_py_start)):
                    break # sufficient accuracy
                else:
                    steps_R_matrix['dx'] = 0.01 * sigma_x_start
                    steps_R_matrix['dy'] = 0.01 * sigma_y_start
                    steps_R_matrix['dpx'] = 0.01 * sigma_px_start
                    steps_R_matrix['dpy'] = 0.01 * sigma_py_start
                    steps_R_matrix['adapted'] = True

    # Check on R matrix
    if RR is not None and matrix_stability_tol is not None:
        lnf._assert_matrix_determinant_within_tol(RR, matrix_stability_tol)
        if method == '4d':
            eigenvals = np.linalg.eigvals(RR[:4, :4])
        else:
            eigenvals = np.linalg.eigvals(RR)
        lnf._assert_matrix_stability(eigenvals, matrix_stability_tol)

    if method == '4d' and W_matrix is None: # the matrix was not provided by the user

        # Compute dispersion (MAD-8 manual eq. 6.13, but I needed to flip the sign ?!)
        A_disp = RR[:4, :4]
        b_disp = RR[:4, 5]
        delta_disp = np.linalg.solve(A_disp - np.eye(4), b_disp)
        dx_dpzeta = -delta_disp[0]
        dpx_dpzeta = -delta_disp[1]
        dy_dpzeta = -delta_disp[2]
        dpy_dpzeta = -delta_disp[3]

        b_disp_crab = RR[:4, 4]
        delta_disp_crab = np.linalg.solve(A_disp - np.eye(4), b_disp_crab)
        dx_zeta = -delta_disp_crab[0]
        dpx_zeta = -delta_disp_crab[1]
        dy_zeta = -delta_disp_crab[2]
        dpy_zeta = -delta_disp_crab[3]

        W[4:, :] = 0
        W[:, 4:] = 0
        W[4, 4] = 1
        W[5, 5] = 1
        W[0, 5] = dx_dpzeta
        W[1, 5] = dpx_dpzeta
        W[2, 5] = dy_dpzeta
        W[3, 5] = dpy_dpzeta
        W[0, 4] = dx_zeta
        W[1, 4] = dpx_zeta
        W[2, 4] = dy_zeta
        W[3, 4] = dpy_zeta

    if isinstance(start, str):
        tw_init_element_name = start
    elif start is None:
        tw_init_element_name = line._element_names_unique[0]
    else:
        tw_init_element_name = line._element_names_unique[start]

    init = TwissInit(particle_on_co=part_on_co, W_matrix=W,
                           element_name=tw_init_element_name,
                           ax_chrom=None, bx_chrom=None,
                           ay_chrom=None, by_chrom=None,
                           reference_frame='proper')

    return init, RR, steps_R_matrix, eigenvalues, Rot, RR_ebe
