# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging

import numpy as np
from scipy.optimize import fsolve

import xobjects as xo
import xpart as xp


from scipy.constants import c as clight

from . import linear_normal_form as lnf
from .table import Table
from .general import _print


import xtrack as xt  # To avoid circular imports

DEFAULT_STEPS_R_MATRIX = {
    'dx':1e-7, 'dpx':1e-10,
    'dy':1e-7, 'dpy':1e-10,
    'dzeta':1e-6, 'ddelta':1e-7
}

DEFAULT_CO_SEARCH_TOL = [1e-12, 1e-12, 1e-12, 1e-12, 1e-5, 1e-12]

AT_TURN_FOR_TWISS = -10 # # To avoid writing in monitors installed in the line

log = logging.getLogger(__name__)

def twiss_line(line, particle_ref=None, method='6d',
        particle_on_co=None, R_matrix=None, W_matrix=None,
        delta0=None, zeta0=None,
        r_sigma=0.01, nemitt_x=1e-6, nemitt_y=2.5e-6,
        delta_disp=1e-5, delta_chrom = 1e-4, zeta_disp=1e-3,
        particle_co_guess=None, steps_r_matrix=None,
        co_search_settings=None, at_elements=None, at_s=None,
        continue_on_closed_orbit_error=False,
        freeze_longitudinal=False,
        values_at_element_exit=False,
        radiation_method='full',
        eneloss_and_damping=False,
        ele_start=None, ele_stop=None, twiss_init=None,
        skip_global_quantities=False,
        matrix_responsiveness_tol=None,
        matrix_stability_tol=None,
        symplectify=False,
        reverse=False,
        use_full_inverse=None,
        strengths=False,
        hide_thin_groups=False,
        _continue_if_lost=False,
        _keep_tracking_data=False,
        ):

    """
    Compute the Twiss parameters of the beam line.

    Parameters
    ----------

    method : {'6d', '4d'}, optional
        Method to be used for the computation. If '6d' the full 6D
        normal form is used. If '4d' the 4D normal form is used.
    ele_start : int or str, optional
        Index of the element at which the computation starts. If not provided,
        the periodic sulution is computed. `twiss_init` must be provided if
        `ele_start` is provided.
    ele_stop : int or str, optional
        Index of the element at which the computation stops.
    twiss_init : TwissInit object, optional
        Initial values for the Twiss parameters.
    delta0 : float, optional
        Initial value for the delta parameter.
    zeta0 : float, optional
        Initial value for the zeta parameter.
    freeze_longitudinal : bool, optional
        If True, the longitudinal motion is frozen.
    at_elements : list, optional
        List of elements at which the Twiss parameters are computed.
        If not provided, the Twiss parameters are computed at all elements.
    at_s : list, optional
        List of positions at which the Twiss parameters are computed.
        If not provided, the Twiss parameters are computed at all positions.
    radiation_method : {'full', 'kick_as_co', 'scale_as_co'}, optional
        Method to be used for the computation of twiss parameters in the presence
        of radiation. If 'full' the method described in E. Forest, "From tracking
        code to analysis" is used. If 'kick_as_co' all particles receive the same
        radiation kicks as the closed orbit. If 'scale_as_co' all particles
        momenta are scaled by radiation as much as the closed orbit.
    eneloss_and_damping : bool, optional
        If True, the energy loss and radiation damping constants are computed.
    strengths : bool, optional
        If True, the strengths of the multipoles are added to the table.
    hide_thin_groups : bool, optional
        If True, values associate to elements in thin groups are replacede with
        NaNs.
    values_at_element_exit : bool, optional (False)
        If True, the Twiss parameters are computed at the exit of the
        elements. If False (default), the Twiss parameters are computed at the
        entrance of the elements.
    matrix_responsiveness_tol : float, optional
        Tolerance to be used tp check the responsiveness of the R matrix.
        If not provided, the default value is used.
    matrix_stability_tol : float, optional
        Tolerance to be used tp check the stability of the R matrix.
        If not provided, the default value is used.
    symplectify : bool, optional
        If True, the R matrix is symplectified before computing the linear normal
        form. Dafault is False.


    Returns
    -------

    twiss : xtrack.TwissTable
        Twiss calculation results. The table contains the following element-by-element quantities:
            - s: position of the element in meters
            - name: name of the element
            - x: horizontal position in meters (closed orbit for periodic solution)
            - px: horizontal momentum (closed orbit for periodic solution)
            - y: vertical position in meters (closed orbit for periodic solution)
            - py: vertical momentum (closed orbit for periodic solution)
            - zeta: longitudinal position in meters (closed orbit for periodic solution)
            - delta: longitudinal momentum deviation (closed orbit for periodic solution)
            - ptau: longitudinal momentum deviation (closed orbit for periodic solution)
            - betx: horizontal beta function
            - bety: vertical beta function
            - alfx: horizontal alpha function
            - alfy: vertical alpha function
            - gamx: horizontal gamma function
            - gamy: vertical gamma function
            - mux: horizontal phase advance
            - muy: vertical phase advance
            - muzeta: longitudinal phase advance
            - dx: horizontal dispersion (d x / d delta)
            - dy: vertical dispersion (d y / d delta)
            - dzeta: longitudinal dispersion (d zeta / d delta)
            - dpx: horizontal dispersion (d px / d delta)
            - dpy: vertical dispersion (d y / d delta)
            - dx_zeta: horizontal crab dispersion (d x / d zeta)
            - dy_zeta: vertical crab dispersion (d y / d zeta)
            - dpx_zeta: horizontal crab dispersion (d px / d zeta)
            - dpy_zeta: vertical crab dispersion (d py / d zeta)
            - W_matrix: W matrix of the linear normal form
            - betx1: computed horizontal beta function (Mais-Ripken)
            - bety1: computed vertical beta function (Mais-Ripken)
            - betx2: computed horizontal beta function (Mais-Ripken)
            - bety2: computed vertical beta function (Mais-Ripken)
        The table also contains the following global quantities:
            - qx: horizontal tune
            - qy: vertical tune
            - qs: synchrotron tune
            - dqx: horizontal chromaticity (d qx / d delta)
            - dqy: vertical chromaticity (d qy / d delta)
            - c_minus: closest tune approach coefficient
            - slip_factor: slip factor (1 / f_ref * d f_ref / d delta)
            - momentum_compaction_factor: momentum compaction factor
            - T_rev0: reference revolution period
            - partice_on_co: particle on closed orbit
            - R_matrix: R matrix (if calculated or provided)
            - eneloss_turn, energy loss per turn in electron volts (if
              eneloss_and_dampingis True)
            - damping_constants_turns, radiation damping constants per turn
              (if `eneloss_and_damping` is True)
            - damping_constants_s:
              radiation damping constants per second (if `eneloss_and_damping` is True)
            - partition_numbers:
                radiation partition numbers (if `eneloss_and_damping` is True)

    Notes
    -----

    The following additional parameters can also be provided:

        - particle_on_co : xpart.Particles, optional
            Particle on the closed orbit. If not provided, the closed orbit
            is searched for.
        - R_matrix : np.ndarray, optional
            R matrix to be used for the computation. If not provided, the
            R matrix is computed using finite differences.
        - W_matrix : np.ndarray, optional
            W matrix to be used for the computation. If not provided, the
            W matrix is computed from the R matrix.
        - particle_co_guess : xpart.Particles, optional
            Initial guess for the closed orbit. If not provided, zero is assumed.
        - co_search_settings : dict, optional
            Settings to be used for the closed orbit search.
            If not provided, the default values are used.
        - continue_on_closed_orbit_error : bool, optional
            If True, the computation is continued even if the closed orbit
            search fails.
        - delta_disp : float, optional
            Momentum deviation for the dispersion computation.
        - delta_chrom : float, optional
            Momentum deviation for the chromaticity computation.
        - skip_global_quantities : bool, optional
            If True, the global quantities are not computed.
        - use_full_inverse : bool, optional
            If True, the full inverse of the W matrik is used. If False, the inverse
            is computed from the symplectic condition.
        - steps_r_matrix : dict, optional
            Steps to be used for the finite difference computation of the R matrix.
            If not provided, the default values are used.
        - r_sigma : float, optional
            Deviation in sigmas used for the propagation of the W matrix.
            Initial value for the r_sigma parameter.
        - nemitt_x : float, optional
            Horizontal emittance assumed for the comutation of the deviation
            used for the propagation of the W matrix.
        - nemitt_y : float, optional
            Vertical emittance assumed for the comutation of the deviation
            used for the propagation of the W matrix.

    """

    assert method in ['6d', '4d'], 'Method must be `6d` or `4d`'

    if matrix_responsiveness_tol is None:
        matrix_responsiveness_tol = line.matrix_responsiveness_tol
    if matrix_stability_tol is None:
        matrix_stability_tol = line.matrix_stability_tol

    if line._radiation_model is not None:
        matrix_stability_tol = None
        if use_full_inverse is None:
            use_full_inverse = True

    if particle_ref is None:
        if particle_co_guess is None and hasattr(line, 'particle_ref'):
            particle_ref = line.particle_ref

    if line.iscollective:
        _print(
            'The line has collective elements.\n'
            'In the twiss computation collective elements are'
            ' replaced by drifts')
        line = line._get_non_collective_line()


    if particle_ref is None and particle_co_guess is None:
        raise ValueError(
            "Either `particle_ref` or `particle_co_guess` must be provided")

    if method == '4d' and delta0 is None:
        delta0 = 0

    if freeze_longitudinal:
        kwargs = locals().copy()
        kwargs.pop('freeze_longitudinal')

        with xt.freeze_longitudinal(line):
            return twiss_line(**kwargs)

    if radiation_method != 'full':
        kwargs = locals().copy()
        kwargs.pop('radiation_method')
        assert radiation_method in ['full', 'kick_as_co', 'scale_as_co']
        assert freeze_longitudinal is False
        with xt.line._preserve_config(line):
            if radiation_method == 'kick_as_co':
                assert isinstance(line._context, xo.ContextCpu) # needs to be serial
                assert eneloss_and_damping is False
                line.config.XTRACK_SYNRAD_KICK_SAME_AS_FIRST = True
            elif radiation_method == 'scale_as_co':
                assert isinstance(line._context, xo.ContextCpu) # needs to be serial
                line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = True
            res = twiss_line(**kwargs)
        return res

    if at_s is not None:
        # Get all arguments
        kwargs = locals().copy()
        if np.isscalar(at_s):
            at_s = [at_s]
        assert at_elements is None
        (auxtracker, names_inserted_markers
            ) = _build_auxiliary_tracker_with_extra_markers(
            tracker=line.tracker, at_s=at_s, marker_prefix='inserted_twiss_marker',
            algorithm='insert')
        kwargs.pop('line')
        kwargs.pop('at_s')
        kwargs.pop('at_elements')
        kwargs.pop('matrix_responsiveness_tol')
        kwargs.pop('matrix_stability_tol')
        res = twiss_line(line=auxtracker.line,
                        at_elements=names_inserted_markers,
                        matrix_responsiveness_tol=matrix_responsiveness_tol,
                        matrix_stability_tol=matrix_stability_tol,
                        **kwargs)
        return res

    if ele_start is not None or ele_stop is not None:
        if ele_start is not None and ele_stop is None:
            raise ValueError(
                'ele_stop must be specified if ele_start is not 0')
        elif ele_start is None and ele_stop is not None:
            raise ValueError(
                'ele_start must be specified if ele_stop is not None')
        assert twiss_init is not None, (
            'twiss_init must be provided if ele_start and ele_stop are used')

        if twiss_init == 'preserve':
            kwargs = locals().copy()
            kwargs.pop('twiss_init')
            kwargs.pop('ele_start')
            kwargs.pop('ele_stop')
            tw0 = twiss_line(**kwargs)
            twiss_init = tw0.get_twiss_init(at_element=ele_start)

        if isinstance(ele_start, str):
            ele_start = line.element_names.index(ele_start)
        if isinstance(ele_stop, str):
            ele_stop = line.element_names.index(ele_stop) + 1

        assert twiss_init.element_name == line.element_names[ele_start]
        particle_on_co = twiss_init.particle_on_co.copy()
        W_matrix = twiss_init.W_matrix
        skip_global_quantities = True
        mux0 = twiss_init.mux
        muy0 = twiss_init.muy
        muzeta0 = twiss_init.muzeta
    else:
        ele_start = 0
        mux0 = 0
        muy0 = 0
        muzeta0 = 0

    if particle_on_co is not None:
        part_on_co = particle_on_co
    else:
        part_on_co = line.find_closed_orbit(
                                particle_co_guess=particle_co_guess,
                                particle_ref=particle_ref,
                                co_search_settings=co_search_settings,
                                continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                                delta0=delta0,
                                zeta0=zeta0)

    if W_matrix is not None:
        W = W_matrix
        RR = None
    else:
        if R_matrix is not None:
            RR = R_matrix
        else:
            RR = line.compute_one_turn_matrix_finite_differences(
                                                steps_r_matrix=steps_r_matrix,
                                                particle_on_co=part_on_co)

        W, _, _ = lnf.compute_linear_normal_form(
                                RR, only_4d_block=(method == '4d'),
                                symplectify=symplectify,
                                responsiveness_tol=matrix_responsiveness_tol,
                                stability_tol=matrix_stability_tol)

    if method == '4d' and W_matrix is None: # the matrix was not provided by the user
        p_disp_minus = line.find_closed_orbit(
                            particle_co_guess=particle_co_guess,
                            particle_ref=particle_ref,
                            co_search_settings=co_search_settings,
                            continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                            delta0=delta0-delta_disp,
                            zeta0=zeta0)
        p_disp_plus = line.find_closed_orbit(particle_co_guess=particle_co_guess,
                            particle_ref=particle_ref,
                            co_search_settings=co_search_settings,
                            continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                            delta0=delta0+delta_disp,
                            zeta0=zeta0)
        p_disp_minus.move(_context=xo.context_default)
        p_disp_plus.move(_context=xo.context_default)
        dx_dpzeta = ((p_disp_plus.x[0] - p_disp_minus.x[0])
                     /(p_disp_plus.ptau[0] - p_disp_minus.ptau[0]))*part_on_co._xobject.beta0[0]
        dpx_dpzeta = ((p_disp_plus.px[0] - p_disp_minus.px[0])
                     /(p_disp_plus.ptau[0] - p_disp_minus.ptau[0]))*part_on_co._xobject.beta0[0]
        dy_dpzeta = ((p_disp_plus.y[0] - p_disp_minus.y[0])
                     /(p_disp_plus.ptau[0] - p_disp_minus.ptau[0]))*part_on_co._xobject.beta0[0]
        dpy_dpzeta = ((p_disp_plus.py[0] - p_disp_minus.py[0])
                      /(p_disp_plus.ptau[0] - p_disp_minus.ptau[0]))*part_on_co._xobject.beta0[0]

        W[4:, :] = 0
        W[:, 4:] = 0
        W[4, 4] = 1
        W[5, 5] = 1
        W[0, 5] = dx_dpzeta
        W[1, 5] = dpx_dpzeta
        W[2, 5] = dy_dpzeta
        W[3, 5] = dpy_dpzeta

    propagate_res = _propagate_optics(
        line=line,
        W_matrix=W,
        particle_on_co=part_on_co,
        mux0=mux0, muy0=muy0, muzeta0=muzeta0,
        ele_start=ele_start, ele_stop=ele_stop,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        r_sigma=r_sigma,
        delta_disp=delta_disp,
        zeta_disp=zeta_disp,
        use_full_inverse=use_full_inverse,
        hide_thin_groups=hide_thin_groups,
        _continue_if_lost=_continue_if_lost,
        _keep_tracking_data=_keep_tracking_data)
    propagate_res['name'] = np.array(propagate_res['name'])

    if method == '4d':
        # Not proper because R_matrix terms related to zeta are forced to zero
        propagate_res.pop('dx_zeta')
        propagate_res.pop('dy_zeta')

    twiss_res = TwissTable(data=propagate_res)

    twiss_res._data['particle_on_co'] = part_on_co.copy(_context=xo.context_default)

    circumference = line.tracker._tracker_data.line_length
    twiss_res._data['circumference'] = circumference

    if not skip_global_quantities:

        s_vect = propagate_res['s']
        mux = propagate_res['mux']
        muy = propagate_res['muy']

        dqx, dqy = _compute_chromaticity(
            line=line,
            W_matrix=W, method=method,
            particle_on_co=part_on_co,
            delta_chrom=delta_chrom,
            tune_x=mux[-1], tune_y=muy[-1],
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            matrix_responsiveness_tol=matrix_responsiveness_tol,
            matrix_stability_tol=matrix_stability_tol,
            symplectify=symplectify, steps_r_matrix=steps_r_matrix)

        dzeta = propagate_res['dzeta']
        qs = np.abs(propagate_res['muzeta'][-1])
        eta = -dzeta[-1]/circumference
        alpha = eta + 1/part_on_co._xobject.gamma0[0]**2

        beta0 = part_on_co._xobject.beta0[0]
        T_rev0 = circumference/clight/beta0
        betz0 = W[4, 4]**2 + W[4, 5]**2
        ptau_co = propagate_res['ptau']

        # Coupling
        r1 = (np.sqrt(propagate_res['bety1'])/
              np.sqrt(propagate_res['betx1']))
        r2 = (np.sqrt(propagate_res['betx2'])/
              np.sqrt(propagate_res['bety2']))

        # Coupling (https://arxiv.org/pdf/2005.02753.pdf)
        cmin_arr = (2 * np.sqrt(r1*r2) *
                    np.abs(np.mod(mux[-1], 1) - np.mod(muy[-1], 1))
                    /(1 + r1 * r2))
        c_minus = np.trapz(cmin_arr, s_vect)/(circumference)
        c_r1_avg = np.trapz(r1, s_vect)/(circumference)
        c_r2_avg = np.trapz(r2, s_vect)/(circumference)
        twiss_res._data.update({
            'qx': mux[-1], 'qy': muy[-1], 'qs': qs, 'dqx': dqx, 'dqy': dqy,
            'slip_factor': eta, 'momentum_compaction_factor': alpha, 'betz0': betz0,
            'circumference': circumference, 'T_rev0': T_rev0,
            'particle_on_co':part_on_co.copy(_context=xo.context_default),
            'c_minus': c_minus, 'c_r1_avg': c_r1_avg, 'c_r2_avg': c_r2_avg
        })
        if hasattr(part_on_co, '_fsolve_info'):
            twiss_res.particle_on_co._fsolve_info = part_on_co._fsolve_info
        else:
            twiss_res.particle_on_co._fsolve_info = None

        twiss_res._data['R_matrix'] = RR

        if method == '4d':
            twiss_res._data['qs'] = 0
            twiss_res.muzeta[:] = 0

        if eneloss_and_damping:
            assert RR is not None
            eneloss_damp_res = _compute_eneloss_and_damping_rates(
                particle_on_co=part_on_co, R_matrix=RR, ptau_co=ptau_co, T_rev0=T_rev0)
            twiss_res._data.update(eneloss_damp_res)

    if values_at_element_exit:
        raise NotImplementedError
        # Untested
        name_exit = twiss_res.name[:-1]
        twiss_res = twiss_res[:, 1:]
        twiss_res['name'][:] = name_exit
        twiss_res._data['values_at'] = 'exit'
    else:
        twiss_res._data['values_at'] = 'entry'

    if strengths:
        strengths = _extract_knl_ksl(line, twiss_res['name'])
        twiss_res._data.update(strengths)
        twiss_res._col_names = (list(twiss_res._col_names) +
                                    list(strengths.keys()))

    if at_elements is not None:
        twiss_res = twiss_res[:, at_elements]

    if reverse:
        raise ValueError('`twiss(..., reverse=True)` not supported anymore. '
                         'Use `twiss(...).reverse()` instead.')
    return twiss_res

def _propagate_optics(line, W_matrix, particle_on_co,
                      mux0, muy0, muzeta0,
                      ele_start, ele_stop,
                      nemitt_x, nemitt_y, r_sigma,
                      delta_disp, zeta_disp,
                      use_full_inverse,
                      hide_thin_groups=False,
                      _continue_if_lost=False,
                      _keep_tracking_data=False):

    ctx2np = line._context.nparray_from_context_array

    gemitt_x = nemitt_x/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    gemitt_y = nemitt_y/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    scale_transverse_x = np.sqrt(gemitt_x)*r_sigma
    scale_transverse_y = np.sqrt(gemitt_y)*r_sigma
    scale_longitudinal = delta_disp
    scale_eigen = min(scale_transverse_x, scale_transverse_y, scale_longitudinal)


    context = line._context
    part_for_twiss = xp.build_particles(_context=context,
                        particle_ref=particle_on_co, mode='shift',
                        x=  list(W_matrix[0, :] * scale_eigen) + [0],
                        px= list(W_matrix[1, :] * scale_eigen) + [0],
                        y=  list(W_matrix[2, :] * scale_eigen) + [0],
                        py= list(W_matrix[3, :] * scale_eigen) + [0],
                        zeta = list(W_matrix[4, :] * scale_eigen) + [0],
                        pzeta = list(W_matrix[5, :] * scale_eigen) + [0],
                        )

    part_disp = xp.build_particles(
        _context=context,
        x_norm=0,
        zeta=particle_on_co._xobject.zeta[0],
        delta=np.array([-delta_disp, +delta_disp])+particle_on_co._xobject.delta[0],
        particle_on_co=particle_on_co,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        W_matrix=W_matrix)
    part_zeta_disp = xp.build_particles(
        _context=context,
        x_norm=0,
        delta=particle_on_co._xobject.delta[0],
        zeta=np.array([-zeta_disp, +zeta_disp])+particle_on_co._xobject.zeta[0],
        particle_on_co=particle_on_co,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        W_matrix=W_matrix)

    part_for_twiss = xp.Particles.merge([part_for_twiss, part_disp, part_zeta_disp])
    part_for_twiss.s = particle_on_co._xobject.s[0]
    part_for_twiss.at_element = particle_on_co._xobject.at_element[0]
    i_start = part_for_twiss._xobject.at_element[0]

    part_for_twiss.at_turn = AT_TURN_FOR_TWISS # To avoid writing in monitors

    #assert np.all(ctx2np(part_for_twiss.at_turn) == 0)
    line.track(part_for_twiss, turn_by_turn_monitor='ONE_TURN_EBE',
                  ele_start=ele_start, ele_stop=ele_stop)
    if not _continue_if_lost:
        assert np.all(ctx2np(part_for_twiss.state) == 1), (
            'Some test particles were lost during twiss!')
    i_stop = part_for_twiss._xobject.at_element[0] + (
        (part_for_twiss._xobject.at_turn[0] - AT_TURN_FOR_TWISS)
         * len(line.element_names))

    x_co = line.record_last_track.x[6, i_start:i_stop+1].copy()
    y_co = line.record_last_track.y[6, i_start:i_stop+1].copy()
    px_co = line.record_last_track.px[6, i_start:i_stop+1].copy()
    py_co = line.record_last_track.py[6, i_start:i_stop+1].copy()
    zeta_co = line.record_last_track.zeta[6, i_start:i_stop+1].copy()
    delta_co = line.record_last_track.delta[6, i_start:i_stop+1].copy()
    ptau_co = line.record_last_track.ptau[6, i_start:i_stop+1].copy()
    s_co = line.record_last_track.s[6, i_start:i_stop+1].copy()

    x_disp_minus = line.record_last_track.x[7, i_start:i_stop+1].copy()
    y_disp_minus = line.record_last_track.y[7, i_start:i_stop+1].copy()
    zeta_disp_minus = line.record_last_track.zeta[7, i_start:i_stop+1].copy()
    px_disp_minus = line.record_last_track.px[7, i_start:i_stop+1].copy()
    py_disp_minus = line.record_last_track.py[7, i_start:i_stop+1].copy()
    delta_disp_minus = line.record_last_track.delta[7, i_start:i_stop+1].copy()

    x_disp_plus = line.record_last_track.x[8, i_start:i_stop+1].copy()
    y_disp_plus = line.record_last_track.y[8, i_start:i_stop+1].copy()
    zeta_disp_plus = line.record_last_track.zeta[8, i_start:i_stop+1].copy()
    px_disp_plus = line.record_last_track.px[8, i_start:i_stop+1].copy()
    py_disp_plus = line.record_last_track.py[8, i_start:i_stop+1].copy()
    delta_disp_plus = line.record_last_track.delta[8, i_start:i_stop+1].copy()

    x_zeta_disp_minus = line.record_last_track.x[9, i_start:i_stop+1].copy()
    y_zeta_disp_minus = line.record_last_track.y[9, i_start:i_stop+1].copy()
    zeta_crab_disp_minus = line.record_last_track.zeta[9, i_start:i_stop+1].copy()
    px_zeta_disp_minus = line.record_last_track.px[9, i_start:i_stop+1].copy()
    py_zeta_disp_minus = line.record_last_track.py[9, i_start:i_stop+1].copy()

    x_zeta_disp_plus = line.record_last_track.x[10, i_start:i_stop+1].copy()
    y_zeta_disp_plus = line.record_last_track.y[10, i_start:i_stop+1].copy()
    zeta_crab_disp_plus = line.record_last_track.zeta[10, i_start:i_stop+1].copy()
    px_zeta_disp_plus = line.record_last_track.px[10, i_start:i_stop+1].copy()
    py_zeta_disp_plus = line.record_last_track.py[10, i_start:i_stop+1].copy()

    dx = (x_disp_plus-x_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dy = (y_disp_plus-y_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dzeta = (zeta_disp_plus-zeta_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dpx = (px_disp_plus-px_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dpy = (py_disp_plus-py_disp_minus)/(delta_disp_plus - delta_disp_minus)

    dx_zeta = (x_zeta_disp_plus-x_zeta_disp_minus)/(zeta_crab_disp_plus - zeta_crab_disp_minus)
    dy_zeta = (y_zeta_disp_plus-y_zeta_disp_minus)/(zeta_crab_disp_plus - zeta_crab_disp_minus)

    # To be tested
    # dpx_zeta = (px_zeta_disp_plus-px_zeta_disp_minus)/(zeta_disp_plus - zeta_disp_minus)
    # dpy_zeta = (py_zeta_disp_plus-py_zeta_disp_minus)/(zeta_disp_plus - zeta_disp_minus)

    Ws = np.zeros(shape=(len(s_co), 6, 6), dtype=np.float64)
    Ws[:, 0, :] = (line.record_last_track.x[:6, i_start:i_stop+1] - x_co).T / scale_eigen
    Ws[:, 1, :] = (line.record_last_track.px[:6, i_start:i_stop+1] - px_co).T / scale_eigen
    Ws[:, 2, :] = (line.record_last_track.y[:6, i_start:i_stop+1] - y_co).T / scale_eigen
    Ws[:, 3, :] = (line.record_last_track.py[:6, i_start:i_stop+1] - py_co).T / scale_eigen
    Ws[:, 4, :] = (line.record_last_track.zeta[:6, i_start:i_stop+1] - zeta_co).T / scale_eigen
    Ws[:, 5, :] = (line.record_last_track.ptau[:6, i_start:i_stop+1] - ptau_co).T / particle_on_co._xobject.beta0[0] / scale_eigen

    # For removal ot thin groups of elements
    i_take = [0]
    for ii in range(1, len(s_co)):
        if s_co[ii] > s_co[ii-1]:
            i_take[-1] = ii-1
            i_take.append(ii)
        else:
            i_take.append(i_take[-1])
    i_take = np.array(i_take)
    _temp_range = np.arange(0, len(s_co), 1, dtype=int)
    mask_replace = _temp_range != i_take
    mask_replace[-1] = False # Force keeping of the last element
    i_replace = _temp_range[mask_replace]
    i_replace_with = i_take[mask_replace]

    # Re normalize eigenvectors (needed when radiation is present)
    nux, nuy, nuzeta = _renormalize_eigenvectors(Ws)

    # Rotate eigenvectors to the Courant-Snyder basis
    phix = np.arctan2(Ws[:, 0, 1], Ws[:, 0, 0])
    phiy = np.arctan2(Ws[:, 2, 3], Ws[:, 2, 2])
    phizeta = np.arctan2(Ws[:, 4, 5], Ws[:, 4, 4])

    v1 = Ws[:, :, 0] + 1j * Ws[:, :, 1]
    v2 = Ws[:, :, 2] + 1j * Ws[:, :, 3]
    v3 = Ws[:, :, 4] + 1j * Ws[:, :, 5]
    for ii in range(6):
        v1[:, ii] *= np.exp(-1j * phix)
        v2[:, ii] *= np.exp(-1j * phiy)
        v3[:, ii] *= np.exp(-1j * phizeta)
    Ws[:, :, 0] = np.real(v1)
    Ws[:, :, 1] = np.imag(v1)
    Ws[:, :, 2] = np.real(v2)
    Ws[:, :, 3] = np.imag(v2)
    Ws[:, :, 4] = np.real(v3)
    Ws[:, :, 5] = np.imag(v3)

    # Computation of twiss parameters

    if use_full_inverse:
        betx, alfx, gamx, bety, alfy, gamy, bety1, betx2 = _extract_twiss_parameters_with_inverse(Ws)
    else:
        betx = Ws[:, 0, 0]**2 + Ws[:, 0, 1]**2
        bety = Ws[:, 2, 2]**2 + Ws[:, 2, 3]**2

        gamx = Ws[:, 1, 0]**2 + Ws[:, 1, 1]**2
        gamy = Ws[:, 3, 2]**2 + Ws[:, 3, 3]**2

        alfx = - Ws[:, 0, 0] * Ws[:, 1, 0] - Ws[:, 0, 1] * Ws[:, 1, 1]
        alfy = - Ws[:, 2, 2] * Ws[:, 3, 2] - Ws[:, 2, 3] * Ws[:, 3, 3]

        bety1 = Ws[:, 2, 0]**2 + Ws[:, 2, 1]**2
        betx2 = Ws[:, 0, 2]**2 + Ws[:, 0, 3]**2

    betx1 = betx
    bety2 = bety

    temp_phix = phix.copy()
    temp_phiy = phiy.copy()
    temp_phix[i_replace] = temp_phix[i_replace_with]
    temp_phiy[i_replace] = temp_phiy[i_replace_with]

    mux = np.unwrap(temp_phix)/2/np.pi
    muy = np.unwrap(temp_phiy)/2/np.pi

    muzeta = np.unwrap(phizeta)/2/np.pi

    mux = mux - mux[0] + mux0
    muy = muy - muy[0] + muy0
    muzeta = muzeta - muzeta[0] + muzeta0

    mux = np.abs(mux)
    muy = np.abs(muy)

    twiss_res_element_by_element = {
        'name': line.element_names[i_start:i_stop] + ('_end_point',),
        's': s_co,
        'x': x_co,
        'px': px_co,
        'y': y_co,
        'py': py_co,
        'zeta': zeta_co,
        'delta': delta_co,
        'ptau': ptau_co,
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'gamx': gamx,
        'gamy': gamy,
        'dx': dx,
        'dpx': dpx,
        'dy': dy,
        'dzeta': dzeta,
        'dpy': dpy,
        'dx_zeta': dx_zeta,
        'dy_zeta': dy_zeta,
        'mux': mux,
        'muy': muy,
        'muzeta': muzeta,
        'nux': nux,
        'nuy': nuy,
        'nuzeta': nuzeta,
        'W_matrix': Ws,
        'betx1': betx1,
        'bety1': bety1,
        'betx2': betx2,
        'bety2': bety2,
    }

    if _keep_tracking_data:
        twiss_res_element_by_element['tracking_data'] = line.record_last_track

    if hide_thin_groups:
        _vars_hide_changes = [
        'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau',
        'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy',
        'betx1', 'bety1', 'betx2', 'bety2',
        'dx', 'dpx', 'dy', 'dzeta', 'dpy',
        ]

        for key in _vars_hide_changes:
                twiss_res_element_by_element[key][i_replace] = np.nan

    return twiss_res_element_by_element

def _compute_chromaticity(line, W_matrix, particle_on_co, delta_chrom,
                    tune_x, tune_y,
                    nemitt_x, nemitt_y, matrix_responsiveness_tol,
                    matrix_stability_tol, symplectify, steps_r_matrix,
                    method='6d'
                    ):

    context = line._context

    part_chrom_plus = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=particle_on_co._xobject.zeta[0], delta=delta_chrom,
                particle_on_co=particle_on_co,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                W_matrix=W_matrix)
    RR_chrom_plus = line.compute_one_turn_matrix_finite_differences(
                                        particle_on_co=part_chrom_plus.copy(),
                                        steps_r_matrix=steps_r_matrix)
    (WW_chrom_plus, WWinv_chrom_plus, Rot_chrom_plus
        ) = lnf.compute_linear_normal_form(RR_chrom_plus,
                            only_4d_block=method=='4d',
                            responsiveness_tol=matrix_responsiveness_tol,
                            stability_tol=matrix_stability_tol,
                            symplectify=symplectify)
    qx_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][0])/(2*np.pi)
    qy_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][2])/(2*np.pi)

    part_chrom_minus = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=particle_on_co._xobject.zeta[0], delta=-delta_chrom,
                particle_on_co=particle_on_co,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                W_matrix=W_matrix)
    RR_chrom_minus = line.compute_one_turn_matrix_finite_differences(
                                        particle_on_co=part_chrom_minus.copy(),
                                        steps_r_matrix=steps_r_matrix)
    (WW_chrom_minus, WWinv_chrom_minus, Rot_chrom_minus
        ) = lnf.compute_linear_normal_form(RR_chrom_minus,
                            only_4d_block=(method=='4d'),
                            symplectify=symplectify,
                            stability_tol=matrix_stability_tol,
                            responsiveness_tol=matrix_responsiveness_tol)
    qx_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][0])/(2*np.pi)
    qy_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][2])/(2*np.pi)

    dist_from_half_integer_x = np.modf(tune_x)[0] - 0.5
    dist_from_half_integer_y = np.modf(tune_y)[0] - 0.5

    if np.abs(qx_chrom_plus - qx_chrom_minus) > np.abs(dist_from_half_integer_x):
        raise NotImplementedError(
                "Qx too close to half integer, impossible to evaluate Q'x")
    if np.abs(qy_chrom_plus - qy_chrom_minus) > np.abs(dist_from_half_integer_y):
        raise NotImplementedError(
                "Qy too close to half integer, impossible to evaluate Q'y")

    dqx = (qx_chrom_plus - qx_chrom_minus)/delta_chrom/2
    dqy = (qy_chrom_plus - qy_chrom_minus)/delta_chrom/2

    if dist_from_half_integer_x > 0:
        dqx = -dqx

    if dist_from_half_integer_y > 0:
        dqy = -dqy

    return dqx, dqy


def _compute_eneloss_and_damping_rates(particle_on_co, R_matrix, ptau_co, T_rev0):
    diff_ptau = np.diff(ptau_co)
    eloss_turn = -sum(diff_ptau[diff_ptau<0]) * particle_on_co._xobject.p0c[0]

    # Get eigenvalues
    w0, v0 = np.linalg.eig(R_matrix)

    # Sort eigenvalues
    indx = [
        int(np.floor(np.argmax(np.abs(v0[:, 2*ii]))/2)) for ii in range(3)]
    eigenvals = np.array([w0[ii*2] for ii in indx])

    # Damping constants and partition numbers
    energy0 = particle_on_co.mass0 * particle_on_co._xobject.gamma0[0]
    damping_constants_turns = -np.log(np.abs(eigenvals))
    damping_constants_s = damping_constants_turns / T_rev0
    partition_numbers = (
        damping_constants_turns* 2 * energy0/eloss_turn)

    eneloss_damp_res = {
        'eneloss_turn': eloss_turn,
        'damping_constants_turns': damping_constants_turns,
        'damping_constants_s':damping_constants_s,
        'partition_numbers': partition_numbers
    }

    return eneloss_damp_res

class ClosedOrbitSearchError(Exception):
    pass

def find_closed_orbit_line(line, particle_co_guess=None, particle_ref=None,
                      co_search_settings=None, delta_zeta=0,
                      delta0=None, zeta0=None,
                      continue_on_closed_orbit_error=False):

    if particle_co_guess is None:
        if particle_ref is None:
            if line.particle_ref is not None:
                particle_ref = line.particle_ref
            else:
                raise ValueError(
                    "Either `particle_co_guess` or `particle_ref` must be provided")

        particle_co_guess = particle_ref.copy()
        particle_co_guess.x = 0
        particle_co_guess.px = 0
        particle_co_guess.y = 0
        particle_co_guess.py = 0
        particle_co_guess.zeta = 0
        particle_co_guess.delta = 0
        particle_co_guess.s = 0
        particle_co_guess.at_element = 0
        particle_co_guess.at_turn = 0
    else:
        particle_ref = particle_co_guess

    if co_search_settings is None:
        co_search_settings = {}

    co_search_settings = co_search_settings.copy()
    if 'xtol' not in co_search_settings.keys():
        co_search_settings['xtol'] = 1e-6 # Relative error between calls

    particle_co_guess = particle_co_guess.copy(
                        _context=line._buffer.context)

    for shift_factor in [0, 1.]: # if not found at first attempt we shift slightly the starting point
        if shift_factor>0:
            _print('Warning! Need second attempt on closed orbit search')

        x0=np.array([particle_co_guess._xobject.x[0] + shift_factor * 1e-5,
                    particle_co_guess._xobject.px[0] + shift_factor * 1e-7,
                    particle_co_guess._xobject.y[0] + shift_factor * 1e-5,
                    particle_co_guess._xobject.py[0] + shift_factor * 1e-7,
                    particle_co_guess._xobject.zeta[0] + shift_factor * 1e-4,
                    particle_co_guess._xobject.delta[0] + shift_factor * 1e-5])
        if delta0 is not None and zeta0 is None:
            x0[5] = delta0
            _error_for_co = _error_for_co_search_4d_delta0
        elif delta0 is None and zeta0 is not None:
            x0[4] = zeta0
            _error_for_co = _error_for_co_search_4d_zeta0
        elif delta0 is not None and zeta0 is not None:
            _error_for_co = _error_for_co_search_4d_delta0_zeta0
        else:
            _error_for_co = _error_for_co_search_6d
        if zeta0 is not None:
            x0[4] = zeta0
        if np.all(np.abs(_error_for_co(x0, particle_co_guess, line, delta_zeta, delta0, zeta0))
                    < DEFAULT_CO_SEARCH_TOL):
            res = x0
            fsolve_info = 'taken_guess'
            ier = 1
            break

        (res, infodict, ier, mesg
            ) = fsolve(lambda p: _error_for_co(p, particle_co_guess, line, delta_zeta, delta0, zeta0),
                x0=x0,
                full_output=True,
                **co_search_settings)
        fsolve_info = {
            'res': res, 'info': infodict, 'ier': ier, 'mesg': mesg}
        if ier == 1:
            break

    if ier != 1 and not(continue_on_closed_orbit_error):
        raise ClosedOrbitSearchError

    particle_on_co = particle_co_guess.copy()
    particle_on_co.x = res[0]
    particle_on_co.px = res[1]
    particle_on_co.y = res[2]
    particle_on_co.py = res[3]
    particle_on_co.zeta = res[4]
    particle_on_co.delta = res[5]

    particle_on_co._fsolve_info = fsolve_info

    return particle_on_co

def _one_turn_map(p, particle_ref, line, delta_zeta):
    part = particle_ref.copy()
    part.x = p[0]
    part.px = p[1]
    part.y = p[2]
    part.py = p[3]
    part.zeta = p[4] + delta_zeta
    part.delta = p[5]
    part.at_turn = AT_TURN_FOR_TWISS

    line.track(part)
    p_res = np.array([
           part._xobject.x[0],
           part._xobject.px[0],
           part._xobject.y[0],
           part._xobject.py[0],
           part._xobject.zeta[0],
           part._xobject.delta[0]])
    return p_res

def _error_for_co_search_6d(p, particle_co_guess, line, delta_zeta, delta0, zeta0):
    return p - _one_turn_map(p, particle_co_guess, line, delta_zeta)

def _error_for_co_search_4d_delta0(p, particle_co_guess, line, delta_zeta, delta0, zeta0):
    one_turn_res = _one_turn_map(p, particle_co_guess, line, delta_zeta)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        0,
        p[5] - delta0])

def _error_for_co_search_4d_zeta0(p, particle_co_guess, line, delta_zeta, delta0, zeta0):
    one_turn_res = _one_turn_map(p, particle_co_guess, line, delta_zeta)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        p[4] - zeta0,
        0])

def _error_for_co_search_4d_delta0_zeta0(p, particle_co_guess, line, delta_zeta, delta0, zeta0):
    one_turn_res = _one_turn_map(p, particle_co_guess, line, delta_zeta)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        p[4] - zeta0,
        p[5] - delta0])

def compute_one_turn_matrix_finite_differences(
        line, particle_on_co,
        steps_r_matrix=None):

    if steps_r_matrix is not None:
        steps_in = steps_r_matrix.copy()
        for nn in steps_in.keys():
            assert nn in DEFAULT_STEPS_R_MATRIX.keys(), (
                '`steps_r_matrix` can contain only ' +
                ' '.join(DEFAULT_STEPS_R_MATRIX.keys())
            )
        steps_r_matrix = DEFAULT_STEPS_R_MATRIX.copy()
        steps_r_matrix.update(steps_in)
    else:
        steps_r_matrix = DEFAULT_STEPS_R_MATRIX.copy()

    context = line._buffer.context

    particle_on_co = particle_on_co.copy(
                        _context=context)

    dx = steps_r_matrix["dx"]
    dpx = steps_r_matrix["dpx"]
    dy = steps_r_matrix["dy"]
    dpy = steps_r_matrix["dpy"]
    dzeta = steps_r_matrix["dzeta"]
    ddelta = steps_r_matrix["ddelta"]
    part_temp = xp.build_particles(_context=context,
            particle_ref=particle_on_co, mode='shift',
            x  =    [dx,  0., 0.,  0.,    0.,     0., -dx,   0.,  0.,   0.,     0.,      0.],
            px =    [0., dpx, 0.,  0.,    0.,     0.,  0., -dpx,  0.,   0.,     0.,      0.],
            y  =    [0.,  0., dy,  0.,    0.,     0.,  0.,   0., -dy,   0.,     0.,      0.],
            py =    [0.,  0., 0., dpy,    0.,     0.,  0.,   0.,  0., -dpy,     0.,      0.],
            zeta =  [0.,  0., 0.,  0., dzeta,     0.,  0.,   0.,  0.,   0., -dzeta,      0.],
            delta = [0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],
            )
    dpzeta = float(context.nparray_from_context_array(
        (part_temp.ptau[5] - part_temp.ptau[11])/2/part_temp.beta0[0]))
    if particle_on_co._xobject.at_element[0]>0:
        part_temp.s[:] = particle_on_co._xobject.s[0]
        part_temp.at_element[:] = particle_on_co._xobject.at_element[0]

    part_temp.at_turn = AT_TURN_FOR_TWISS

    if particle_on_co._xobject.at_element[0]>0:
        i_start = particle_on_co._xobject.at_element[0]
        line.track(part_temp, ele_start=i_start)
        line.track(part_temp, num_elements=i_start)
    else:
        assert particle_on_co._xobject.at_element[0] == 0
        line.track(part_temp)

    temp_mat = np.zeros(shape=(6, 12), dtype=np.float64)
    temp_mat[0, :] = context.nparray_from_context_array(part_temp.x)
    temp_mat[1, :] = context.nparray_from_context_array(part_temp.px)
    temp_mat[2, :] = context.nparray_from_context_array(part_temp.y)
    temp_mat[3, :] = context.nparray_from_context_array(part_temp.py)
    temp_mat[4, :] = context.nparray_from_context_array(part_temp.zeta)
    temp_mat[5, :] = context.nparray_from_context_array(
                                part_temp.ptau/part_temp.beta0) # pzeta

    RR = np.zeros(shape=(6, 6), dtype=np.float64)

    for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
        RR[:, jj] = (temp_mat[:, jj] - temp_mat[:, jj+6])/(2*dd)

    return RR


def _build_auxiliary_tracker_with_extra_markers(tracker, at_s, marker_prefix,
                                                algorithm='auto'):

    assert algorithm in ['auto', 'insert', 'regen_all_drift']
    if algorithm == 'auto':
        if len(at_s)<10:
            algorithm = 'insert'
        else:
            algorithm = 'regen_all_drifts'

    auxline = xt.Line(elements=list(tracker.line.elements).copy(),
                      element_names=list(tracker.line.element_names).copy())

    names_inserted_markers = []
    markers = []
    for ii, ss in enumerate(at_s):
        nn = marker_prefix + f'{ii}'
        names_inserted_markers.append(nn)
        markers.append(xt.Drift(length=0))

    if algorithm == 'insert':
        for nn, mm, ss in zip(names_inserted_markers, markers, at_s):
            auxline.insert_element(element=mm, name=nn, at_s=ss)
    elif algorithm == 'regen_all_drifts':
        raise ValueError # This algorithm is not enabled since it is not fully tested
        # s_elems = auxline.get_s_elements()
        # s_keep = []
        # enames_keep = []
        # for ss, nn in zip(s_elems, auxline.element_names):
        #     if not (_behaves_like_drift(auxline[nn]) and np.abs(auxline[nn].length)>0):
        #         s_keep.append(ss)
        #         enames_keep.append(nn)
        #         assert not xt.line._is_thick(auxline[nn]) or auxline[nn].length == 0

        # s_keep.extend(list(at_s))
        # enames_keep.extend(names_inserted_markers)

        # ind_sorted = np.argsort(s_keep)
        # s_keep = np.take(s_keep, ind_sorted)
        # enames_keep = np.take(enames_keep, ind_sorted)

        # i_new_drift = 0
        # new_enames = []
        # new_ele_dict = auxline.element_dict.copy()
        # new_ele_dict.update({nn: ee for nn, ee in zip(names_inserted_markers, markers)})
        # s_curr = 0
        # for ss, nn in zip(s_keep, enames_keep):
        #     if ss > s_curr + 1e-6:
        #         new_drift = xt.Drift(length=ss-s_curr)
        #         new_dname = f'_auxrift_{i_new_drift}'
        #         new_ele_dict[new_dname] = new_drift
        #         new_enames.append(new_dname)
        #         i_new_drift += 1
        #         s_curr = ss
        #     new_enames.append(nn)
        # auxline = xt.Line(elements=new_ele_dict, element_names=new_enames)

    auxtracker = xt.Tracker(
        _buffer=tracker._buffer,
        line=auxline,
        track_kernel=tracker.track_kernel,
        element_classes=tracker.element_classes,
        particles_class=tracker.particles_class,
        particles_monitor_class=None,
        local_particle_src=tracker.local_particle_src
    )
    auxtracker.line.config = tracker.line.config.copy()
    auxtracker.line._extra_config = tracker.line._extra_config.copy()

    return auxtracker, names_inserted_markers



class TwissInit:
    def __init__(self, particle_on_co=None, W_matrix=None, element_name=None,
                 mux=0, muy=0, muzeta=0.):
        self.__dict__['particle_on_co'] = particle_on_co
        self.W_matrix = W_matrix
        self.element_name = element_name
        self.mux = mux
        self.muy = muy
        self.muzeta = muzeta

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.__dict__['particle_on_co'], name):
            return getattr(self.__dict__['particle_on_co'], name)
        else:
            raise AttributeError(f'No attribute {name} found in TwissInit')

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        elif hasattr(self.particle_on_co, name):
            setattr(self.particle_on_co, name, value)
        else:
            self.__dict__[name] = value

class TwissTable(Table):

    def to_pandas(self, index=None, columns=None):
        if columns is None:
            columns = self._col_names

        data = self._data.copy()
        if 'W_matrix' in data.keys():
            data['W_matrix'] = [
                self.W_matrix[ii] for ii in range(len(self.W_matrix))]

        import pandas as pd
        df = pd.DataFrame(data, columns=self._col_names)
        if index is not None:
            df.set_index(index, inplace=True)
        return df

    def get_twiss_init(self, at_element):

        assert self.values_at == 'entry', 'Not yet implemented for exit'

        if isinstance(at_element, str):
            at_element = np.where(self.name == at_element)[0][0]
        part = self.particle_on_co.copy()
        part.x[:] = self.x[at_element]
        part.px[:] = self.px[at_element]
        part.y[:] = self.y[at_element]
        part.py[:] = self.py[at_element]
        part.zeta[:] = self.zeta[at_element]
        part.ptau[:] = self.ptau[at_element]
        part.s[:] = self.s[at_element]
        part.at_element[:] = at_element

        W = self.W_matrix[at_element]

        return TwissInit(particle_on_co=part, W_matrix=W,
                        element_name=self.name[at_element],
                        mux=self.mux[at_element],
                        muy=self.muy[at_element],
                        muzeta=self.muzeta[at_element])

    def get_betatron_sigmas(self, nemitt_x, nemitt_y, gemitt_z=0):

        beta0 = self.particle_on_co.beta0
        gamma0 = self.particle_on_co.gamma0
        gemitt_x = nemitt_x / (beta0 * gamma0)
        gemitt_y = nemitt_y / (beta0 * gamma0)

        Ws = self.W_matrix
        v1 = Ws[:,:,0] + 1j * Ws[:,:,1]
        v2 = Ws[:,:,2] + 1j * Ws[:,:,3]
        v3 = Ws[:,:,4] + 1j * Ws[:,:,5]

        Sigma1 = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)
        Sigma2 = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)
        Sigma3 = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)

        for ii in range(6):
            for jj in range(6):
                Sigma1[:, ii, jj] = np.real(v1[:,ii] * v1[:,jj].conj())
                Sigma2[:, ii, jj] = np.real(v2[:,ii] * v2[:,jj].conj())
                Sigma3[:, ii, jj] = np.real(v3[:,ii] * v3[:,jj].conj())

        Sigma = gemitt_x * Sigma1 + gemitt_y * Sigma2

        res_data = {}
        res_data['s'] = self.s.copy()
        res_data['name'] = self.name

        res_data['Sigma'] = Sigma
        res_data['Sigma11'] = Sigma[:, 0, 0]
        res_data['Sigma12'] = Sigma[:, 0, 1]
        res_data['Sigma13'] = Sigma[:, 0, 2]
        res_data['Sigma14'] = Sigma[:, 0, 3]
        res_data['Sigma15'] = Sigma[:, 0, 4]
        res_data['Sigma16'] = Sigma[:, 0, 5]

        res_data['Sigma21'] = Sigma[:, 1, 0]
        res_data['Sigma22'] = Sigma[:, 1, 1]
        res_data['Sigma23'] = Sigma[:, 1, 2]
        res_data['Sigma24'] = Sigma[:, 1, 3]
        res_data['Sigma25'] = Sigma[:, 1, 4]
        res_data['Sigma26'] = Sigma[:, 1, 5]

        res_data['Sigma31'] = Sigma[:, 2, 0]
        res_data['Sigma32'] = Sigma[:, 2, 1]
        res_data['Sigma33'] = Sigma[:, 2, 2]
        res_data['Sigma34'] = Sigma[:, 2, 3]
        res_data['Sigma41'] = Sigma[:, 3, 0]
        res_data['Sigma42'] = Sigma[:, 3, 1]
        res_data['Sigma43'] = Sigma[:, 3, 2]
        res_data['Sigma44'] = Sigma[:, 3, 3]
        res_data['Sigma51'] = Sigma[:, 4, 0]
        res_data['Sigma52'] = Sigma[:, 4, 1]

        res_data['sigma_x'] = np.sqrt(Sigma[:, 0, 0])
        res_data['sigma_y'] = np.sqrt(Sigma[:, 2, 2])
        res_data['sigma_z'] = np.sqrt(Sigma[:, 4, 4])

        return Table(res_data)

    def get_R_matrix(self, ele_start, ele_stop):

        assert self.values_at == 'entry', 'Not yet implemented for exit'

        if isinstance(ele_start, str):
            ele_start = np.where(self.name == ele_start)[0][0]
        if isinstance(ele_stop, str):
            ele_stop = np.where(self.name == ele_stop)[0][0]

        if ele_start > ele_stop:
            raise ValueError('ele_start must be smaller than ele_end')

        W_start = self.W_matrix[ele_start]
        W_end = self.W_matrix[ele_stop]

        mux_start = self.mux[ele_start]
        mux_end = self.mux[ele_stop]
        muy_start = self.muy[ele_start]
        muy_end = self.muy[ele_stop]
        muzeta_start = self.muzeta[ele_start]
        muzeta_end = self.muzeta[ele_stop]

        phi_x = 2 * np.pi * (mux_end - mux_start)
        phi_y = 2 * np.pi * (muy_end - muy_start)
        phi_zeta = 2 * np.pi * (muzeta_end - muzeta_start)

        Rot = np.zeros(shape=(6, 6), dtype=np.float64)

        Rot[0:2,0:2] = lnf.Rot2D(phi_x)
        Rot[2:4,2:4] = lnf.Rot2D(phi_y)
        Rot[4:6,4:6] = lnf.Rot2D(phi_zeta)

        R_matrix = W_end @ Rot @ np.linalg.inv(W_start)

        return R_matrix

    def get_normalized_coordinates(self, particles, nemitt_x=None, nemitt_y=None,
                                   _force_at_element=None):

        # TODO: check consistency of gamma0

        if nemitt_x is None:
            gemitt_x = 1
        else:
            gemitt_x = (nemitt_x / particles._xobject.beta0[0]
                        / particles._xobject.gamma0[0])

        if nemitt_y is None:
            gemitt_y = 1
        else:
            gemitt_y = (nemitt_y / particles._xobject.beta0[0]
                        / particles._xobject.gamma0[0])


        ctx2np = particles._context.nparray_from_context_array
        at_element_particles = ctx2np(particles.at_element)

        part_id = ctx2np(particles.particle_id).copy()
        at_element = part_id.copy() * 0 + xp.particles.LAST_INVALID_STATE
        x_norm = ctx2np(particles.x).copy() * 0 + xp.particles.LAST_INVALID_STATE
        px_norm = x_norm.copy()
        y_norm = x_norm.copy()
        py_norm = x_norm.copy()
        zeta_norm = x_norm.copy()
        pzeta_norm = x_norm.copy()

        at_element_no_rep = list(set(
            at_element_particles[part_id > xp.particles.LAST_INVALID_STATE]))

        for at_ele in at_element_no_rep:

            if _force_at_element is not None:
                at_ele = _force_at_element

            W = self.W_matrix[at_ele]
            W_inv = np.linalg.inv(W)

            mask_at_ele = at_element_particles == at_ele

            if _force_at_element is not None:
                mask_at_ele = ctx2np(particles.state) > xp.particles.LAST_INVALID_STATE

            n_at_ele = np.sum(mask_at_ele)

            # Coordinates wrt to the closed orbit
            XX = np.zeros(shape=(6, n_at_ele), dtype=np.float64)
            XX[0, :] = ctx2np(particles.x)[mask_at_ele] - self.x[at_ele]
            XX[1, :] = ctx2np(particles.px)[mask_at_ele] - self.px[at_ele]
            XX[2, :] = ctx2np(particles.y)[mask_at_ele] - self.y[at_ele]
            XX[3, :] = ctx2np(particles.py)[mask_at_ele] - self.py[at_ele]
            XX[4, :] = ctx2np(particles.zeta)[mask_at_ele] - self.zeta[at_ele]
            XX[5, :] = ((ctx2np(particles.ptau)[mask_at_ele] - self.ptau[at_ele])
                        / particles._xobject.beta0[0])

            XX_norm = np.dot(W_inv, XX)

            x_norm[mask_at_ele] = XX_norm[0, :] / np.sqrt(gemitt_x)
            px_norm[mask_at_ele] = XX_norm[1, :] / np.sqrt(gemitt_x)
            y_norm[mask_at_ele] = XX_norm[2, :] / np.sqrt(gemitt_y)
            py_norm[mask_at_ele] = XX_norm[3, :] / np.sqrt(gemitt_y)
            zeta_norm[mask_at_ele] = XX_norm[4, :]
            pzeta_norm[mask_at_ele] = XX_norm[5, :]
            at_element[mask_at_ele] = at_ele

        return Table({'particle_id': part_id, 'at_element': at_element,
                      'x_norm': x_norm, 'px_norm': px_norm, 'y_norm': y_norm,
                      'py_norm': py_norm, 'zeta_norm': zeta_norm,
                      'pzeta_norm': pzeta_norm}, index='particle_id')

    def reverse(self):

        assert self.values_at == 'entry', 'Not yet implemented for exit'

        new_data = {}
        for kk, vv in self._data.items():
            if hasattr(vv, 'copy'):
                new_data[kk] = vv.copy()
            else:
                new_data[kk] = vv

        for kk in self._col_names:
            if kk == 'name':
                new_data[kk][:-1] = new_data[kk][:-1][::-1]
                new_data[kk][-1] = self.name[-1]
            elif kk == 'W_matrix':
                continue
            elif kk.startswith('k') and kk.endswith('nl', 'sl'):
                continue # Not yet implemented
            else:
                new_data[kk] = new_data[kk][::-1].copy()

        out = self.__class__(data=new_data, col_names=self._col_names)

        circumference = (
            out.circumference if hasattr(out, 'circumference') else np.max(out.s))


        out.s = circumference - out.s

        out.x = -out.x
        out.px = out.px # Dx/Ds
        out.y = out.y
        out.py = -out.py # Dy/Ds
        out.zeta = -out.zeta
        out.delta = out.delta
        out.ptau = out.ptau

        out.betx = out.betx
        out.bety = out.bety
        out.alfx = -out.alfx # Dpx/Dx
        out.alfy = -out.alfy # Dpy/Dy
        out.gamx = out.gamx
        out.gamy = out.gamy

        qx = (out.qx if hasattr(out, 'qx') else np.max(out.mux))
        qy = (out.qy if hasattr(out, 'qy') else np.max(out.muy))
        qs = (out.qs if hasattr(out, 'qs') else np.max(out.muzeta))

        out.mux = qx - out.mux
        out.muy = qy - out.muy
        out.muzeta = qs - out.muzeta

        out.dx = -out.dx
        out.dpx = out.dpx
        out.dy = out.dy
        out.dpy = -out.dpy
        out.dzeta = -out.dzeta

        if 'dx_zeta' in out._col_names:
            out.dx_zeta = out.dx_zeta
            out.dy_zeta = -out.dy_zeta

        out.W_matrix = np.array(out.W_matrix)
        out.W_matrix = out.W_matrix[::-1, :, :].copy()
        out.W_matrix[:, 0, :] = -out.W_matrix[:, 0, :]
        out.W_matrix[:, 1, :] = out.W_matrix[:, 1, :]
        out.W_matrix[:, 2, :] = out.W_matrix[:, 2, :]
        out.W_matrix[:, 3, :] = -out.W_matrix[:, 3, :]
        out.W_matrix[:, 4, :] = -out.W_matrix[:, 4, :]
        out.W_matrix[:, 5, :] = out.W_matrix[:, 5, :]

        if hasattr(out, 'R_matrix'): out.R_matrix = None # To be implemented
        if hasattr(out, 'particle_on_co'):
            out.particle_on_co = self.particle_on_co.copy()
            out.particle_on_co.x = -out.particle_on_co.x
            out.particle_on_co.py = -out.particle_on_co.py
            out.particle_on_co.zeta = -out.particle_on_co.zeta

        if 'qs' in self.keys() and self.qs == 0:
            # 4d calculation
            out.qs = 0
            out.muzeta[:] = 0

        return out


def _renormalize_eigenvectors(Ws):
    # Re normalize eigenvectors
    v1 = Ws[:, :, 0] + 1j * Ws[:, :, 1]
    v2 = Ws[:, :, 2] + 1j * Ws[:, :, 3]
    v3 = Ws[:, :, 4] + 1j * Ws[:, :, 5]

    S = lnf.S
    S_v1_imag = v1 * 0.0
    S_v2_imag = v2 * 0.0
    S_v3_imag = v3 * 0.0
    for ii in range(6):
        for jj in range(6):
            if S[ii, jj] !=0:
                S_v1_imag[:, ii] +=  S[ii, jj] * v1.imag[:, jj]
                S_v2_imag[:, ii] +=  S[ii, jj] * v2.imag[:, jj]
                S_v3_imag[:, ii] +=  S[ii, jj] * v3.imag[:, jj]

    nux = np.squeeze(Ws[:, 0, 0]) * (0.0 + 0j)
    nuy = nux * 0.0
    nuzeta = nux * 0.0

    for ii in range(6):
        nux += v1.real[:, ii] * S_v1_imag[:, ii]
        nuy += v2.real[:, ii] * S_v2_imag[:, ii]
        nuzeta += v3.real[:, ii] * S_v3_imag[:, ii]

    nux = np.sqrt(np.abs(nux)) # nux is always positive
    nuy = np.sqrt(np.abs(nuy)) # nuy is always positive
    nuzeta = np.sqrt(np.abs(nuzeta)) # nuzeta is always positive

    for ii in range(6):
        v1[:, ii] /= nux
        v2[:, ii] /= nuy
        v3[:, ii] /= nuzeta

    Ws[:, :, 0] = np.real(v1)
    Ws[:, :, 1] = np.imag(v1)
    Ws[:, :, 2] = np.real(v2)
    Ws[:, :, 3] = np.imag(v2)
    Ws[:, :, 4] = np.real(v3)
    Ws[:, :, 5] = np.imag(v3)

    return nux, nuy, nuzeta


def _extract_twiss_parameters_with_inverse(Ws):

    # From E. Forest, "From tracking code to analysis", Sec 4.1.2 or better
    # https://iopscience.iop.org/article/10.1088/1748-0221/7/07/P07012

    EE = np.zeros(shape=(3, Ws.shape[0], 6, 6), dtype=np.float64)

    for ii in range(3):
        Iii = np.zeros(shape=(6, 6))
        Iii[2*ii, 2*ii] = 1
        Iii[2*ii+1, 2*ii+1] = 1
        Sii = lnf.S @ Iii

        Ws_inv = np.linalg.inv(Ws)

        EE[ii, :, :, :] = - Ws @ Sii @ Ws_inv @ lnf.S

    betx = EE[0, :, 0, 0]
    bety = EE[1, :, 2, 2]
    alfx = -EE[0, :, 0, 1]
    alfy = -EE[1, :, 2, 3]
    gamx = EE[0, :, 1, 1]
    gamy = EE[1, :, 3, 3]

    bety1 = np.abs(EE[0, :, 2, 2])
    betx2 = np.abs(EE[1, :, 0, 0])

    sign_x = np.sign(betx)
    sign_y = np.sign(bety)
    betx *= sign_x
    alfx *= sign_x
    gamx *= sign_x
    bety *= sign_y
    alfy *= sign_y
    gamy *= sign_y

    return betx, alfx, gamx, bety, alfy, gamy, bety1, betx2

def _extract_knl_ksl(line, names):

    knl = []
    ksl = []

    ctx2np = line._context.nparray_from_context_array

    for nn in names:
        if nn in line.element_names:
            if hasattr(line.element_dict[nn], 'knl'):
                knl.append(ctx2np(line.element_dict[nn].knl).copy())
            else:
                knl.append([])

            if hasattr(line.element_dict[nn], 'ksl'):
                ksl.append(ctx2np(line.element_dict[nn].ksl).copy())
            else:
                ksl.append([])
        else:
            knl.append([])
            ksl.append([])

    # Find maximum length of knl and ksl
    max_knl = 0
    max_ksl = 0
    for ii in range(len(knl)):
        max_knl = max(max_knl, len(knl[ii]))
        max_ksl = max(max_ksl, len(ksl[ii]))

    knl_array = np.zeros(shape=(len(knl), max_knl), dtype=np.float64)
    ksl_array = np.zeros(shape=(len(ksl), max_ksl), dtype=np.float64)

    for ii in range(len(knl)):
        knl_array[ii, :len(knl[ii])] = knl[ii]
        ksl_array[ii, :len(ksl[ii])] = ksl[ii]

    k_dict = {}
    for jj in range(max_knl):
        k_dict[f'k{jj}nl'] = knl_array[:, jj]
    for jj in range(max_ksl):
        k_dict[f'k{jj}sl'] = ksl_array[:, jj]

    return k_dict