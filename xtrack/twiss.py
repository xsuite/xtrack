# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging

import io
import json
from functools import partial
import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0
from scipy.constants import e as qe
from scipy.special import factorial
from scipy.constants import electron_volt

if hasattr(np, 'trapezoid'): # numpy >= 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz

import xobjects as xo
from xdeps import Table

from . import linear_normal_form as lnf
from .general import _print
from .twissplot import TwissPlot

import xtrack as xt  # To avoid circular imports

DEFAULT_STEPS_R_MATRIX = {
    'dx':1e-6, 'dpx':1e-7,
    'dy':1e-6, 'dpy':1e-7,
    'dzeta':1e-6, 'ddelta':1e-6
}

DEFAULT_CO_SEARCH_TOL = [1e-11, 1e-11, 1e-11, 1e-11, 1e-5, 1e-9]

DEFAULT_MATRIX_RESPONSIVENESS_TOL = 1e-15
DEFAULT_MATRIX_STABILITY_TOL = 2e-3
DEFAULT_NUM_TURNS_SEARCH_T_REV = 10

AT_TURN_FOR_TWISS = -10 # # To avoid writing in monitors installed in the line

VARS_FOR_TWISS_INIT_GENERATION = [
    'x', 'px', 'y', 'py', 'zeta', 'delta',
    'betx', 'alfx', 'bety', 'alfy', 'bets',
    'dx', 'dpx', 'dy', 'dpy', 'dzeta',
    'mux', 'muy', 'muzeta',
    'ax_chrom', 'bx_chrom', 'ay_chrom', 'by_chrom',
    'ddx', 'ddpx', 'ddy', 'ddpy',
]

CYCLICAL_QUANTITIES = ['mux', 'muy', 'dzeta', 's']

NORMAL_STRENGTHS_FROM_ATTR=['k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l']
SKEW_STRENGTHS_FROM_ATTR=['k0sl', 'k1sl', 'k2sl', 'k3sl', 'k4sl', 'k5sl']
OTHER_FIELDS_FROM_ATTR=['angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'length']
OTHER_FIELDS_FROM_TABLE=['element_type', 'isthick', 'parent_name']
SIGN_FLIP_FOR_ATTR_REVERSE=['k0l', 'k2l', 'k4l', 'k1sl', 'k3sl', 'k5sl', 'vkick', 'angle_rad']


log = logging.getLogger(__name__)

def twiss_line(line, particle_ref=None, method=None,
        particle_on_co=None, R_matrix=None, W_matrix=None,
        delta0=None, zeta0=None,
        r_sigma=None, nemitt_x=None, nemitt_y=None,
        delta_disp=None, delta_chrom=None, zeta_disp=None,
        co_guess=None, steps_r_matrix=None,
        co_search_settings=None, at_elements=None, at_s=None,
        continue_on_closed_orbit_error=None,
        freeze_longitudinal=None,
        freeze_energy=None,
        values_at_element_exit=None,
        radiation_method=None,
        eneloss_and_damping=None,
        radiation_integrals=None,
        start=None, end=None, init=None,
        num_turns=None,
        skip_global_quantities=None,
        matrix_responsiveness_tol=None,
        matrix_stability_tol=None,
        symplectify=None,
        reverse=None,
        use_full_inverse=None,
        strengths=None,
        hide_thin_groups=None,
        search_for_t_rev=None,
        num_turns_search_t_rev=None,
        only_twiss_init=None,
        only_orbit=None,
        compute_R_element_by_element=None,
        compute_lattice_functions=None,
        compute_chromatic_properties=None,
        init_at=None,
        x=None, px=None, y=None, py=None, zeta=None, delta=None,
        betx=None, alfx=None, bety=None, alfy=None, bets=None,
        dx=None, dpx=None, dy=None, dpy=None, dzeta=None,
        mux=None, muy=None, muzeta=None,
        ax_chrom=None, bx_chrom=None, ay_chrom=None, by_chrom=None,
        ddx=None, ddpx=None, ddy=None, ddpy=None,
        zero_at=None,
        co_search_at=None,
        include_collective=False,
        _continue_if_lost=None,
        _keep_tracking_data=None,
        _keep_initial_particles=None,
        _initial_particles=None,
        _ebe_monitor=None,
        only_markers=None,
        ):

    """
    Compute the Twiss parameters of the beam line.

    Parameters
    ----------

    method : {'6d', '4d'}, optional
        Method to be used for the computation. If '6d' the full 6D
        normal form is used. If '4d' the 4D normal form is used.
    start : int or str, optional
        Index of the element at which the computation starts. If not provided,
        the periodic solution is computed. `init` must be provided if
        `start` is provided.
    end : int or str, optional
        Index of the element at which the computation stops.
    init : TwissInit object, optional
        Initial values for the Twiss parameters. If `init="periodic"` is
        passed, the periodic solution for the selected range is computed.
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
        List of positions in meters at which the Twiss parameters are computed.
        If not provided, the Twiss parameters are computed at all positions.
    radiation_method : {'full', 'kick_as_co', 'scale_as_co'}, optional
        Method to be used for the computation of twiss parameters in the presence
        of radiation. If 'full' the method described in E. Forest, "From tracking
        code to analysis" is used. If 'kick_as_co' all particles receive the same
        radiation kicks as the closed orbit. If 'scale_as_co' all particles
        momenta are scaled by radiation as much as the closed orbit.
    radiation_integrals : bool, optional
        If True, the radiation integrals are computed.
    eneloss_and_damping : bool, optional
        If True, the energy loss and radiation damping constants are computed.
    strengths : bool, optional
        If True, the strengths of the multipoles are added to the table.
    hide_thin_groups : bool, optional
        If True, values associate to elements in thin groups are replacede with
        NaNs.
    search_for_t_rev : bool, optional
        If True, the revolution period is searched for, otherwise the revolution
        period computed from the circumference is assumed.
    num_turns_search_t_rev : int, optional
        Number of turns used for the search of the revolution period. Used only
        if `search_for_t_rev` is True.
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
            - betx: horizontal beta function in meters
            - bety: vertical beta function in meters
            - alfx: horizontal alpha function
            - alfy: vertical alpha function
            - gamx: horizontal gamma function in 1/meters
            - gamy: vertical gamma function in 1/meters
            - mux: horizontal phase advance in tune units (angle/2/pi)
            - muy: vertical phase advance in tune units (angle/2/pi)
            - muzeta: longitudinal phase advance in tune units (angle/2/pi)
            - dx: horizontal dispersion (d x / d delta) in meters
            - dy: vertical dispersion (d y / d delta) in meters
            - dzeta: longitudinal dispersion (d zeta / d delta) in meters
            - dpx: horizontal momentum dispersion (d px / d delta)
            - dpy: vertical momentum dispersion (d py / d delta)
            - ddx: horizontal second order dispersion (d^2 x / d delta^2) in meters
            - ddy: vertical second order dispersion (d^2 y / d delta^2) in meters
            - ddpx: horizontal second order dispersion (d^2 px / d delta^2)
            - ddpy: vertical second order dispersion (d^2 py / d delta^2)
            - dx_zeta: horizontal crab dispersion (d x / d zeta)
            - dy_zeta: vertical crab dispersion (d y / d zeta)
            - dpx_zeta: horizontal momentum crab dispersion (d px / d zeta)
            - dpy_zeta: vertical momentum crab dispersion (d py / d zeta)
            - ax_chrom: chromatic function (d alfx / d delta - alfx / betx d betx / d delta)
            - ay_chrom: chromatic function (d alfy / d delta - alfy / bety d bety / d delta)
            - bx_chrom: chromatic function (d betx / d delta)
            - by_chrom: chromatic function (d bety / d delta)
            - wx_chrom: sqrt(ax_chrom**2 + bx_chrom**2)
            - wy_chrom: sqrt(ay_chrom**2 + by_chrom**2)
            - W_matrix: W matrix of the linear normal form
            - betx1: computed horizontal beta function (Mais-Ripken) in meters
            - bety1: computed vertical beta function (Mais-Ripken) in meters
            - betx2: computed horizontal beta function (Mais-Ripken) in meters
            - bety2: computed vertical beta function (Mais-Ripken) in meters
            - c_minus_re: real part of the closest tune approach coefficient
            - c_minus_im: imaginary part of the closest tune approach coefficient
            - c_r1: horizontal r1 coefficient for betatron coupling
            - c_r2: vertical r2 coefficient for betatron coupling
            - c_phi1: phase advance of the closest tune approach coefficient
            - c_phi2: phase advance of the closest tune approach coefficient
        The table also contains the following global quantities:
            - qx: horizontal tune
            - qy: vertical tune
            - qs: synchrotron tune
            - dqx: horizontal chromaticity (d qx / d delta)
            - dqy: vertical chromaticity (d qy / d delta)
            - ddqx: horizontal second order chromaticity (d^2 qx / d delta^2)
            - ddqy: vertical second order chromaticity (d^2 qy / d delta^2)
            - c_minus: closest tune approach coefficient
            - c_minus_re_0: real part of the closest tune approach coefficient (at start of the ring)
            - c_minus_im_0: imaginary part of the closest tune approach coefficient (at start of the ring)
            - slip_factor: slip factor (-1 / f_ref * d f_ref / d delta) (positive above transition)
            - momentum_compaction_factor: momentum compaction factor (slip_factor + 1/gamma_0^2)
            - T_rev0: reference revolution period in seconds
            - circumference: reference trajectory length in meters
            - particle_on_co: particle on closed orbit
            - R_matrix: R matrix (if calculated or provided)
            - eneloss_turn, energy loss per turn in electron volts (if
              eneloss_and_damping is True)
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
        - co_guess : xpart.Particles or dict, optional
            Initial guess for the closed orbit. If not provided, zero is assumed.
        - co_search_settings : dict, optional
            Settings to be used for the closed orbit search.
            If not provided, the default values are used.
        - num_turns: int, optional
            If specified the periodic solution and the twiss table are computed
            on multiple turns.
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
    input_kwargs = locals().copy()

    # defaults
    r_sigma=(r_sigma or 0.01)
    nemitt_x=(nemitt_x or 1e-6)
    nemitt_y=(nemitt_y or 1e-6)
    delta_disp=(delta_disp or 1e-5)
    delta_chrom=(delta_chrom or 5e-5)
    zeta_disp=(zeta_disp or 1e-3)
    values_at_element_exit=(values_at_element_exit or False)
    continue_on_closed_orbit_error=(continue_on_closed_orbit_error or False)
    freeze_longitudinal=(freeze_longitudinal or False)
    radiation_method=(radiation_method or None)
    radiation_integrals=(radiation_integrals or False)
    eneloss_and_damping=(eneloss_and_damping or False)
    symplectify=(symplectify or False)
    reverse=(reverse or False)
    strengths=(strengths or False)
    hide_thin_groups=(hide_thin_groups or False)
    search_for_t_rev=(search_for_t_rev or False)
    num_turns_search_t_rev=(num_turns_search_t_rev or None)
    only_twiss_init=(only_twiss_init or False)
    only_markers=(only_markers or False)
    only_orbit=(only_orbit or False)
    compute_R_element_by_element=(compute_R_element_by_element or False)
    compute_lattice_functions=(compute_lattice_functions
                        if compute_lattice_functions is not None else True)
    compute_chromatic_properties=(compute_chromatic_properties
                        if compute_chromatic_properties is not None else None)
    num_turns = (num_turns or 1)

    if only_markers:
        raise NotImplementedError('`only_markers` not supported anymore')

    if isinstance(init, TwissInit):
        init = init.copy()

    kwargs = locals().copy()

    if (init is not None or betx is not None or bety is not None) and start is None:
        # is open twiss
        start = xt.START
        end = end or xt.END

    if num_turns != 1:
        # Untested cases
        assert num_turns > 0
        assert start is None
        assert end is None
        assert init is None
        assert reverse is False

    if zero_at is not None:
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        kwargs.pop('zero_at')
        out = twiss_line(**kwargs)
        out.zero_at(zero_at)
        return _add_action_in_res(out, input_kwargs)

    if start is not None:
        if isinstance(start, xt.match._LOC):
            assert start in [xt.START, xt.END]
            if reverse:
                start = {xt.START: xt.END, xt.END: xt.START}[start]
            start = {xt.START: line._element_names_unique[0],
                     xt.END: line._element_names_unique[-1]}[start]
        assert isinstance(start, str)  # index not supported anymore

    if end is not None:
        if isinstance(end, xt.match._LOC):
            assert end in [xt.START, xt.END]
            if reverse:
                end = {xt.START: xt.END, xt.END: xt.START}[end]
            end = {xt.START: line._element_names_unique[0],
                     xt.END: line._element_names_unique[-1]}[end]
        assert isinstance(end, str)  # index not supported anymore

    if start is not None and end is None:
        # One turn twiss from start to start
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        kwargs.pop('start')
        if (init is None or init == 'periodic') and betx is None and bety is None:
            # Periodic twiss
            tw = twiss_line(**kwargs)
            t1 = tw.rows[start:]
            t2 = tw.rows[:start]
            out = xt.TwissTable.concatenate([t1, t2])
            out.zero_at(out.name[0])
            out.name[-1] = '_end_point'
            out['periodic'] = True
            out['completed_init'] = tw.completed_init
        else:
            # Initial conditions are given -> open twiss
            kwargs.pop('end')
            t1o = twiss_line(start=start, end=xt.END, **kwargs)
            init_part2 = t1o.get_twiss_init('_end_point')
            # Dummy twiss to get the name at the start of the second part
            init_part2.element_name = line.twiss(
                start=xt.START, end=xt.START, betx=1, bety=1).name[0]

            for kk in VARS_FOR_TWISS_INIT_GENERATION:
                kwargs.pop(kk, None)
            kwargs.pop('init')
            t2o = twiss_line(start=xt.START, end=start, init=init_part2, **kwargs)
            # remove repeated element
            t2o = t2o.rows[:-1]
            t2o.name[-1] = '_end_point'
            out = xt.TwissTable.concatenate([t1o, t2o])
            out['completed_init'] = t1o.completed_init
        return _add_action_in_res(out, input_kwargs)

    if init == 'full_periodic' and (start is not None or end is not None):
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        kwargs.pop('init')
        kwargs.pop('start')
        kwargs.pop('end')
        kwargs.pop('init_at')
        tw = twiss_line(**kwargs) # Periodic twiss of the full line
        init = tw.get_twiss_init(init_at or start)
        out = twiss_line(start=start, end=end, init=init, **kwargs)
        if zero_at is None:
            out.zero_at(start)
        return _add_action_in_res(out, input_kwargs)
    elif (init is not None and init not in ['periodic', 'periodic_symmetric']
        or betx is not None or bety is not None):
        periodic = False
        periodic_mode = None
    else:
        periodic = True
        periodic_mode = init or 'periodic'

    if freeze_longitudinal:
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        kwargs.pop('freeze_longitudinal')

        with xt.freeze_longitudinal(line):
            return _add_action_in_res(twiss_line(**kwargs), input_kwargs)
    elif freeze_energy or (freeze_energy is None and method=='4d'):
        if not line._energy_is_frozen():
            kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
            kwargs.pop('freeze_energy')
            with xt.line._preserve_config(line):
                line.freeze_energy(force=True) # need to force for collective lines
                return _add_action_in_res(
                    twiss_line(freeze_energy=False, **kwargs), input_kwargs)

    if at_s is not None:
        if reverse:
            raise NotImplementedError('`at_s` not implemented for `reverse`=True')
        # Get all arguments
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
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
        return _add_action_in_res(res, input_kwargs)

    if radiation_method is None and line._radiation_model is not None:
        if line._radiation_model == 'quantum':
            raise ValueError(
                'twiss cannot be called when the radiation model is `quantum`')
        radiation_method = 'kick_as_co'

    if radiation_method is not None and radiation_method != 'full':
        assert isinstance(line._context, xo.ContextCpu), (
            'Twiss with radiation computation is only supported on CPU')
        assert not line._context.openmp_enabled, (
            'Twiss with radiation computation is not supported with OpenMP'
            ' parallelization')
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        assert radiation_method in ['full', 'kick_as_co', 'scale_as_co']
        assert freeze_longitudinal is False
        if (radiation_method == 'kick_as_co' and (
            not hasattr(line.config, 'XTRACK_SYNRAD_KICK_SAME_AS_FIRST') or
            not line.config.XTRACK_SYNRAD_KICK_SAME_AS_FIRST)):
            with xt.line._preserve_config(line):
                line.config.XTRACK_SYNRAD_KICK_SAME_AS_FIRST = True
                return _add_action_in_res(twiss_line(**kwargs), input_kwargs)
        elif (radiation_method == 'scale_as_co' and (
            not hasattr(line.config, 'XTRACK_SYNRAD_SCALE_SAME_AS_FIRST') or
            not line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST)):
            with xt.line._preserve_config(line):
                line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = True
                return _add_action_in_res(twiss_line(**kwargs), input_kwargs)

    if radiation_method == 'kick_as_co':
        assert hasattr(line.config, 'XTRACK_SYNRAD_KICK_SAME_AS_FIRST')
        assert line.config.XTRACK_SYNRAD_KICK_SAME_AS_FIRST

    if line.enable_time_dependent_vars:
        raise RuntimeError('Time dependent variables not supported in Twiss')

    if isinstance(init_at, xt.match._LOC):
        if init_at.name == 'START':
            init_at = start
        elif init_at.name == 'END':
            init_at = end

    if isinstance(init, TwissTable):
        if init_at is None:
            init_at = start
        init = init.get_twiss_init(at_element=init_at)
        init_at = None

    init = _complete_twiss_init(
        start=start, end=end, init_at=init_at, init=init,
        line=line, reverse=reverse,
        x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
        alfx=alfx, alfy=alfy, betx=betx, bety=bety, bets=bets,
        dx=dx, dpx=dpx, dy=dy, dpy=dpy, dzeta=dzeta,
        mux=mux, muy=muy, muzeta=muzeta,
        ax_chrom=ax_chrom, bx_chrom=bx_chrom, ay_chrom=ay_chrom, by_chrom=by_chrom,
        ddx=ddx, ddpx=ddpx, ddy=ddy, ddpy=ddpy,
        )
    completed_init = (init.copy() if hasattr(init, 'copy') else init)

    # clean quantities embedded in init
    init_at=None
    x=None; px=None; y=None; py=None; zeta=None; delta=None
    alfx=None; alfy=None; betx=None; bety=None; bets=None
    dx=None; dpx=None; dy=None; dpy=None; dzeta=None
    mux=None; muy=None; muzeta=None
    ax_chrom=None; bx_chrom=None; ay_chrom=None; by_chrom=None
    ddx=None; ddpx=None; ddy=None; ddpy=None

    # Twiss goes through the start of the line
    rv = (-1 if reverse else 1)
    if not periodic and (
        rv * _str_to_index(line, start) > rv * _str_to_index(line, end)):

        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        tw_res = _handle_loop_around(kwargs)

        return _add_action_in_res(tw_res, input_kwargs)

    # init is not at the boundary
    if (not periodic and not isinstance(init, str)
            and init.element_name != start
            and init.element_name != end):

        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        tw_res = _handle_init_inside_range(kwargs)

        return _add_action_in_res(tw_res, input_kwargs)

    if reverse:
        if start is not None and end is not None:
            assert (_str_to_index(line, start) >= _str_to_index(line, end)), (
                'start must be smaller than end in reverse mode')
        start, end = end, start
    else:
        if start is not None and end is not None:
            assert _str_to_index(line, start) <= _str_to_index(line, end), (
                'start must be larger than end in forward mode')

    if start is not None and init is None:
        assert init is not None, (
            'init must be provided if start and end are used')

    if matrix_responsiveness_tol is None:
        matrix_responsiveness_tol = line.matrix_responsiveness_tol
    if matrix_stability_tol is None:
        matrix_stability_tol = line.matrix_stability_tol

    if (line._radiation_model is not None
            and radiation_method != 'kick_as_co'):
        matrix_stability_tol = None
        if use_full_inverse is None:
            use_full_inverse = True

    if particle_ref is None:
        if particle_on_co is not None:
            particle_ref = particle_on_co.copy()
        elif co_guess is None and hasattr(line, 'particle_ref'):
            particle_ref = line.particle_ref

    if line.iscollective and not include_collective:
        _print(
            'The line has collective elements.\n'
            'In the twiss computation collective elements are'
            ' replaced by drifts')
        line = line._get_non_collective_line()

    if particle_ref is None and co_guess is None:
        raise ValueError(
            "Either `particle_ref` or `co_guess` must be provided")

    if method is None:
        method = '6d'

    assert method in ['6d', '4d'], 'Method must be `6d` or `4d`'

    if isinstance(init, str):
        if init in ['preserve', 'preserve_start', 'preserve_end']:
            raise ValueError(f'init={init} not anymore supported')
        assert init == 'periodic' or 'full_periodic'

    if periodic:

        assert not _initial_particles

        steps_r_matrix = _complete_steps_r_matrix_with_default(steps_r_matrix)

        init, R_matrix, steps_r_matrix, eigenvalues, Rot, RR_ebe = _find_periodic_solution(
            line=line, particle_on_co=particle_on_co,
            particle_ref=particle_ref, method=method,
            co_search_settings=co_search_settings,
            continue_on_closed_orbit_error=continue_on_closed_orbit_error,
            delta0=delta0, zeta0=zeta0, steps_r_matrix=steps_r_matrix,
            W_matrix=W_matrix, R_matrix=R_matrix,
            co_guess=co_guess,
            delta_disp=delta_disp, symplectify=symplectify,
            matrix_responsiveness_tol=matrix_responsiveness_tol,
            matrix_stability_tol=matrix_stability_tol,
            start=start, end=end,
            num_turns=num_turns,
            co_search_at=co_search_at,
            search_for_t_rev=search_for_t_rev,
            num_turns_search_t_rev=num_turns_search_t_rev,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y, r_sigma=r_sigma,
            compute_R_element_by_element=compute_R_element_by_element,
            only_markers=only_markers,
            only_orbit=only_orbit,
            periodic_mode=periodic_mode,
            include_collective=include_collective,
            )
    else:
        # force
        skip_global_quantities = True

    if only_twiss_init:
        assert periodic, '`only_twiss_init` can only be used in periodic mode'
        if reverse:
            return init.reverse()
        else:
            return init

    if only_markers and eneloss_and_damping:
        raise NotImplementedError(
            '`only_markers` not implemented for `eneloss_and_damping`')

    twiss_res = _twiss_open(
        line=line,
        init=init,
        start=start, end=end,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        r_sigma=r_sigma,
        delta_disp=delta_disp,
        zeta_disp=zeta_disp,
        use_full_inverse=use_full_inverse,
        hide_thin_groups=hide_thin_groups,
        only_markers=only_markers,
        only_orbit=only_orbit,
        compute_lattice_functions=compute_lattice_functions,
        _continue_if_lost=_continue_if_lost,
        _keep_tracking_data=_keep_tracking_data,
        _keep_initial_particles=_keep_initial_particles,
        _initial_particles=_initial_particles,
        _ebe_monitor=_ebe_monitor)



    if (not only_orbit and (
        (compute_chromatic_properties is True)
        or (compute_chromatic_properties is None and periodic))):

        with xt.line._preserve_config(line):
            line.freeze_energy(force=True)
            line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
            line.config.XTRACK_SYNRAD_KICK_SAME_AS_FIRST = False
            cols_chrom, scalars_chrom = _compute_chromatic_functions(
                line=line,
                init=init,
                delta_chrom=delta_chrom,
                steps_r_matrix=steps_r_matrix,
                matrix_responsiveness_tol=matrix_responsiveness_tol,
                matrix_stability_tol=matrix_stability_tol,
                symplectify=symplectify,
                method=method,
                use_full_inverse=use_full_inverse,
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
                on_momentum_twiss_res=twiss_res,
                r_sigma=r_sigma,
                delta_disp=delta_disp,
                zeta_disp=zeta_disp,
                start=start,
                end=end,
                num_turns=num_turns,
                hide_thin_groups=hide_thin_groups,
                only_markers=only_markers,
                periodic=periodic,
                periodic_mode=periodic_mode,
                include_collective=include_collective,
            )
        twiss_res._data.update(cols_chrom)
        twiss_res._data.update(scalars_chrom)
        twiss_res._col_names += list(cols_chrom.keys())

    if not skip_global_quantities and not only_orbit:
        twiss_res._data['R_matrix'] = R_matrix
        twiss_res._data['steps_r_matrix'] = steps_r_matrix
        twiss_res._data['R_matrix_ebe'] = RR_ebe

        _compute_global_quantities(
                            line=line, twiss_res=twiss_res)

        twiss_res._data['eigenvalues'] = eigenvalues.copy()
        twiss_res._data['rotation_matrix'] = Rot.copy()

    if eneloss_and_damping and not only_orbit:
        assert 'R_matrix' in twiss_res._data
        if radiation_method != 'full' or twiss_res._data['R_matrix_ebe'] is None:
            with xt.line._preserve_config(line):
                line.config.XTRACK_SYNRAD_KICK_SAME_AS_FIRST = False
                line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = False
                _, RR, _, _, _, RR_ebe = _find_periodic_solution(
                    line=line, particle_on_co=particle_on_co,
                    particle_ref=particle_ref, method='6d',
                    co_search_settings=co_search_settings,
                    continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                    steps_r_matrix=steps_r_matrix,
                    co_guess=co_guess,
                    symplectify=False,
                    matrix_responsiveness_tol=matrix_responsiveness_tol,
                    matrix_stability_tol=None,
                    start=start, end=end,
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y, r_sigma=r_sigma,
                    delta0=None, zeta0=None, W_matrix=None, R_matrix=None,
                    delta_disp=None,
                    compute_R_element_by_element=True,
                    only_markers=only_markers,
                    )
        else:
            RR = twiss_res._data['R_matrix']
            RR_ebe = twiss_res._data['R_matrix_ebe']

        eneloss_damp_res = _compute_eneloss_and_damping_rates(
                particle_on_co=twiss_res.particle_on_co, R_matrix=RR,
                W_matrix=twiss_res.W_matrix,
                px_co=twiss_res.px, py_co=twiss_res.py,
                ptau_co=twiss_res.ptau, T_rev0=twiss_res.T_rev0,
                line=line, radiation_method=radiation_method)
        twiss_res._data.update(eneloss_damp_res)

        # Equilibrium emittances
        if radiation_method == 'kick_as_co':
            eq_emitts = _compute_equilibrium_emittance_kick_as_co(
                        twiss_res.kin_px, twiss_res.kin_py, twiss_res.ptau,
                        twiss_res.W_matrix,
                        line, radiation_method,
                        eneloss_damp_res['damping_constants_turns'])
            twiss_res._data.update(eq_emitts)
        elif radiation_method == 'full':
            eq_emitts = _compute_equilibrium_emittance_full(
                        kin_px_co=twiss_res.kin_px, kin_py_co=twiss_res.kin_py,
                        ptau_co=twiss_res.ptau, R_matrix_ebe=RR_ebe,
                        line=line, radiation_method=radiation_method)
            twiss_res._data.update(eq_emitts)

    if method == '4d' and 'muzeta' in twiss_res._data:
        twiss_res.muzeta[:] = 0
        if 'qs' in twiss_res._data:
            twiss_res._data['qs'] = 0

    if values_at_element_exit:
        raise NotImplementedError
        # Untested
        name_exit = twiss_res.name[:-1]
        #twiss_res = twiss_res.rows[1:]
        twiss_res = twiss_res._select_rows(slice(1,None,None))
        twiss_res['name'][:] = name_exit
        twiss_res._data['values_at'] = 'exit'
    else:
        twiss_res._data['values_at'] = 'entry'

    if strengths or radiation_integrals:
        _add_strengths_to_twiss_res(twiss_res, line)

    if radiation_integrals:
        twiss_res._compute_radiation_integrals(add_to_tw=True)

    twiss_res._data['method'] = method
    twiss_res._data['radiation_method'] = radiation_method
    twiss_res._data['reference_frame'] = 'proper'
    twiss_res._data['line_config'] = dict(line.config.copy())

    if reverse:
        twiss_res = twiss_res.reverse()

    # twiss_res.mux += init.mux - twiss_res.mux[0]
    # twiss_res.muy += init.muy - twiss_res.muy[0]
    # twiss_res.muzeta += init.muzeta - twiss_res.muzeta[0]
    # twiss_res.dzeta += init.dzeta - twiss_res.dzeta[0]

    if not periodic and not only_orbit:
        # Start phase advance with provided init
        if ((twiss_res.orientation == 'forward' and not reverse)
                or (twiss_res.orientation == 'backward' and reverse)):
            twiss_res.muzeta += init.muzeta - twiss_res.muzeta[0]
            if 'dzeta' in twiss_res._data:
                twiss_res.dzeta += init.dzeta - twiss_res.dzeta[0]
            if 'mux' in twiss_res._data:
                twiss_res.mux += init.mux - twiss_res.mux[0]
                twiss_res.muy += init.muy - twiss_res.muy[0]
        elif ((twiss_res.orientation == 'forward' and reverse)
            or (twiss_res.orientation == 'backward' and not reverse)):
            twiss_res.muzeta += init.muzeta - twiss_res.muzeta[-1]
            if 'dzeta' in twiss_res._data:
                twiss_res.dzeta += init.dzeta - twiss_res.dzeta[-1]
            if 'mux' in twiss_res._data:
                twiss_res.mux += init.mux - twiss_res.mux[-1]
                twiss_res.muy += init.muy - twiss_res.muy[-1]

    if search_for_t_rev:
        twiss_res._data['T_rev'] = twiss_res.T_rev0 - (
            twiss_res.zeta[-1] - twiss_res.zeta[0])/(twiss_res.beta0*clight)

    if num_turns > 1:

        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        kwargs.pop('num_turns')
        kwargs.pop('init')
        kwargs.pop('start')
        kwargs.pop('end')

        tw_mt = _multiturn_twiss(tw0=twiss_res, num_turns=num_turns,
                                 kwargs=kwargs)
        tw_mt._data['_tw0'] = twiss_res
        twiss_res = tw_mt

    if at_elements is not None:
        twiss_res = twiss_res.rows[at_elements]

    twiss_res['periodic'] = periodic
    twiss_res['completed_init'] = completed_init

    return _add_action_in_res(twiss_res, input_kwargs)

def _twiss_open(line, init,
                      start, end,
                      nemitt_x, nemitt_y, r_sigma,
                      delta_disp, zeta_disp,
                      use_full_inverse,
                      hide_thin_groups=False,
                      only_markers=False,
                      only_orbit=False,
                      compute_lattice_functions=True,
                      _continue_if_lost=False,
                      _keep_tracking_data=False,
                      _keep_initial_particles=False,
                      _initial_particles=None,
                      _ebe_monitor=None):

    if init.reference_frame == 'reverse':
        init = init.reverse()

    particle_on_co = init.particle_on_co
    W_matrix = init.W_matrix

    if start is not None and end is None:
        raise ValueError('end must be specified if start is not None')

    if end is not None and start is None:
        raise ValueError('start must be specified if end is not None')

    if start is None:
        start = 0

    if isinstance(start, str):
        start = line._element_names_unique.index(start)
    if isinstance(end, str):
        if end == '_end_point':
            end = len(line._element_names_unique) - 1
        else:
            end = line._element_names_unique.index(end)

    if init.element_name == line._element_names_unique[start]:
        twiss_orientation = 'forward'
    elif init.element_name == '_end_point' and end == len(line._element_names_unique) - 1:
        twiss_orientation = 'backward'
    elif end is not None and init.element_name == line._element_names_unique[end]:
        twiss_orientation = 'backward'
    else:
        raise ValueError(
            '`init` must be given at the start or end of the specified element range.')

    ctx2np = line._context.nparray_from_context_array

    gemitt_x = nemitt_x/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    gemitt_y = nemitt_y/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    scale_transverse_x = np.sqrt(gemitt_x)*r_sigma
    scale_transverse_y = np.sqrt(gemitt_y)*r_sigma
    scale_longitudinal = delta_disp
    scale_eigen = min(scale_transverse_x, scale_transverse_y, scale_longitudinal)

    context = line._context
    if _initial_particles is not None: # used in match
        part_for_twiss = _initial_particles.copy()
    else:
        import xpart
        part_for_twiss = xpart.build_particles(_context=context,
            particle_ref=particle_on_co, mode='shift',
            include_collective=True,
            x     = [0] + list(W_matrix[0, :] * -scale_eigen) + list(W_matrix[0, :] * scale_eigen),
            px    = [0] + list(W_matrix[1, :] * -scale_eigen) + list(W_matrix[1, :] * scale_eigen),
            y     = [0] + list(W_matrix[2, :] * -scale_eigen) + list(W_matrix[2, :] * scale_eigen),
            py    = [0] + list(W_matrix[3, :] * -scale_eigen) + list(W_matrix[3, :] * scale_eigen),
            zeta  = [0] + list(W_matrix[4, :] * -scale_eigen) + list(W_matrix[4, :] * scale_eigen),
            pzeta = [0] + list(W_matrix[5, :] * -scale_eigen) + list(W_matrix[5, :] * scale_eigen),
            )

        if twiss_orientation == 'forward':
            part_for_twiss.at_element = start
            part_for_twiss.s = line.tracker._tracker_data_base.element_s_locations[start]
        elif twiss_orientation == 'backward':
            part_for_twiss.at_element = end + 1 # to include the last element
            part_for_twiss.s = line.tracker._tracker_data_base.element_s_locations[end]
        else:
            raise ValueError('Invalid twiss_orientation')

    part_for_twiss.at_turn = AT_TURN_FOR_TWISS # To avoid writing in monitors

    if _keep_initial_particles:
        part_for_twiss0 = part_for_twiss.copy()

    if _ebe_monitor is not None:
        _monitor = _ebe_monitor
    elif hasattr(line.tracker._tracker_data_base, '_reusable_ebe_monitor_for_twiss'):
        _monitor = line.tracker._tracker_data_base._reusable_ebe_monitor_for_twiss
    else:
        _monitor = 'ONE_TURN_EBE'

    if end is None:
        ele_stop_track = None
    else:
        ele_stop_track = end + 1 # to include the last element

    line.track(part_for_twiss, turn_by_turn_monitor=_monitor,
                ele_start=start,
                ele_stop=ele_stop_track,
                backtrack=(twiss_orientation == 'backward'))

    # We keep the monitor to speed up future calls (attached to tracker data
    # so that it is trashed if number of elements changes)
    line.tracker._tracker_data_base._reusable_ebe_monitor_for_twiss = line.record_last_track

    if not _continue_if_lost:
        assert np.all(ctx2np(part_for_twiss.state) == 1), (
            'Some test particles were lost during twiss! '
          + f'(state {np.unique(ctx2np(part_for_twiss.state))}, '
          + f'at element {np.unique(ctx2np(part_for_twiss.at_element))})')

    if twiss_orientation == 'forward':
        i_start = start
        i_stop = part_for_twiss._xobject.at_element[0] + (
                (part_for_twiss._xobject.at_turn[0] - AT_TURN_FOR_TWISS)
                * len(line._element_names_unique))
    elif twiss_orientation == 'backward':
        i_start = start
        if ele_stop_track is not None:
            i_stop = ele_stop_track
        else:
            i_stop = len(line._element_names_unique) - 1

    recorded_state = line.record_last_track.state[:, i_start:i_stop+1].copy()
    if not _continue_if_lost:
        assert np.all(recorded_state == 1), (
             'Some test particles were lost during twiss! '
          + f'(state {np.unique(recorded_state)}, '
          + f'at element {np.unique(line.record_last_track.at_element[:, i_start:i_stop+1].copy())})')

    x_co = line.record_last_track.x[0, i_start:i_stop+1].copy()
    y_co = line.record_last_track.y[0, i_start:i_stop+1].copy()
    px_co = line.record_last_track.px[0, i_start:i_stop+1].copy()
    py_co = line.record_last_track.py[0, i_start:i_stop+1].copy()
    zeta_co = line.record_last_track.zeta[0, i_start:i_stop+1].copy()
    delta_co = np.array(line.record_last_track.delta[0, i_start:i_stop+1].copy())
    ptau_co = np.array(line.record_last_track.ptau[0, i_start:i_stop+1].copy())
    s_co = line.record_last_track.s[0, i_start:i_stop+1].copy()
    kin_px_co = line.record_last_track.kin_px[0, i_start:i_stop+1].copy()
    kin_py_co = line.record_last_track.kin_py[0, i_start:i_stop+1].copy()
    kin_ps_co = line.record_last_track.kin_ps[0, i_start:i_stop+1].copy()
    kin_xprime_co = line.record_last_track.kin_xprime[0, i_start:i_stop+1].copy()
    kin_yprime_co = line.record_last_track.kin_yprime[0, i_start:i_stop+1].copy()

    Ws = np.zeros(shape=(len(s_co), 6, 6), dtype=np.float64)
    Ws[:, 0, :] = 0.5 * (line.record_last_track.x[1:7, i_start:i_stop+1] - x_co).T / scale_eigen
    Ws[:, 1, :] = 0.5 * (line.record_last_track.px[1:7, i_start:i_stop+1] - px_co).T / scale_eigen
    Ws[:, 2, :] = 0.5 * (line.record_last_track.y[1:7, i_start:i_stop+1] - y_co).T / scale_eigen
    Ws[:, 3, :] = 0.5 * (line.record_last_track.py[1:7, i_start:i_stop+1] - py_co).T / scale_eigen
    Ws[:, 4, :] = 0.5 * (line.record_last_track.zeta[1:7, i_start:i_stop+1] - zeta_co).T / scale_eigen
    Ws[:, 5, :] = 0.5 * (line.record_last_track.ptau[1:7, i_start:i_stop+1] - ptau_co).T / particle_on_co._xobject.beta0[0] / scale_eigen

    Ws[:, 0, :] -= 0.5 * (line.record_last_track.x[7:13, i_start:i_stop+1] - x_co).T / scale_eigen
    Ws[:, 1, :] -= 0.5 * (line.record_last_track.px[7:13, i_start:i_stop+1] - px_co).T / scale_eigen
    Ws[:, 2, :] -= 0.5 * (line.record_last_track.y[7:13, i_start:i_stop+1] - y_co).T / scale_eigen
    Ws[:, 3, :] -= 0.5 * (line.record_last_track.py[7:13, i_start:i_stop+1] - py_co).T / scale_eigen
    Ws[:, 4, :] -= 0.5 * (line.record_last_track.zeta[7:13, i_start:i_stop+1] - zeta_co).T / scale_eigen
    Ws[:, 5, :] -= 0.5 * (line.record_last_track.ptau[7:13, i_start:i_stop+1] - ptau_co).T / particle_on_co._xobject.beta0[0] / scale_eigen

    name_co = np.array(line._element_names_unique[i_start:i_stop] + ('_end_point',))
    name_co_env = np.array(line.element_names[i_start:i_stop] + ('_end_point',))

    if only_markers:
        raise NotImplementedError('only_markers not supported anymore')

    twiss_res_element_by_element = {}

    twiss_res_element_by_element.update({
        'name': name_co,
        's': s_co,
        'x': x_co,
        'px': px_co,
        'y': y_co,
        'py': py_co,
        'zeta': zeta_co,
        'delta': delta_co,
        'ptau': ptau_co,
        'W_matrix': Ws,
        'kin_px': kin_px_co,
        'kin_py': kin_py_co,
        'kin_ps': kin_ps_co,
        'kin_xprime': kin_xprime_co,
        'kin_yprime': kin_yprime_co,
        'name_env': name_co_env,
    })

    if not only_orbit and compute_lattice_functions:
        lattice_functions, i_replace = _compute_lattice_functions(Ws, use_full_inverse, s_co)
        twiss_res_element_by_element.update(lattice_functions)

    extra_data = {}
    extra_data['only_markers'] = only_markers
    if _keep_tracking_data:
        extra_data['tracking_data'] = line.record_last_track.copy()

    if _keep_initial_particles:
        extra_data['_initial_particles'] = part_for_twiss0.copy()

    if hide_thin_groups:
        _vars_hide_changes = [
        'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau',
        'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy',
        'betx1', 'bety1', 'betx2', 'bety2',
        'dx', 'dpx', 'dy', 'dpy',
        ]

        for key in _vars_hide_changes:
            if key in twiss_res_element_by_element:
                twiss_res_element_by_element[key][i_replace] = np.nan

    twiss_res_element_by_element['name'] = np.array(twiss_res_element_by_element['name'])

    twiss_res = TwissTable(data=twiss_res_element_by_element)
    twiss_res._data.update(extra_data)

    twiss_res._data['particle_on_co'] = particle_on_co.copy(_context=xo.context_default)

    circumference = line.tracker._tracker_data_base.line_length
    twiss_res._data['circumference'] = circumference
    twiss_res._data['orientation'] = twiss_orientation

    return twiss_res


def _compute_lattice_functions(Ws, use_full_inverse, s_co):

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
        (betx, alfx, gamx, bety, alfy, gamy, bety1, betx2
                    )= _extract_twiss_parameters_with_inverse(Ws)
    else:
        betx = Ws[:, 0, 0]**2 + Ws[:, 0, 1]**2
        bety = Ws[:, 2, 2]**2 + Ws[:, 2, 3]**2

        gamx = Ws[:, 1, 0]**2 + Ws[:, 1, 1]**2
        gamy = Ws[:, 3, 2]**2 + Ws[:, 3, 3]**2

        alfx = -Ws[:, 0, 0] * Ws[:, 1, 0] - Ws[:, 0, 1] * Ws[:, 1, 1]
        alfy = -Ws[:, 2, 2] * Ws[:, 3, 2] - Ws[:, 2, 3] * Ws[:, 3, 3]

        bety1 = Ws[:, 2, 0]**2 + Ws[:, 2, 1]**2
        betx2 = Ws[:, 0, 2]**2 + Ws[:, 0, 3]**2

        # Untested:
        # alfx2 = -Ws[:, 0, 2] * Ws[:, 1, 2] - Ws[:, 0, 3] * Ws[:, 1, 3]
        # alfy1 = -Ws[:, 2, 0] * Ws[:, 3, 0] - Ws[:, 2, 1] * Ws[:, 3, 1]
        # gamx2 = Ws[:, 1, 2]**2 + Ws[:, 1, 3]**2
        # gamy1 = Ws[:, 3, 0]**2 + Ws[:, 3, 1]**2

    betx1 = betx
    bety2 = bety


    temp_phix = phix.copy()
    temp_phiy = phiy.copy()
    temp_phix[i_replace] = temp_phix[i_replace_with]
    temp_phiy[i_replace] = temp_phiy[i_replace_with]

    mux = np.unwrap(temp_phix) / 2 / np.pi
    muy = np.unwrap(temp_phiy) / 2  /np.pi
    muzeta = np.unwrap(phizeta) / 2 / np.pi

    # Crab dispersion
    dx_zeta = (Ws[:, 0, 4] - Ws[:, 0, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
               Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])
    dpx_zeta = (Ws[:, 1, 4] - Ws[:, 1, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
                Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])
    dy_zeta = (Ws[:, 2, 4] - Ws[:, 2, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
                Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])
    dpy_zeta = (Ws[:, 3, 4] - Ws[:, 3, 5] * Ws[:, 5, 4] / Ws[:, 5, 5]) / (
                Ws[:, 4, 4] - Ws[:, 4, 5] * Ws[:, 5, 4] / Ws[:, 5, 5])

    # Dispersion
    dx_pzeta = (Ws[:, 0, 5] - Ws[:, 0, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])
    dpx_pzeta = (Ws[:, 1, 5] - Ws[:, 1, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])
    dy_pzeta = (Ws[:, 2, 5] - Ws[:, 2, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])
    dpy_pzeta = (Ws[:, 3, 5] - Ws[:, 3, 4] * Ws[:, 4, 5] / Ws[:, 4, 4]) / (
                Ws[:, 5, 5] - Ws[:, 5, 4] * Ws[:, 4, 5] / Ws[:, 4, 4])

    mux = mux - mux[0]
    muy = muy - muy[0]
    muzeta = muzeta - muzeta[0]

    res = {
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'gamx': gamx,
        'gamy': gamy,
        'dx': dx_pzeta,
        'dpx': dpx_pzeta,
        'dy': dy_pzeta,
        'dpy': dpy_pzeta,
        'dx_zeta': dx_zeta,
        'dpx_zeta': dpx_zeta,
        'dy_zeta': dy_zeta,
        'dpy_zeta': dpy_zeta,
        'betx1': betx1,
        'bety1': bety1,
        'betx2': betx2,
        'bety2': bety2,
        'mux': mux,
        'muy': muy,
        'muzeta': muzeta,
        'nux': nux,
        'nuy': nuy,
        'nuzeta': nuzeta,
        'W_matrix': Ws,
        'phix': phix,
        'phiy': phiy,
        'phizeta': phizeta,
    }
    return res, i_replace


def _compute_global_quantities(line, twiss_res):

        s_vect = twiss_res['s']
        circumference = line.tracker._tracker_data_base.line_length
        part_on_co = twiss_res['particle_on_co']
        W_matrix = twiss_res['W_matrix']

        beta0 = part_on_co._xobject.beta0[0]
        T_rev0 = circumference/clight/beta0
        bets0 = W_matrix[0, 4, 4]**2 + W_matrix[0, 4, 5]**2
        if 'slip_factor' in twiss_res:
            if twiss_res['slip_factor'] < 0: # below transition
                bets0 = -bets0
        else:
            bets0 = 0

        twiss_res._data.update({
            'bets0': bets0,
            'circumference': circumference, 'T_rev0': T_rev0,
            'particle_on_co':part_on_co.copy(_context=xo.context_default),
            'gamma0': part_on_co._xobject.gamma0[0],
            'beta0': part_on_co._xobject.beta0[0],
            'p0c': part_on_co._xobject.p0c[0],
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
            # Proceedings of the 2018 CERN–Accelerator–School
            cmin_arr = (2 * np.sqrt(c_r1*c_r2) *
                        np.abs(np.mod(mux[-1], 1) - np.mod(muy[-1], 1))
                        /(1 + c_r1 * c_r2))
            c_minus = trapz(cmin_arr, s_vect)/(circumference)

            c_minus_cplx = c_minus * np.exp(1j * c_phi1)
            c_minus_re = np.real(c_minus_cplx)
            c_minus_im = np.imag(c_minus_cplx)

            qs = np.abs(twiss_res['muzeta'][-1])

            # Scalars
            twiss_res._data.update({
                'qx': mux[-1], 'qy': muy[-1], 'qs': qs,
                'c_minus': c_minus,
                'c_minus_re_0': c_minus_re[0], 'c_minus_im_0': c_minus_im[0],
            })

            # Coupling columns
            twiss_res['c_minus_re'] = c_minus_re
            twiss_res['c_minus_im'] = c_minus_im
            twiss_res['c_r1'] = c_r1
            twiss_res['c_r2'] = c_r2
            twiss_res['c_phi1'] = c_phi1
            twiss_res['c_phi2'] = c_phi2

def _compute_chromatic_functions(line, init, delta_chrom, steps_r_matrix,
                    matrix_responsiveness_tol, matrix_stability_tol, symplectify,
                    method='6d', use_full_inverse=False,
                    nemitt_x=None, nemitt_y=None,
                    r_sigma=1e-3, delta_disp=1e-3, zeta_disp=1e-3,
                    on_momentum_twiss_res=None,
                    start=None, end=None, num_turns=None,
                    hide_thin_groups=False,
                    only_markers=False,
                    periodic=False,
                    periodic_mode=None,
                    include_collective=False):

    if only_markers:
        raise NotImplementedError('only_markers not supported anymore')

    tw_chrom_res = []
    for dd in [-delta_chrom, delta_chrom]:
        tw_init_chrom = init.copy()

        if periodic:
            import xpart
            part_guess = xpart.build_particles(
                _context=line._context,
                x_norm=0,
                zeta=tw_init_chrom.zeta,
                delta=tw_init_chrom.delta+ dd,
                particle_on_co=on_momentum_twiss_res.particle_on_co.copy(),
                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                W_matrix=tw_init_chrom.W_matrix,
                include_collective=include_collective)
            part_chrom = line.find_closed_orbit(delta0=dd, co_guess=part_guess,
                                    start=start, end=end, num_turns=num_turns,
                                    symmetrize=(periodic_mode == 'periodic_symmetric'),
                                    include_collective=include_collective,
                                    )
            tw_init_chrom.particle_on_co = part_chrom
            RR_chrom = line.compute_one_turn_matrix_finite_differences(
                                        particle_on_co=tw_init_chrom.particle_on_co.copy(),
                                        start=start, end=end, num_turns=num_turns,
                                        steps_r_matrix=steps_r_matrix,
                                        symmetrize=(periodic_mode == 'periodic_symmetric'),
                                        include_collective=include_collective,
                                        )['R_matrix']
            (WW_chrom, _, _, _) = lnf.compute_linear_normal_form(RR_chrom,
                                    only_4d_block=True,
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
                r_sigma=r_sigma,
                delta_disp=delta_disp,
                zeta_disp=zeta_disp,
                use_full_inverse=use_full_inverse,
                hide_thin_groups=hide_thin_groups,
                only_markers=only_markers,
                _continue_if_lost=False,
                _keep_tracking_data=False,
                _keep_initial_particles=False,
                _initial_particles=None,
                _ebe_monitor=None))

    dmux = (tw_chrom_res[1].mux - tw_chrom_res[0].mux)/(2*delta_chrom)
    dmuy = (tw_chrom_res[1].muy - tw_chrom_res[0].muy)/(2*delta_chrom)

    dbetx = (tw_chrom_res[1].betx - tw_chrom_res[0].betx)/(2*delta_chrom)
    dbety = (tw_chrom_res[1].bety - tw_chrom_res[0].bety)/(2*delta_chrom)
    dalfx = (tw_chrom_res[1].alfx - tw_chrom_res[0].alfx)/(2*delta_chrom)
    dalfy = (tw_chrom_res[1].alfy - tw_chrom_res[0].alfy)/(2*delta_chrom)
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

    dzeta = (tw_chrom_res[1].zeta - tw_chrom_res[0].zeta)/(2*delta_chrom)
    dzeta -= dzeta[0]
    dzeta = np.array(dzeta)

    slip_factor = -dzeta[-1] / tw_chrom_res[0].circumference
    momentum_compaction_factor = (slip_factor
                        + 1/tw_chrom_res[0].particle_on_co._xobject.gamma0[0]**2)

    cols_chrom = {'dmux': dmux, 'dmuy': dmuy, 'dzeta': dzeta,
                  'bx_chrom': bx_chrom, 'by_chrom': by_chrom,
                  'ax_chrom': ax_chrom, 'ay_chrom': ay_chrom,
                  'wx_chrom': wx_chrom, 'wy_chrom': wy_chrom,
                  }
    scalars_chrom = {'dqx': dqx, 'dqy': dqy, 'slip_factor': slip_factor,
                     'momentum_compaction_factor': momentum_compaction_factor,
                     }

    if on_momentum_twiss_res is not None:
        mux = on_momentum_twiss_res.mux
        muy = on_momentum_twiss_res.muy
        x = on_momentum_twiss_res.x
        px = on_momentum_twiss_res.px
        y = on_momentum_twiss_res.y
        py = on_momentum_twiss_res.py
        ddqx = (tw_chrom_res[1].mux[-1] - 2 * mux[-1] + tw_chrom_res[0].mux[-1]
                ) / delta_chrom**2
        ddqy = (tw_chrom_res[1].muy[-1] - 2 * muy[-1] + tw_chrom_res[0].muy[-1]
                ) / delta_chrom**2
        ddx = (tw_chrom_res[1].x - 2 * x + tw_chrom_res[0].x) / delta_chrom**2
        ddpx = (tw_chrom_res[1].px - 2 * px + tw_chrom_res[0].px) / delta_chrom**2
        ddy = (tw_chrom_res[1].y - 2 * y + tw_chrom_res[0].y) / delta_chrom**2
        ddpy = (tw_chrom_res[1].py - 2 * py + tw_chrom_res[0].py) / delta_chrom**2

        cols_chrom.update({'ddx': ddx, 'ddpx': ddpx,
                           'ddy': ddy, 'ddpy': ddpy})
        scalars_chrom.update({'ddqx': ddqx, 'ddqy': ddqy})

    return cols_chrom, scalars_chrom


def _compute_eneloss_and_damping_rates(particle_on_co, R_matrix,
                                       px_co, py_co, ptau_co, W_matrix,
                                       T_rev0, line, radiation_method):
    diff_ptau = np.diff(ptau_co)
    mask_loss = diff_ptau < 0
    eloss_turn = -sum(diff_ptau[mask_loss]) * particle_on_co._xobject.p0c[0]

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

    # https://cds.cern.ch/record/175614 , Eq. 4.24
    partition_numbers = (
        damping_constants_turns * 2
        / (-np.sum(diff_ptau[mask_loss] / (1 + ptau_co[:-1][mask_loss]))))

    eneloss_damp_res = {
        'eneloss_turn': eloss_turn,
        'damping_constants_turns': damping_constants_turns,
        'damping_constants_s':damping_constants_s,
        'partition_numbers': partition_numbers,
    }

    return eneloss_damp_res

def _extract_sr_distribution_properties(line, px_co, py_co, ptau_co):


    radiation_flag = line.attr['radiation_flag']
    if np.any(radiation_flag > 1):
        raise ValueError('Incompatible radiation flag')

    hxl = line.attr['angle_rad'] * np.cos(line.attr['rot_s_rad'])
    hyl = line.attr['angle_rad'] * np.sin(line.attr['rot_s_rad'])
    dl = line.attr['length'] * (radiation_flag == 1)

    mask = (dl != 0)
    hx = np.zeros(shape=(len(dl),), dtype=np.float64)
    hy = np.zeros(shape=(len(dl),), dtype=np.float64)
    hx[mask] = (np.diff(px_co)[mask] + hxl[mask] * (1 + ptau_co[:-1][mask])) / dl[mask]
    hy[mask] = (np.diff(py_co)[mask] + hyl[mask] * (1 + ptau_co[:-1][mask])) / dl[mask]
    # TODO: remove also term due to weak focusing
    hh = np.sqrt(hx**2 + hy**2)

    mass0 = line.particle_ref.mass0
    q0 = line.particle_ref.q0
    gamma0 = line.particle_ref._xobject.gamma0[0]
    beta0 = line.particle_ref._xobject.beta0[0]

    gamma = gamma0 * (1 + beta0 * ptau_co)[:-1]

    mass0_kg = mass0 / clight**2 * qe
    q_coul = q0 * qe
    B_T = hh * mass0_kg * clight * gamma0 / np.abs(q_coul)
    r0_m = q_coul**2/(4*np.pi*epsilon_0*mass0_kg*clight**2)
    E_crit_J = 3 * np.abs(q_coul) * hbar * gamma**2 * B_T / (2 * mass0_kg)
    n_dot = 60 / 72 * np.sqrt(3) * r0_m * clight * np.abs(q_coul) * B_T / hbar
    E_sq_ave_J = 11 / 27 * E_crit_J**2
    E_ave_J = 8 * np.sqrt(3) / 45 * E_crit_J
    E0_J = mass0_kg * clight**2 * gamma0

    n_dot_delta_kick_sq_ave = n_dot * E_sq_ave_J / E0_J**2

    res = {
        'B_T': B_T,
        'E_crit_J': E_crit_J, 'n_dot': n_dot,
        'E_sq_ave_J': E_sq_ave_J, 'E_ave_J': E_ave_J,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
        'dl_radiation': dl,
    }

    return res

def _compute_equilibrium_emittance_kick_as_co(kin_px_co, kin_py_co, ptau_co, W_matrix,
                                  line, radiation_method,
                                  damping_constants_turns):

    assert radiation_method == 'kick_as_co'

    sr_distrib_properties = _extract_sr_distribution_properties(
                                line, kin_px_co, kin_py_co, ptau_co)
    beta0 = line.particle_ref._xobject.beta0[0]
    gamma0 = line.particle_ref._xobject.gamma0[0]

    n_dot_delta_kick_sq_ave = sr_distrib_properties['n_dot_delta_kick_sq_ave']
    dl = sr_distrib_properties['dl_radiation']

    px_left = kin_px_co[:-1]
    px_right = kin_px_co[1:]
    py_left = kin_py_co[:-1]
    py_right = kin_py_co[1:]
    one_pl_del_left = (1 + ptau_co[:-1]) # Assuming ultrarelativistic
    one_pl_del_right = (1 + ptau_co[1:]) # Assuming ultrarelativistic
    W_left = W_matrix[:-1, :, :]
    W_right = W_matrix[1:, :, :]

    a11_left = np.squeeze(W_left[:, 0, 0])
    a13_left = np.squeeze(W_left[:, 2, 0])
    a15_left = np.squeeze(W_left[:, 4, 0])
    b11_left = np.squeeze(W_left[:, 0, 1])
    b13_left = np.squeeze(W_left[:, 2, 1])
    b15_left = np.squeeze(W_left[:, 4, 1])

    a11_right = np.squeeze(W_right[:, 0, 0])
    a13_right = np.squeeze(W_right[:, 2, 0])
    a15_right = np.squeeze(W_right[:, 4, 0])
    b11_right = np.squeeze(W_right[:, 0, 1])
    b13_right = np.squeeze(W_right[:, 2, 1])
    b15_right = np.squeeze(W_right[:, 4, 1])

    a21_left = np.squeeze(W_left[:, 0, 2])
    a23_left = np.squeeze(W_left[:, 2, 2])
    a25_left = np.squeeze(W_left[:, 4, 2])
    b21_left = np.squeeze(W_left[:, 0, 3])
    b23_left = np.squeeze(W_left[:, 2, 3])
    b25_left = np.squeeze(W_left[:, 4, 3])

    a21_right = np.squeeze(W_right[:, 0, 2])
    a23_right = np.squeeze(W_right[:, 2, 2])
    a25_right = np.squeeze(W_right[:, 4, 2])
    b21_right = np.squeeze(W_right[:, 0, 3])
    b23_right = np.squeeze(W_right[:, 2, 3])
    b25_right = np.squeeze(W_right[:, 4, 3])

    a31_left = np.squeeze(W_left[:, 0, 4])
    a33_left = np.squeeze(W_left[:, 2, 4])
    a35_left = np.squeeze(W_left[:, 4, 4])
    b31_left = np.squeeze(W_left[:, 0, 5])
    b33_left = np.squeeze(W_left[:, 2, 5])
    b35_left = np.squeeze(W_left[:, 4, 5])

    a31_right = np.squeeze(W_right[:, 0, 4])
    a33_right = np.squeeze(W_right[:, 2, 4])
    a35_right = np.squeeze(W_right[:, 4, 4])
    b31_right = np.squeeze(W_right[:, 0, 5])
    b33_right = np.squeeze(W_right[:, 2, 5])
    b35_right = np.squeeze(W_right[:, 4, 5])

    Kx_left = (a11_left * px_left + a13_left * py_left) / one_pl_del_left + a15_left
    Kpx_left = (b11_left * px_left + b13_left * py_left) / one_pl_del_left + b15_left
    Ky_left = (a21_left * px_left + a23_left * py_left) / one_pl_del_left + a25_left
    Kpy_left = (b21_left * px_left + b23_left * py_left) / one_pl_del_left + b25_left
    Kz_left = (a31_left * px_left + a33_left * py_left) / one_pl_del_left + a35_left
    Kpz_left = (b31_left * px_left + b33_left * py_left) / one_pl_del_left + b35_left

    Kx_right = (a11_right * px_right + a13_right * py_right) / one_pl_del_right + a15_right
    Kpx_right = (b11_right * px_right + b13_right * py_right) / one_pl_del_right + b15_right
    Ky_right = (a21_right * px_right + a23_right * py_right) / one_pl_del_right + a25_right
    Kpy_right = (b21_right * px_right + b23_right * py_right) / one_pl_del_right + b25_right
    Kz_right = (a31_right * px_right + a33_right * py_right) / one_pl_del_right + a35_right
    Kpz_right = (b31_right * px_right + b33_right * py_right) / one_pl_del_right + b35_right

    Kx_sq = 0.5 * (Kx_left**2 + Kx_right**2)
    Kpx_sq = 0.5 * (Kpx_left**2 + Kpx_right**2)
    Ky_sq = 0.5 * (Ky_left**2 + Ky_right**2)
    Kpy_sq = 0.5 * (Kpy_left**2 + Kpy_right**2)
    Kz_sq = 0.5 * (Kz_left**2 + Kz_right**2)
    Kpz_sq = 0.5 * (Kpz_left**2 + Kpz_right**2)

    eq_gemitt_x = 1 / (4 * clight * damping_constants_turns[0]) * np.sum(
                        (Kx_sq + Kpx_sq) * n_dot_delta_kick_sq_ave * dl)
    eq_gemitt_y = 1 / (4 * clight * damping_constants_turns[1]) * np.sum(
                        (Ky_sq + Kpy_sq) * n_dot_delta_kick_sq_ave * dl)
    eq_gemitt_zeta = 1 / (4 * clight * damping_constants_turns[2]) * np.sum(
                        (Kz_sq + Kpz_sq) * n_dot_delta_kick_sq_ave * dl)

    eq_nemitt_x = float(eq_gemitt_x * (beta0 * gamma0))
    eq_nemitt_y = float(eq_gemitt_y * (beta0 * gamma0))
    eq_nemitt_zeta = float(eq_gemitt_zeta * (beta0 * gamma0))

    res = {
        'eq_gemitt_x': eq_gemitt_x,
        'eq_gemitt_y': eq_gemitt_y,
        'eq_gemitt_zeta': eq_gemitt_zeta,
        'eq_nemitt_x': eq_nemitt_x,
        'eq_nemitt_y': eq_nemitt_y,
        'eq_nemitt_zeta': eq_nemitt_zeta,
        'dl_radiation': dl,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
    }

    return res

def _compute_equilibrium_emittance_full(kin_px_co, kin_py_co, ptau_co, R_matrix_ebe,
                                  line, radiation_method):

    sr_distrib_properties = _extract_sr_distribution_properties(
                                line, kin_px_co, kin_py_co, ptau_co)

    n_dot_delta_kick_sq_ave = sr_distrib_properties['n_dot_delta_kick_sq_ave']
    dl = sr_distrib_properties['dl_radiation']

    assert radiation_method == 'full'

    d_delta_sq_ave = n_dot_delta_kick_sq_ave * dl / clight

    # Going to x', y'
    RR_ebe = R_matrix_ebe
    delta = ptau_co # ultrarelativistic approximation

    TT = RR_ebe * 0.
    TT[:, 0, 0] = 1
    TT[:, 1, 1] = (1 - delta)
    TT[:, 1, 5] = -kin_px_co
    TT[:, 2, 2] = 1
    TT[:, 3, 3] = (1 - delta)
    TT[:, 3, 5] = -kin_py_co
    TT[:, 4, 4] = 1
    TT[:, 5, 5] = 1

    TTinv = np.linalg.inv(TT)
    TTinv0 = TTinv.copy()
    for ii in range(6):
        for jj in range(6):
            TTinv0[:, ii, jj] = TTinv[0, ii, jj]

    RR_ebe_hat = TT @ RR_ebe @ TTinv0
    RR = RR_ebe_hat[-1, :, :]

    lnf = xt.linear_normal_form
    WW, _, Rot, lam_eig = lnf.compute_linear_normal_form(RR)
    DSigma = np.zeros_like(RR_ebe_hat)

    # The following is needed if RR is in px, py instead of x', y'
    # DSigma[:-1, 1, 1] = (d_delta_sq_ave * 0.5 * (px_co[:-1]**2 + px_co[1:]**2)
    #                                             / (ptau_co[:-1] + 1)**2)
    # DSigma[:-1, 3, 3] = (d_delta_sq_ave * 0.5 * (py_co[:-1]**2 + py_co[1:]**2)
    #                                             / (ptau_co[:-1] + 1)**2)

    # DSigma[:-1, 1, 5] = (d_delta_sq_ave * 0.5 * (px_co[:-1] + px_co[1:])
    #                                             / (ptau_co[:-1] + 1))
    # DSigma[:-1, 5, 1] = (d_delta_sq_ave * 0.5 * (px_co[:-1] + px_co[1:])
    #                                             / (ptau_co[:-1] + 1))

    # DSigma[:-1, 3, 5] = (d_delta_sq_ave * 0.5 * (py_co[:-1] + py_co[1:])
    #                                              / (ptau_co[:-1] + 1))
    # DSigma[:-1, 5, 3] = (d_delta_sq_ave * 0.5 * (py_co[:-1] + py_co[1:])
    #                                              / (ptau_co[:-1] + 1))

    DSigma[:-1, 5, 5] = d_delta_sq_ave

    RR_ebe_hat_inv = np.linalg.inv(RR_ebe_hat)

    DSigma0 = np.zeros((6, 6))

    n_calc = d_delta_sq_ave.shape[0]
    for ii in range(n_calc):
        if d_delta_sq_ave[ii] > 0:
            DSigma0 += RR_ebe_hat_inv[ii, :, :] @ DSigma[ii, :, :] @ RR_ebe_hat_inv[ii, :, :].T

    CC_split, _, RRR, reig = lnf.compute_linear_normal_form(Rot)
    reig_full = np.zeros_like(Rot, dtype=complex)
    reig_full[0, 0] = reig[0]
    reig_full[1, 1] = reig[0].conjugate()
    reig_full[2, 2] = reig[1]
    reig_full[3, 3] = reig[1].conjugate()
    reig_full[4, 4] = reig[2]
    reig_full[5, 5] = reig[2].conjugate()

    lam_eig_full = np.zeros_like(reig_full, dtype=complex)
    lam_eig_full[0] = lam_eig[0]
    lam_eig_full[1] = lam_eig[0].conjugate()
    lam_eig_full[2] = lam_eig[1]
    lam_eig_full[3] = lam_eig[1].conjugate()
    lam_eig_full[4] = lam_eig[2]
    lam_eig_full[5] = lam_eig[2].conjugate()

    CC = np.zeros_like(CC_split, dtype=complex)
    CC[:, 0] = 0.5*np.sqrt(2)*(CC_split[:, 0] + 1j*CC_split[:, 1])
    CC[:, 1] = 0.5*np.sqrt(2)*(CC_split[:, 0] - 1j*CC_split[:, 1])
    CC[:, 2] = 0.5*np.sqrt(2)*(CC_split[:, 2] + 1j*CC_split[:, 3])
    CC[:, 3] = 0.5*np.sqrt(2)*(CC_split[:, 2] - 1j*CC_split[:, 3])
    CC[:, 4] = 0.5*np.sqrt(2)*(CC_split[:, 4] + 1j*CC_split[:, 5])
    CC[:, 5] = 0.5*np.sqrt(2)*(CC_split[:, 4] - 1j*CC_split[:, 5])

    BB = WW @ CC

    BB_inv = np.linalg.inv(BB)

    EE_norm = (BB_inv @ DSigma0 @ BB_inv.T).real

    eq_gemitt_x = EE_norm[0, 1]/(1 - np.abs(lam_eig[0])**2)
    eq_gemitt_y = EE_norm[2, 3]/(1 - np.abs(lam_eig[1])**2)
    eq_gemitt_zeta = EE_norm[4, 5]/(1 - np.abs(lam_eig[2])**2)

    beta0 = line.particle_ref._xobject.beta0[0]
    gamma0 = line.particle_ref._xobject.gamma0[0]

    eq_nemitt_x = float(eq_gemitt_x * (beta0 * gamma0))
    eq_nemitt_y = float(eq_gemitt_y * (beta0 * gamma0))
    eq_nemitt_zeta = float(eq_gemitt_zeta * (beta0 * gamma0))

    Sigma_norm = np.zeros_like(EE_norm, dtype=complex)
    for ii in range(6):
        for jj in range(6):
            Sigma_norm[ii, jj] = EE_norm[ii, jj]/(1 - lam_eig_full[ii, ii]*lam_eig_full[jj, jj])

    Sigma_at_start = (BB @ Sigma_norm @ BB.T).real

    Sigma = RR_ebe @ Sigma_at_start @ np.transpose(RR_ebe, axes=(0,2,1))

    eq_sigma_tab = _build_sigma_table(Sigma=Sigma, s=None,
        name=np.array(tuple(line._element_names_unique) + ('_end_point',)))

    res = {
        'eq_gemitt_x': eq_gemitt_x,
        'eq_gemitt_y': eq_gemitt_y,
        'eq_gemitt_zeta': eq_gemitt_zeta,
        'eq_nemitt_x': eq_nemitt_x,
        'eq_nemitt_y': eq_nemitt_y,
        'eq_nemitt_zeta': eq_nemitt_zeta,
        'eq_beam_covariance_matrix': eq_sigma_tab,
        'dl_radiation': dl,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
    }

    return res


class ClosedOrbitSearchError(Exception):
    pass

def _find_periodic_solution(line, particle_on_co, particle_ref, method,
                            co_search_settings, continue_on_closed_orbit_error,
                            delta0, zeta0, steps_r_matrix, W_matrix,
                            R_matrix, co_guess,
                            delta_disp, symplectify,
                            matrix_responsiveness_tol,
                            matrix_stability_tol,
                            nemitt_x, nemitt_y, r_sigma,
                            start=None, end=None,
                            num_turns=1,
                            co_search_at=None,
                            search_for_t_rev=False,
                            num_turns_search_t_rev=1,
                            compute_R_element_by_element=False,
                            only_markers=False,
                            only_orbit=False,
                            periodic_mode='periodic',
                            include_collective=False):

    eigenvalues = None
    Rot = None
    RR_ebe = None

    assert periodic_mode in ['periodic', 'periodic_symmetric']

    if start is not None or end is not None:
        assert start is not None and end is not None, (
            'start and end must be both None or both not None')

    if start is not None:
        assert _str_to_index(line, start) <= _str_to_index(line, end)

    if method == '4d' and delta0 is None:
        delta0 = 0

    if periodic_mode == 'periodic_symmetric':
        assert R_matrix is None, 'R_matrix must be None for `periodic_symmetric`'
        assert W_matrix is None, 'W_matrix must be None for `periodic_symmetric`'

    if particle_on_co is not None:
        part_on_co = particle_on_co
    else:
        if search_for_t_rev:
            assert method == '6d', 'search_for_t_rev possible when `method` is "6d"'
        part_on_co = line.find_closed_orbit(
                                co_guess=co_guess,
                                particle_ref=particle_ref,
                                co_search_settings=co_search_settings,
                                continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                                delta0=delta0,
                                zeta0=zeta0,
                                start=start,
                                end=end,
                                num_turns=num_turns,
                                co_search_at=co_search_at,
                                search_for_t_rev=search_for_t_rev,
                                num_turns_search_t_rev=num_turns_search_t_rev,
                                symmetrize=(periodic_mode == 'periodic_symmetric'),
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
            W, _, Rot, eigenvalues = lnf.compute_linear_normal_form(
                        RR, only_4d_block=(method == '4d'),
                        symplectify=symplectify,
                        responsiveness_tol=matrix_responsiveness_tol,
                        stability_tol=matrix_stability_tol)
        else:
            steps_r_matrix['adapted'] = False
            for iter in range(2):
                RR_out = line.compute_one_turn_matrix_finite_differences(
                    steps_r_matrix=steps_r_matrix,
                    particle_on_co=part_on_co,
                    start=start,
                    end=end,
                    num_turns=num_turns,
                    element_by_element=compute_R_element_by_element,
                    only_markers=only_markers,
                    symmetrize=(periodic_mode == 'periodic_symmetric'),
                    include_collective=include_collective
                    )
                RR = RR_out['R_matrix']
                RR_ebe = RR_out['R_matrix_ebe']

                if matrix_responsiveness_tol is not None:
                    lnf._assert_matrix_responsiveness(RR,
                        matrix_responsiveness_tol, only_4d=(method == '4d'))

                W, _, Rot, eigenvalues = lnf.compute_linear_normal_form(
                            RR, only_4d_block=(method == '4d'),
                            symplectify=symplectify,
                            responsiveness_tol=None,
                            stability_tol=None)

                # Estimate beam size (betatron part)
                gemitt_x = nemitt_x/part_on_co._xobject.beta0[0]/part_on_co._xobject.gamma0[0]
                gemitt_y = nemitt_y/part_on_co._xobject.beta0[0]/part_on_co._xobject.gamma0[0]
                betx_at_start = W[0, 0]**2 + W[0, 1]**2
                bety_at_start = W[2, 2]**2 + W[2, 3]**2
                sigma_x_start = np.sqrt(betx_at_start * gemitt_x)
                sigma_y_start = np.sqrt(bety_at_start * gemitt_y)

                if ((steps_r_matrix['dx'] < 0.3 * sigma_x_start)
                    and (steps_r_matrix['dy'] < 0.3 * sigma_y_start)):
                    break # sufficient accuracy
                else:
                    steps_r_matrix['dx'] = 0.01 * sigma_x_start
                    steps_r_matrix['dy'] = 0.01 * sigma_y_start
                    steps_r_matrix['adapted'] = True

    # Check on R matrix
    if RR is not None and matrix_stability_tol is not None:
        lnf._assert_matrix_determinant_within_tol(RR, matrix_stability_tol)
        if method == '4d':
            eigenvals = np.linalg.eigvals(RR[:4, :4])
        else:
            eigenvals = np.linalg.eigvals(RR)
        lnf._assert_matrix_stability(eigenvals, matrix_stability_tol)
        RR_ebe = None


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

    return init, RR, steps_r_matrix, eigenvalues, Rot, RR_ebe

def _handle_loop_around(kwargs):

    kwargs = kwargs.copy()

    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')

    line = kwargs['line']
    reverse = kwargs['reverse']

    ele_name_init = init.element_name

    # if reversed, elements in the line are sorted opposite to the twiss table
    if not reverse:
        assert _str_to_index(line, end) < _str_to_index(line, start), (
            'This function should not have been called')
        if _str_to_index(line, ele_name_init) >= _str_to_index(line, start):
            tw1 = twiss_line(start=start,
                            end='_end_point',
                            init=init, **kwargs)
            twini_2 = tw1.get_twiss_init(at_element='_end_point')
            twini_2.element_name = line._element_names_unique[0]
            tw2 = twiss_line(start=line._element_names_unique[0], end=end,
                                    init=twini_2, **kwargs)
            completed_init = tw1.completed_init
        elif _str_to_index(line, ele_name_init) <= _str_to_index(line, end):
            tw2 = twiss_line(start=line._element_names_unique[0], end=end,
                                init=init, **kwargs)
            twini_1 = tw2.get_twiss_init(at_element=line._element_names_unique[0])
            twini_1.element_name = '_end_point'
            tw1 = twiss_line(start=start, end='_end_point',
                                init=twini_1, **kwargs)
            completed_init = tw2.completed_init
        else:
            raise RuntimeError(
                'Boundary conditions not at start or end of the specified range')
    else: # reversed
        assert _str_to_index(line, end) > _str_to_index(line, start), (
            'This function should not have been called')
        if _str_to_index(line, ele_name_init) <= _str_to_index(line, start):
            tw1 = twiss_line(start=start,
                            end=line._element_names_unique[0],
                            init=init, **kwargs)
            twini_2 = tw1.get_twiss_init(at_element='_end_point')
            twini_2.element_name = line._element_names_unique[-1]
            tw2 = twiss_line(start=line._element_names_unique[-1], end=end,
                                    init=twini_2, **kwargs)
            completed_init = tw1.completed_init
        elif _str_to_index(line, ele_name_init) >= _str_to_index(line, end):
            tw2 = twiss_line(start=line._element_names_unique[-1], end=end,
                                init=init, **kwargs)
            twini_1 = tw2.get_twiss_init(at_element=line._element_names_unique[-1])
            twini_1.element_name = line._element_names_unique[0]
            tw1 = twiss_line(start=start, end=line._element_names_unique[0],
                                init=twini_1, **kwargs)
            completed_init = tw2.completed_init
        else:
            raise RuntimeError(
                'Boundary conditions not at start or end of the specified range')

    tw_res = TwissTable.concatenate([tw1, tw2])

    tw_res.s -= tw_res['s', ele_name_init] - init.s

    tw_res['completed_init'] = completed_init

    if 'mux' in tw_res.keys():
        tw_res.mux -= tw_res['mux', ele_name_init] - init.mux
        tw_res.muy -= tw_res['muy', ele_name_init] - init.muy
        tw_res.muzeta -= tw_res['muzeta', ele_name_init] - init.muzeta

    if 'dzeta' in tw_res.keys():
        tw_res.dzeta -= tw_res['dzeta', ele_name_init] - init.dzeta

    # Not yet supported
    if 'dmux' in tw_res.keys():
        tw_res._data.pop('dmux')
        tw_res._col_names.remove('dmux')
    if 'dmuy' in tw_res.keys():
        tw_res._data.pop('dmuy')
        tw_res._col_names.remove('dmuy')

    tw_res._data['loop_around'] = True

    for kk in ['method', 'radiation_method', 'reference_frame']:
        if tw1[kk] == tw2[kk]:
            tw_res._data[kk] = tw1[kk]
        else:
            tw_res._data[kk] = (tw1[kk], tw2[kk])

    return tw_res

def _handle_init_inside_range(kwargs):

    kwargs = kwargs.copy()
    line = kwargs.pop('line')
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    init = kwargs.pop('init')
    reverse = kwargs.pop('reverse')

    ele_name_init =  init.element_name
    if reverse:
        assert _str_to_index(line, ele_name_init) <= _str_to_index(line, start)
        assert _str_to_index(line, ele_name_init) >= _str_to_index(line, end)
    else:
        assert _str_to_index(line, ele_name_init) >= _str_to_index(line, start)
        assert _str_to_index(line, ele_name_init) <= _str_to_index(line, end)

    tw1 = twiss_line(line, start=start, end=ele_name_init,
                     init=init, reverse=reverse, **kwargs)
    tw2 = twiss_line(line, start=ele_name_init, end=end,
                     init=init, reverse=reverse, **kwargs)

    tw_res = TwissTable.concatenate([tw1, tw2])
    tw_res['completed_init'] = tw1.completed_init

    tw_res.s -= tw_res['s', ele_name_init] - init.s
    tw_res.mux -= tw_res['mux', ele_name_init] - init.mux
    tw_res.muy -= tw_res['muy', ele_name_init] - init.muy
    tw_res.muzeta -= tw_res['muzeta', ele_name_init] - init.muzeta

    if 'dzeta' in tw_res:
        tw_res.dzeta -= tw_res['dzeta', ele_name_init] - init.dzeta

    # Not correctly handled yet
    if 'dmux' in tw_res.keys():
        tw_res._data.pop('dmux')
        tw_res._col_names.remove('dmux')
    if 'dmuy' in tw_res.keys():
        tw_res._data.pop('dmuy')
        tw_res._col_names.remove('dmuy')

    for kk in ['method', 'radiation_method', 'reference_frame']:
        if tw1[kk] == tw2[kk]:
            tw_res._data[kk] = tw1[kk]
        else:
            tw_res._data[kk] = (tw1[kk], tw2[kk])

    return tw_res


def find_closed_orbit_line(line, co_guess=None, particle_ref=None,
                      co_search_settings=None, delta_zeta=0,
                      delta0=None, zeta0=None,
                      start=None, end=None, num_turns=1,
                      co_search_at=None,
                      search_for_t_rev=False,
                      continue_on_closed_orbit_error=False,
                      num_turns_search_t_rev=None,
                      symmetrize=False):

    if search_for_t_rev:
        assert line.particle_ref is not None
        assert co_guess is None, '`co_guess` not supported when `search_for_t_rev` is True'
        assert co_search_settings is None, '`co_search_settings` not supported when `search_for_t_rev` is True'
        assert delta_zeta == 0, '`delta_zeta` not supported when `search_for_t_rev` is True'
        assert delta0 is None, '`delta0` not supported when `search_for_t_rev` is True'
        assert zeta0 is None, '`zeta0` not supported when `search_for_t_rev` is True'
        assert start is None, '`start` not supported when `search_for_t_rev` is True'
        assert end is None, '`end` not supported when `search_for_t_rev` is True'
        assert num_turns == 1, '`num_turns` not supported when `search_for_t_rev` is True'
        assert co_search_at is None, '`co_search_at` not supported when `search_for_t_rev` is True'
        assert continue_on_closed_orbit_error is False, '`continue_on_closed_orbit_error` not supported when `search_for_t_rev` is True'
        assert symmetrize is False, '`symmetrize` not supported when `search_for_t_rev` is True'

        out = _find_closed_orbit_search_t_rev(line, num_turns_search_t_rev)
        return out

    if line.enable_time_dependent_vars:
        raise RuntimeError(
            'Time-dependent vars not supported in closed orbit search')

    if start is not None and end is not None:
        co_search_at = None # needs to be implemented

    if co_search_at is not None:
        assert not symmetrize, 'Symmetrize not supported when `co_search_at` is provided'
        kwargs = locals().copy()
        kwargs.pop('start')
        kwargs.pop('end')
        kwargs.pop('co_search_at')
        p_co_at_ele_co_search = find_closed_orbit_line(
            start=co_search_at, end=co_search_at,
            **kwargs)
        line.track(p_co_at_ele_co_search, ele_start=co_search_at, ele_stop=0)
        return p_co_at_ele_co_search

    if isinstance(start, str):
        start = line._element_names_unique.index(start)

    if isinstance(end, str):
        end = line._element_names_unique.index(end)

    if isinstance(co_guess, dict):
        co_guess = line.build_particles(**co_guess, include_collective=True)

    if co_guess is None:
        if particle_ref is None:
            if line.particle_ref is not None:
                particle_ref = line.particle_ref
            else:
                raise ValueError(
                    "Either `co_guess` or `particle_ref` must be provided")

        co_guess = particle_ref.copy()
        co_guess.x = 0
        co_guess.px = 0
        co_guess.y = 0
        co_guess.py = 0
        co_guess.zeta = 0
        co_guess.delta = 0
        co_guess.s = 0
        co_guess.at_element = (start or 0)
        co_guess.at_turn = 0
    else:
        particle_ref = co_guess

    if co_search_settings is None:
        co_search_settings = {}

    co_search_settings = co_search_settings.copy()
    if 'xtol' not in co_search_settings.keys():
        co_search_settings['xtol'] = 1e-6 # Relative error between calls

    co_guess = co_guess.copy(
                        _context=line._buffer.context)

    for shift_factor in [0, 1.]: # if not found at first attempt we shift slightly the starting point
        if shift_factor>0:
            _print('Warning! Need second attempt on closed orbit search')

        x0=np.array([co_guess._xobject.x[0] + shift_factor * 1e-5,
                    co_guess._xobject.px[0] + shift_factor * 1e-7,
                    co_guess._xobject.y[0] + shift_factor * 1e-5,
                    co_guess._xobject.py[0] + shift_factor * 1e-7,
                    co_guess._xobject.zeta[0] + shift_factor * 1e-4,
                    co_guess._xobject.delta[0] + shift_factor * 1e-5])
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
        if np.all(np.abs(_error_for_co(
                x0, co_guess, line, delta_zeta, delta0, zeta0,
                start=start, end=end,
                num_turns=num_turns,
                symmetrize=symmetrize)) < DEFAULT_CO_SEARCH_TOL):
            res = x0
            fsolve_info = 'taken_guess'
            ier = 1
            break

        opt = xt.match.opt_from_callable(
            lambda p: _error_for_co(p, co_guess, line,
                            delta_zeta, delta0, zeta0, start=start,
                            end=end, num_turns=num_turns,
                            symmetrize=symmetrize),
                x0=x0, steps=[1e-8, 1e-9, 1e-8, 1e-9, 1e-7, 1e-8],
                tar=[0., 0., 0., 0., 0., 0.],
                tols=[1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
        try:
            opt.solve(verbose=-1)
            ier = 1
        except Exception as e:
            ier = -1
        fsolve_info = opt

        res = opt.vary[0].container
        if ier == 1:
            break

    if ier != 1 and not(continue_on_closed_orbit_error):
        raise ClosedOrbitSearchError

    particle_on_co = co_guess.copy()
    particle_on_co.x = res[0]
    particle_on_co.px = res[1]
    particle_on_co.y = res[2]
    particle_on_co.py = res[3]
    particle_on_co.zeta = res[4]
    particle_on_co.delta = res[5]

    particle_on_co._fsolve_info = fsolve_info

    return particle_on_co

def _one_turn_map(p, particle_ref, line, delta_zeta, start, end, num_turns, symmetrize):
    part = particle_ref.copy()
    part.x = p[0]
    part.px = p[1]
    part.y = p[2]
    part.py = p[3]
    part.zeta = p[4] + delta_zeta
    part.delta = p[5]
    part.at_turn = AT_TURN_FOR_TWISS

    if line.energy_program is not None:
        dp0c = line.energy_program.get_p0c_increse_per_turn_at_t_s(
                                                        line.vv['t_turn_s'])
        part.update_p0c_and_energy_deviations(p0c = part._xobject.p0c[0] + dp0c)

    line.track(part, ele_start=start, ele_stop=end, num_turns=num_turns)
    if symmetrize:
        assert num_turns == 1
        with xt.line._preserve_config(line):
            line.config.XSUITE_MIRROR = True
            line.track(part, ele_start=start, ele_stop=end, num_turns=1)
    if part.state[0] < 0:
        raise ClosedOrbitSearchError(
            f'Particle lost, p.state = {part.state[0]}')
    p_res = np.array([
           part._xobject.x[0],
           part._xobject.px[0],
           part._xobject.y[0],
           part._xobject.py[0],
           part._xobject.zeta[0],
           part._xobject.delta[0]])
    return p_res

def _error_for_co_search_6d(p, co_guess, line, delta_zeta, delta0, zeta0, start, end, num_turns, symmetrize):
    return p - _one_turn_map(p, co_guess, line, delta_zeta, start, end, num_turns, symmetrize)

def _error_for_co_search_4d_delta0(p, co_guess, line, delta_zeta, delta0, zeta0, start, end, num_turns, symmetrize):
    one_turn_res = _one_turn_map(p, co_guess, line, delta_zeta, start, end, num_turns, symmetrize)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        0,
        p[5] - delta0])

def _error_for_co_search_4d_zeta0(p, co_guess, line, delta_zeta, delta0, zeta0, start, end, num_turns, symmetrize):
    one_turn_res = _one_turn_map(p, co_guess, line, delta_zeta, start, end, num_turns, symmetrize)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        p[4] - zeta0,
        0])

def _error_for_co_search_4d_delta0_zeta0(p, co_guess, line, delta_zeta, delta0, zeta0, start, end, num_turns, symmetrize):
    one_turn_res = _one_turn_map(p, co_guess, line, delta_zeta, start, end, num_turns, symmetrize)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        p[4] - zeta0,
        p[5] - delta0])

def compute_one_turn_matrix_finite_differences(
        line, particle_on_co,
        steps_r_matrix=None,
        start=None, end=None,
        num_turns=1,
        element_by_element=False,
        only_markers=False,
        symmetrize=True):
    import xpart

    if steps_r_matrix is None:
        steps_r_matrix = {}

    steps_r_matrix = _complete_steps_r_matrix_with_default(steps_r_matrix)

    if line.enable_time_dependent_vars:
        raise RuntimeError(
            'Time-dependent vars not supported in one-turn matrix computation')

    if isinstance(start, str):
        start = line._element_names_unique.index(start)

    if isinstance(end, str):
        end = line._element_names_unique.index(end)

    if start is not None and end is not None and start > end:
        raise ValueError('start > end')

    context = line._buffer.context

    particle_on_co = particle_on_co.copy(
                        _context=context)

    dx = steps_r_matrix["dx"]
    dpx = steps_r_matrix["dpx"]
    dy = steps_r_matrix["dy"]
    dpy = steps_r_matrix["dpy"]
    dzeta = steps_r_matrix["dzeta"]
    ddelta = steps_r_matrix["ddelta"]
    part_temp = xpart.build_particles(_context=context,
            particle_ref=particle_on_co, mode='shift',
            x  =    [0., dx,  0., 0.,  0.,    0.,     0., -dx,   0.,  0.,   0.,     0.,      0.],
            px =    [0., 0., dpx, 0.,  0.,    0.,     0.,  0., -dpx,  0.,   0.,     0.,      0.],
            y  =    [0., 0.,  0., dy,  0.,    0.,     0.,  0.,   0., -dy,   0.,     0.,      0.],
            py =    [0., 0.,  0., 0., dpy,    0.,     0.,  0.,   0.,  0., -dpy,     0.,      0.],
            zeta =  [0., 0.,  0., 0.,  0., dzeta,     0.,  0.,   0.,  0.,   0., -dzeta,      0.],
            delta = [0., 0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],
            )
    dpzeta = float(context.nparray_from_context_array(
        (part_temp.ptau[6] - part_temp.ptau[12])/2/part_temp.beta0[0]))
    if particle_on_co._xobject.at_element[0]>0:
        part_temp.s[:] = particle_on_co._xobject.s[0]
        part_temp.at_element[:] = particle_on_co._xobject.at_element[0]

    part_temp.at_turn = AT_TURN_FOR_TWISS

    if start is not None:
        assert element_by_element is False, 'Not yet implemented'
        assert num_turns == 1, 'Not yet implemented'
        assert num_turns == 1, 'Not yet implemented'
        assert end is not None
        line.track(part_temp, ele_start=start, ele_stop=end)
        if symmetrize:
            with xt.line._preserve_config(line):
                line.config.XSUITE_MIRROR = True
                line.track(part_temp, ele_start=start, ele_stop=end)
    elif particle_on_co._xobject.at_element[0]>0:
        assert element_by_element is False, 'Not yet implemented'
        assert num_turns == 1, 'Not yet implemented'
        assert symmetrize is False, 'Not yet implemented'
        i_start = particle_on_co._xobject.at_element[0]
        line.track(part_temp, ele_start=i_start)
        line.track(part_temp, num_elements=i_start)
    else:
        assert particle_on_co._xobject.at_element[0] == 0
        if element_by_element and num_turns != 1:
            raise NotImplementedError
        monitor_setting = 'ONE_TURN_EBE' if element_by_element else None
        line.track(part_temp, num_turns=num_turns,
                   turn_by_turn_monitor=monitor_setting)
        if symmetrize:
            with xt.line._preserve_config(line):
                line.config.XSUITE_MIRROR = True
                line.track(part_temp, num_turns=num_turns)

    temp_mat = np.zeros(shape=(6, 13), dtype=np.float64)
    temp_mat[0, :] = context.nparray_from_context_array(part_temp.x)
    temp_mat[1, :] = context.nparray_from_context_array(part_temp.px)
    temp_mat[2, :] = context.nparray_from_context_array(part_temp.y)
    temp_mat[3, :] = context.nparray_from_context_array(part_temp.py)
    temp_mat[4, :] = context.nparray_from_context_array(part_temp.zeta)
    temp_mat[5, :] = context.nparray_from_context_array(
                                part_temp.ptau/part_temp.beta0) # pzeta

    RR = np.zeros(shape=(6, 6), dtype=np.float64)

    for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
        RR[:, jj] = (temp_mat[:, jj+1] - temp_mat[:, jj+1+6])/(2*dd)

    out = {'R_matrix': RR}

    if element_by_element:
        mon = line.record_last_track
        temp_mad_ebe = np.zeros(shape=(len(line._element_names_unique) + 1, 6, 13), dtype=np.float64)
        temp_mad_ebe[:, 0, :] = mon.x.T
        temp_mad_ebe[:, 1, :] = mon.px.T
        temp_mad_ebe[:, 2, :] = mon.y.T
        temp_mad_ebe[:, 3, :] = mon.py.T
        temp_mad_ebe[:, 4, :] = mon.zeta.T
        temp_mad_ebe[:, 5, :] = mon.ptau.T/mon.beta0.T

        RR_ebe = np.zeros(shape=(len(line._element_names_unique) + 1, 6, 6), dtype=np.float64)
        for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
            RR_ebe[:, :, jj] = (temp_mad_ebe[:, :, jj+1] - temp_mad_ebe[:, :, jj+1+6])/(2*dd)

        if only_markers:
            mask_twiss = line.tracker._get_twiss_mask_markers()
            mask_twiss[-1] = True # to include the "_end_point"

        out['R_matrix_ebe'] = RR_ebe
    else:
        out['R_matrix_ebe'] = None

    return out


def _updated_kwargs_from_locals(kwargs, loc):

    out = kwargs.copy()

    for kk in kwargs.keys():
        if kk in loc:
            out[kk] = loc[kk]

    out.pop('input_kwargs', None)

    return out


def _build_auxiliary_tracker_with_extra_markers(tracker, at_s, marker_prefix,
                                                algorithm='auto'):

    assert algorithm in ['auto', 'insert', 'regen_all_drift']
    if algorithm == 'auto':
        if len(at_s)<10:
            algorithm = 'insert'
        else:
            algorithm = 'regen_all_drifts'

    auxline = xt.Line(elements=tracker.line.element_dict.copy(),
                      element_names=list(tracker.line.element_names).copy())
    if tracker.line.particle_ref is not None:
        auxline.particle_ref = tracker.line.particle_ref.copy()

    names_inserted_markers = []
    markers = []
    for ii, ss in enumerate(at_s):
        nn = marker_prefix + f'{ii}'
        names_inserted_markers.append(nn)
        markers.append(xt.Drift(length=0))

    auxline.cut_at_s(at_s)
    for nn, mm, ss in zip(names_inserted_markers, markers, at_s):
        auxline.insert_element(element=mm, name=nn, at_s=ss)

    auxtracker = xt.Tracker(
        _buffer=tracker._buffer,
        io_buffer=tracker.io_buffer,
        line=auxline,
        particles_monitor_class=None,
        local_particle_src=tracker.local_particle_src
    )
    auxtracker.line.config = tracker.line.config.copy()
    auxtracker.line._extra_config = tracker.line._extra_config.copy()

    return auxtracker, names_inserted_markers


class TwissInit:

    def __init__(self, particle_on_co=None, W_matrix=None, element_name=None,
                line=None, particle_ref=None,
                x=None, px=None, y=None, py=None, zeta=None, delta=None,
                betx=None, alfx=None, bety=None, alfy=None, bets=None,
                dx=None, dpx=None, dy=None, dpy=None, dzeta=None,
                mux=None, muy=None, muzeta=None,
                ddx=None, ddpx=None, ddy=None, ddpy=None, ddzeta=None,
                ax_chrom=None, bx_chrom=None, ay_chrom=None, by_chrom=None,
                reference_frame=None):

        # Custom setattr needs to be bypassed for creation of attributes
        object.__setattr__(self, 'particle_on_co', None)
        self._temp_co_data = None
        self._temp_optics_data = None

        if particle_on_co is None:
            self._temp_co_data = dict(
                x=(x or 0.),
                px=(px or 0.),
                y=(y or 0.),
                py=(py or 0.),
                zeta=(zeta or 0.),
                delta=(delta or 0.),
            )
        else:
            assert x is None, "`x` must be None if `particle_on_co` is provided"
            assert px is None, "`px` must be None if `particle_on_co` is provided"
            assert y is None, "`y` must be None if `particle_on_co` is provided"
            assert py is None, "`py` must be None if `particle_on_co` is provided"
            assert zeta is None, "`zeta` must be None if `particle_on_co` is provided"
            assert delta is None, "`delta` must be None if `particle_on_co` is provided"
            assert particle_ref is None, (
                "`particle_ref` must be None if `particle_on_co` is provided")
            self.__dict__['particle_on_co'] = particle_on_co

        if W_matrix is None:
            alfx = alfx or 0
            alfy = alfy or 0
            betx = betx or 1
            bety = bety or 1
            bets = bets or 1
            dx = dx or 0
            dpx = dpx or 0
            dy = dy or 0
            dpy = dpy or 0

            self._temp_optics_data = dict(
                betx=betx, alfx=alfx, bety=bety, alfy=alfy, bets=bets,
                dx=dx, dpx=dpx, dy=dy, dpy=dpy)
        else:
            assert betx is None, "`betx` must be None if `W_matrix` is provided"
            assert alfx is None, "`alfx` must be None if `W_matrix` is provided"
            assert bety is None, "`bety` must be None if `W_matrix` is provided"
            assert alfy is None, "`alfy` must be None if `W_matrix` is provided"
            assert bets is None, "`bets` must be None if `W_matrix` is provided"
            self._temp_co_data = None

        self.element_name = element_name
        self.W_matrix = W_matrix
        self.mux = (mux or 0.)
        self.muy = (muy or 0.)
        self.muzeta = (muzeta or 0.)
        self.dzeta = (dzeta or 0.)
        self.ax_chrom = (ax_chrom or 0.)
        self.bx_chrom = (bx_chrom or 0.)
        self.ay_chrom = (ay_chrom or 0.)
        self.by_chrom = (by_chrom or 0.)
        self.ddx = (ddx or 0.)
        self.ddpx = (ddpx or 0.)
        self.ddy = (ddy or 0.)
        self.ddpy = (ddpy or 0.)
        self.reference_frame = reference_frame

        if line is not None and element_name is not None:
            self._complete(line, element_name)

    def to_dict(self):
        '''
        Convert to dictionary representation.
        '''
        out = self.__dict__.copy()
        out['particle_on_co'] = out['particle_on_co'].to_dict()
        return out

    def to_json(self, file, **kwargs):

        '''
        Convert to JSON representation.

        Parameters
        ----------
        file : str or file-like

        '''

        # Can reuse the one from the Line (it is general enough)
        return xt.Line.to_json(self, file, **kwargs)

    @classmethod
    def from_dict(cls, dct):
        '''
        Convert from dictionary representation.

        Parameters
        ----------
        dct : dict
            Dictionary representation.

        Returns
        -------
        out : TwissInit
            TwissInit instance.
        '''

        # Need the values as numpy types, in particular arrays
        numpy_dct = {}
        for key, value in dct.items():
            if isinstance(value, int):
                numpy_dct[key] = np.int64(value)
            elif isinstance(value, float):
                numpy_dct[key] = np.float64(value)
            elif isinstance(value, str):
                numpy_dct[key] = np.str_(value)
            elif isinstance(value, list):
                numpy_dct[key] = np.array(value)
            else:
                numpy_dct[key] = value

        numpy_dct['particle_on_co'] = xt.Particles.from_dict(dct['particle_on_co'])

        out = cls()
        out.__dict__.update(numpy_dct)
        return out

    @classmethod
    def from_json(cls, file):

        '''
        Convert from JSON representation.

        Parameters
        ----------
        file : str or file-like
            File name or file-like object.

        Returns
        -------
        out : TwissInit
            TwissInit instance.

        '''

        if isinstance(file, io.IOBase):
            dct = json.load(file)
        else:
            with open(file, 'r') as fid:
                dct = json.load(fid)

        return cls.from_dict(dct)

    def _complete(self, line, element_name):

        if (line is not None and 'reverse' in line.twiss_default
            and line.twiss_default['reverse']):
            input_reversed = True
            assert self.reference_frame is None, ("`reference_frame` must be None "
                "if `twiss_default['reverse']` is True")
        else:
            input_reversed = False

        if self._temp_co_data is not None:
            import xpart
            assert line is not None, (
                "`line` must be provided if `particle_on_co` is None")

            i_ele_in_line = _str_to_index(line, element_name, allow_end_point=False)
            s_ele_in_line = line.tracker._tracker_data_base.element_s_locations[i_ele_in_line]

            if input_reversed:
                s_ele_twiss = line.tracker._tracker_data_base.line_length - s_ele_in_line
                first_ele = line[line._element_names_unique[i_ele_in_line]]
                if hasattr(first_ele, 'isthick') and first_ele.isthick:
                    s_ele_twiss -= first_ele.length
            else:
                s_ele_twiss = s_ele_in_line

            particle_on_co = xpart.build_particles(
                x=self._temp_co_data['x'], px=self._temp_co_data['px'],
                y=self._temp_co_data['y'], py=self._temp_co_data['py'],
                delta=self._temp_co_data['delta'], zeta=self._temp_co_data['zeta'],
                line=line,
                include_collective=True, # In fact it does not matter
            )
            particle_on_co.s = s_ele_twiss
            self.__dict__['particle_on_co'] = particle_on_co
            self._temp_co_data = None

        if self._temp_optics_data is not None:

            aux_segment = xt.LineSegmentMap(
                length=1., # dummy
                qx=0.55, # dummy
                qy=0.57, # dummy
                qs=0.0000001, # dummy
                bets=self._temp_optics_data['bets'],
                betx=self._temp_optics_data['betx'],
                bety=self._temp_optics_data['bety'],
                alfx=self._temp_optics_data['alfx'] * (-1 if input_reversed else 1),
                alfy=self._temp_optics_data['alfy'] * (-1 if input_reversed else 1),
                dx=self._temp_optics_data['dx'] * (-1 if input_reversed else 1),
                dy=self._temp_optics_data['dy'],
                dpx=self._temp_optics_data['dpx'],
                dpy=self._temp_optics_data['dpy'] * (-1 if input_reversed else 1),
                )
            aux_line = xt.Line(elements=[aux_segment])
            aux_line.particle_ref = particle_on_co.copy(
                                        _context=xo.context_default)
            aux_line.particle_ref.reorganize()
            aux_line.build_tracker()
            aux_tw = aux_line.twiss()
            W_matrix = aux_tw.W_matrix[0]

            if input_reversed:
                W_matrix[0, :] = -W_matrix[0, :]
                W_matrix[1, :] = W_matrix[1, :]
                W_matrix[2, :] = W_matrix[2, :]
                W_matrix[3, :] = -W_matrix[3, :]
                W_matrix[4, :] = -W_matrix[4, :]
                W_matrix[5, :] = W_matrix[5, :]
                self.reference_frame = 'reverse'

            self.W_matrix = W_matrix
            self._temp_optics_data = None

        self.element_name = element_name

    def _needs_complete(self):
        return self._temp_co_data is not None or self._temp_optics_data is not None

    def copy(self):
        if self.particle_on_co is not None:
            pco = self.particle_on_co.copy()
        else:
            pco = None

        if self.W_matrix is not None:
            wmat = self.W_matrix.copy()
        else:
            wmat = None

        out =  TwissInit(
            particle_on_co=pco,
            W_matrix=wmat,
            element_name=self.element_name,
            mux=self.mux,
            muy=self.muy,
            muzeta=self.muzeta,
            dzeta=self.dzeta,
            ax_chrom=self.ax_chrom,
            bx_chrom=self.bx_chrom,
            ay_chrom=self.ay_chrom,
            by_chrom=self.by_chrom,
            ddx=self.ddx,
            ddpx=self.ddpx,
            ddy=self.ddy,
            ddpy=self.ddpy,
            reference_frame=self.reference_frame)

        if self._temp_co_data is not None:
            out._temp_co_data = self._temp_co_data.copy()

        if self._temp_optics_data is not None:
            out._temp_optics_data = self._temp_optics_data.copy()

        return out

    def reverse(self):
        out = TwissInit(
            particle_on_co=self.particle_on_co.copy(),
            W_matrix=self.W_matrix.copy(),
            ax_chrom=(-self.ax_chrom if self.ax_chrom is not None else None),
            ay_chrom=(-self.ay_chrom if self.ay_chrom is not None else None),
            bx_chrom=self.bx_chrom,
            by_chrom=self.by_chrom,
            ddx=(-self.ddx if self.ddx is not None else None),
            ddpx=(self.ddpx if self.ddpx is not None else None),
            ddy=(self.ddy if self.ddy is not None else None),
            ddpy=(-self.ddpy if self.ddpy is not None else None),
            )
        out.particle_on_co.x = -out.particle_on_co.x
        out.particle_on_co.py = -out.particle_on_co.py
        out.particle_on_co.zeta = -out.particle_on_co.zeta

        out.W_matrix[0, :] = -out.W_matrix[0, :]
        out.W_matrix[1, :] = out.W_matrix[1, :]
        out.W_matrix[2, :] = out.W_matrix[2, :]
        out.W_matrix[3, :] = -out.W_matrix[3, :]
        out.W_matrix[4, :] = -out.W_matrix[4, :]
        out.W_matrix[5, :] = out.W_matrix[5, :]

        out.mux = 0
        out.muy = 0
        out.muzeta = 0
        out.dzeta = 0

        out.element_name = self.element_name
        out.reference_frame = {'proper': 'reverse', 'reverse': 'proper'}[self.reference_frame]

        return out

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.__dict__['particle_on_co'], name):
            # e.g. tw_init['x'] returns tw_init.particle_on_co.x
            out = getattr(self.__dict__['particle_on_co'], name)
            #always cpu
            if hasattr(out, 'get'):
                out = out.get()
            if hasattr(out, '__iter__'):
                out = out [0]
            return out
        else:
            raise AttributeError(f'No attribute {name} found in TwissInit')

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        elif hasattr(self.particle_on_co, name):
            setattr(self.particle_on_co, name, value)
        else:
            self.__dict__[name] = value

    def get_normalized_coordinates(self, particles, nemitt_x=None, nemitt_y=None,
                                   nemitt_zeta=None):

        ctx2np = particles._context.nparray_from_context_array

        part_id = ctx2np(particles.particle_id).copy()
        at_element = ctx2np(particles.at_element).copy()
        at_turn = ctx2np(particles.at_turn).copy()
        x_norm = ctx2np(particles.x).copy()
        px_norm = x_norm.copy()
        y_norm = x_norm.copy()
        py_norm = x_norm.copy()
        zeta_norm = x_norm.copy()
        pzeta_norm = x_norm.copy()

        XX_norm  = _W_phys2norm(x = ctx2np(particles.x),
                                px = ctx2np(particles.px),
                                y = ctx2np(particles.y),
                                py = ctx2np(particles.py),
                                zeta = ctx2np(particles.zeta),
                                pzeta = ctx2np(particles.ptau)/ctx2np(particles.beta0),
                                W_matrix = self.W_matrix,
                                co_dict = self.particle_on_co.copy(_context=xo.context_default).to_dict(),
                                nemitt_x = nemitt_x,
                                nemitt_y = nemitt_y,
                                nemitt_zeta = nemitt_zeta)

        if XX_norm.ndim == 2:
            x_norm = XX_norm[0, :]
            px_norm = XX_norm[1, :]
            y_norm = XX_norm[2, :]
            py_norm = XX_norm[3, :]
            zeta_norm = XX_norm[4, :]
            pzeta_norm = XX_norm[5, :]

        elif XX_norm.ndim == 3:
            x_norm = XX_norm[0, :, :].flatten()
            px_norm = XX_norm[1, :, :].flatten()
            y_norm = XX_norm[2, :, :].flatten()
            py_norm = XX_norm[3, :, :].flatten()
            zeta_norm = XX_norm[4, :, :].flatten()
            pzeta_norm = XX_norm[5, :, :].flatten()
            part_id = part_id.flatten()
            at_element = at_element.flatten()
            at_turn = at_turn.flatten()

        return Table({'particle_id': part_id, 'at_element': at_element,'at_turn':at_turn,
                      'x_norm': x_norm, 'px_norm': px_norm, 'y_norm': y_norm,
                      'py_norm': py_norm, 'zeta_norm': zeta_norm,
                      'pzeta_norm': pzeta_norm}, index='particle_id')

    @property
    def betx(self):
        WW = self.W_matrix
        return WW[0, 0]**2 + WW[0, 1]**2

    @property
    def bety(self):
        WW = self.W_matrix
        return WW[2, 2]**2 + WW[2, 3]**2

    @property
    def alfx(self):
        WW = self.W_matrix
        return -WW[0, 0] * WW[1, 0] - WW[0, 1] * WW[1, 1]

    @property
    def alfy(self):
        WW = self.W_matrix
        return -WW[2, 2] * WW[3, 2] - WW[2, 3] * WW[3, 3]

    @property
    def dx(self):
        WW = self.W_matrix
        return (WW[0, 5] - WW[0, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])

    @property
    def dpx(self):
        WW = self.W_matrix
        return (WW[1, 5] - WW[1, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])

    @property
    def dy(self):
        WW = self.W_matrix
        return (WW[2, 5] - WW[2, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])

    @property
    def dpy(self):
        WW = self.W_matrix
        return (WW[3, 5] - WW[3, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])


class TwissTable(Table):

    def __init__(self, *args, **kwargs):
        kwargs['sep_count'] = kwargs.get('sep_count', '::::')
        super().__init__(*args, **kwargs)
        self['periodic'] = False

    _error_on_row_not_found = True

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
        part.at_element[:] = -1

        W = self.W_matrix[at_element]

        if 'ax_chrom' in self.keys():
            ax_chrom = self.ax_chrom[at_element]
            bx_chrom = self.bx_chrom[at_element]
            ay_chrom = self.ay_chrom[at_element]
            by_chrom = self.by_chrom[at_element]
            ddx = self.ddx[at_element]
            ddpx = self.ddpx[at_element]
            ddy = self.ddy[at_element]
            ddpy = self.ddpy[at_element]
        else:
            ax_chrom = None
            bx_chrom = None
            ay_chrom = None
            by_chrom = None
            ddx = None
            ddpx = None
            ddy = None
            ddpy = None

        if 'mux' in self.keys():
            mux = self.mux[at_element]
            muy = self.muy[at_element]
            muzeta = self.muzeta[at_element]
        else:
            mux = 0
            muy = 0
            muzeta = 0

        if 'dzeta' in self.keys():
            dzeta = self.dzeta[at_element]
        else:
            dzeta = 0

        return TwissInit(particle_on_co=part, W_matrix=W,
                        element_name=str(self.name[at_element]),
                        mux=mux, muy=muy, muzeta=muzeta, dzeta=dzeta,
                        ax_chrom=ax_chrom, bx_chrom=bx_chrom,
                        ay_chrom=ay_chrom, by_chrom=by_chrom,
                        ddx=ddx, ddpx=ddpx, ddy=ddy, ddpy=ddpy,
                        reference_frame=self.reference_frame)

    def get_betatron_sigmas(self, nemitt_x, nemitt_y):
        # For backward compatibility
        return self.get_beam_covariance(
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    def get_beam_covariance(self,
            nemitt_x=None, nemitt_y=None, nemitt_zeta=None,
            gemitt_x=None, gemitt_y=None, gemitt_zeta=None):

        # See MAD8 physics manual (Eq. 8.59)

        beta0 = self.particle_on_co.beta0
        gamma0 = self.particle_on_co.gamma0

        if nemitt_x is not None:
            assert gemitt_x is None, 'Cannot provide both nemitt_x and gemitt_x'
            gemitt_x = nemitt_x / (beta0 * gamma0)

        if nemitt_y is not None:
            assert gemitt_y is None, 'Cannot provide both nemitt_y and gemitt_y'
            gemitt_y = nemitt_y / (beta0 * gamma0)

        if nemitt_zeta is not None:
            assert gemitt_zeta is None, 'Cannot provide both nemitt_zeta and gemitt_zeta'
            gemitt_zeta = nemitt_zeta / (beta0 * gamma0)

        gemitt_x = gemitt_x or 0
        gemitt_y = gemitt_y or 0
        gemitt_zeta = gemitt_zeta or 0

        Ws = self.W_matrix.copy()

        if self.method == '4d':
            Ws[:, 4:, 4:] = 0

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

        Sigma = gemitt_x * Sigma1 + gemitt_y * Sigma2 + gemitt_zeta * Sigma3
        res = _build_sigma_table(Sigma=Sigma, s=self.s, name=self.name)

        return res

    def get_ibs_growth_rates(
        self,
        formalism: str,
        total_beam_intensity: int = None,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
        bunch_length: float = None,
        bunched: bool = True,
        **kwargs,
    ):
        """
        Computes IntraBeam Scattering (amplitude) growth rates.

        Parameters
        ----------
        formalism : str
            Which formalism to use for the computation. Can be ``Nagaitsev``
            or ``Bjorken-Mtingwa`` (also accepts ``B&M``), case-insensitively.
        total_beam_intensity : int, optional
            The beam intensity. Required if `particles` is not provided.
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. If `particles` is not
            provided, either this parameter or `nemitt_x` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. If `particles` is not
            provided, either this parameter or `gemitt_x` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. If `particles` is not
            provided, either this parameter or `nemitt_y` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. If `particles` is not
            provided, either this parameter or `gemitt_y` is required.
        sigma_delta : float, optional
            The momentum spread. Required if `particles` is not provided.
        bunch_length : float, optional
            The bunch length in [m]. Required if `particles` is not provided.
        bunched : bool, optional
            Whether the beam is bunched or not (coasting). Defaults to `True`.
            Required if `particles` is not provided.
        **kwargs : dict
            Keyword arguments are passed to the growth rates computation method of
            the chosen IBS formalism implementation. See the IBS details from the
            `xfields` package directly.

        Returns
        -------
        IBSGrowthRates
            An ``IBSGrowthRates`` object with the computed growth rates.
        """
        try:
            from xfields.ibs import get_intrabeam_scattering_growth_rates
        except ImportError:
            raise ImportError("Please install xfields to use this feature.")
        return get_intrabeam_scattering_growth_rates(
            self, formalism, total_beam_intensity,
            gemitt_x, nemitt_x, gemitt_y, nemitt_y,
            sigma_delta, bunch_length, bunched,
            **kwargs,
        )

    def get_R_matrix(self, start, end):

        assert self.values_at == 'entry', 'Not yet implemented for exit'

        if isinstance(start, str):
            start = np.where(self.name == start)[0][0]
        if isinstance(end, str):
            end = np.where(self.name == end)[0][0]

        if start > end:
            raise ValueError('start must be smaller than ele_end')

        W_start = self.W_matrix[start]
        W_end = self.W_matrix[end]

        phix_start = self.phix[start]
        phix_end = self.phix[end]
        phiy_start = self.phiy[start]
        phiy_end = self.phiy[end]
        phizeta_start = self.phizeta[start]
        phizeta_end = self.phizeta[end]

        phi_x = phix_end - phix_start
        phi_y = phiy_end - phiy_start
        phi_zeta = phizeta_end - phizeta_start

        Rot = np.zeros(shape=(6, 6), dtype=np.float64)

        Rot[0:2,0:2] = lnf.Rot2D(phi_x)
        Rot[2:4,2:4] = lnf.Rot2D(phi_y)
        Rot[4:6,4:6] = lnf.Rot2D(phi_zeta)

        R_matrix = W_end @ Rot @ np.linalg.inv(W_start)

        return R_matrix

    def get_R_matrix_table(self):

        Rot = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)

        cos_phix = np.cos(self.phix - self.phix[0])
        sin_phix = np.sin(self.phix - self.phix[0])
        cos_phiy = np.cos(self.phiy - self.phiy[0])
        sin_phiy = np.sin(self.phiy - self.phiy[0])
        cos_phizeta = np.cos(self.phizeta - self.phizeta[0])
        sin_phizeta = np.sin(self.phizeta - self.phizeta[0])

        Rot[:, 0, 0] = cos_phix
        Rot[:, 0, 1] = sin_phix
        Rot[:, 1, 0] = -sin_phix
        Rot[:, 1, 1] = cos_phix
        Rot[:, 2, 2] = cos_phiy
        Rot[:, 2, 3] = sin_phiy
        Rot[:, 3, 2] = -sin_phiy
        Rot[:, 3, 3] = cos_phiy
        Rot[:, 4, 4] = cos_phizeta
        Rot[:, 4, 5] = sin_phizeta
        Rot[:, 5, 4] = -sin_phizeta
        Rot[:, 5, 5] = cos_phizeta

        # Compute W @ Rot @ W_inv slice by slice
        WW = self.W_matrix
        R_matrix_ebe = np.einsum('ijk,ikl->ijl', WW, Rot) @ np.linalg.inv(WW[0, :, :])

        out_dct = {'s': self.s, 'name': self.name, 'R_matrix': R_matrix_ebe}
        for ii in range(6):
            for jj in range(6):
                out_dct[f'r{ii+1}{jj+1}'] = R_matrix_ebe[:, ii, jj]

        return Table(out_dct)

    def get_normalized_coordinates(self, particles, nemitt_x=None, nemitt_y=None,
                                   nemitt_zeta=None, _force_at_element=None):

        # TODO: check consistency of gamma0

        ctx2np = particles._context.nparray_from_context_array
        at_element_particles = ctx2np(particles.at_element)

        part_id = ctx2np(particles.particle_id).copy()
        at_element = part_id.copy() * 0 + xt.particles.LAST_INVALID_STATE
        x_norm = ctx2np(particles.x).copy() * 0 + xt.particles.LAST_INVALID_STATE
        px_norm = x_norm.copy()
        y_norm = x_norm.copy()
        py_norm = x_norm.copy()
        zeta_norm = x_norm.copy()
        pzeta_norm = x_norm.copy()

        at_element_no_rep = list(set(
            at_element_particles[part_id > xt.particles.LAST_INVALID_STATE]))

        for at_ele in at_element_no_rep:

            if _force_at_element is not None:
                at_ele = _force_at_element

            W = self.W_matrix[at_ele]

            mask_at_ele = at_element_particles == at_ele

            if _force_at_element is not None:
                mask_at_ele = ctx2np(particles.state) > xt.particles.LAST_INVALID_STATE


            XX_norm  = _W_phys2norm(x = ctx2np(particles.x)[mask_at_ele],
                                    px = ctx2np(particles.px)[mask_at_ele],
                                    y = ctx2np(particles.y)[mask_at_ele],
                                    py = ctx2np(particles.py)[mask_at_ele],
                                    zeta = ctx2np(particles.zeta)[mask_at_ele],
                                    pzeta = ctx2np(particles.ptau)[mask_at_ele]/ctx2np(particles.beta0)[mask_at_ele],
                                    W_matrix = W,
                                    co_dict = {'x': self.x[at_ele], 'px': self.px[at_ele],
                                               'y': self.y[at_ele], 'py': self.py[at_ele],
                                               'zeta': self.zeta[at_ele], 'ptau': self.ptau[at_ele],
                                               'beta0': self.particle_on_co._xobject.beta0[0],
                                               'gamma0': self.particle_on_co._xobject.gamma0[0]},
                                    nemitt_x = nemitt_x,
                                    nemitt_y = nemitt_y,
                                    nemitt_zeta = nemitt_zeta)

            x_norm[mask_at_ele] = XX_norm[0, :]
            px_norm[mask_at_ele] = XX_norm[1, :]
            y_norm[mask_at_ele] = XX_norm[2, :]
            py_norm[mask_at_ele] = XX_norm[3, :]
            zeta_norm[mask_at_ele] = XX_norm[4, :]
            pzeta_norm[mask_at_ele] = XX_norm[5, :]
            at_element[mask_at_ele] = at_ele

        return Table({'particle_id': part_id, 'at_element': at_element,
                      'x_norm': x_norm, 'px_norm': px_norm, 'y_norm': y_norm,
                      'py_norm': py_norm, 'zeta_norm': zeta_norm,
                      'pzeta_norm': pzeta_norm}, index='particle_id')

    def reverse(self):

        assert self.values_at == 'entry', 'Not yet implemented for exit'
        assert self.name[-1] == '_end_point' # Needed for the present implementation

        new_data = {}
        for kk, vv in self._data.items():
            if hasattr(vv, 'copy'):
                new_data[kk] = vv.copy()
            else:
                new_data[kk] = vv

        if self.only_markers:
            itake = slice(None, -1, None)
        else:
            # To keep association name <-> quantities at elemement entry
            itake = slice(1, None, None)

        for kk in self._col_names:
            if (kk == 'name' or kk == 'name_env'
                    or kk in NORMAL_STRENGTHS_FROM_ATTR
                    or kk in SKEW_STRENGTHS_FROM_ATTR
                    or kk in OTHER_FIELDS_FROM_ATTR
                    or kk in OTHER_FIELDS_FROM_TABLE
                    ):
                new_data[kk][:-1] = new_data[kk][:-1][::-1]
                new_data[kk][-1] = self[kk][-1]
            elif kk == 'W_matrix':
                new_data[kk][:-1, :, :] = new_data[kk][itake, :, :][::-1, :, :]
                new_data[kk][-1, :, :] = self[kk][0, :, :]
            else:
                new_data[kk][:-1] = new_data[kk][itake][::-1]
                new_data[kk][-1] = self[kk][0]

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

        if 'kin_px' in out:
            out.kin_px = out.kin_px
            out.kin_py = -out.kin_py
            out.kin_xprime = out.kin_xprime
            out.kin_yprime = -out.kin_yprime

        if 'betx' in out:
            # if optics calculation is not skipped
            out.betx = out.betx
            out.bety = out.bety
            out.alfx = -out.alfx # Dpx/Dx
            out.alfy = -out.alfy # Dpy/Dy
            out.gamx = out.gamx
            out.gamy = out.gamy

            out.dx = -out.dx
            out.dpx = out.dpx
            out.dy = out.dy
            out.dpy = -out.dpy
            if 'dzeta' in out:
                out.dzeta = -out.dzeta

            if 'dx_zeta' in out._col_names:
                out.dx_zeta = out.dx_zeta
                out.dpx_zeta = -out.dpx_zeta
                out.dy_zeta = -out.dy_zeta
                out.dpy_zeta = out.dpy_zeta

            # Untested:
            # if 'alfx2' in out._col_names:
            #     out.alfx2 = -out.alfx2
            #     out.alfy2 = -out.alfy2

            out.W_matrix[:, 0, :] = -out.W_matrix[:, 0, :]
            out.W_matrix[:, 1, :] = out.W_matrix[:, 1, :]
            out.W_matrix[:, 2, :] = out.W_matrix[:, 2, :]
            out.W_matrix[:, 3, :] = -out.W_matrix[:, 3, :]
            out.W_matrix[:, 4, :] = -out.W_matrix[:, 4, :]
            out.W_matrix[:, 5, :] = out.W_matrix[:, 5, :]

            out.mux = out.mux[0] - out.mux
            out.muy = out.muy[0] - out.muy
            out.muzeta = out.muzeta[0] - out.muzeta
            out.phix = -out.phix
            out.phiy = -out.phiy

            if 'dzeta' in out:
                out.dzeta = out.dzeta[0] - out.dzeta

        if 'ax_chrom' in out._col_names:
            out.ax_chrom = -out.ax_chrom
            out.ay_chrom = -out.ay_chrom
            out.ddx = -out.ddx
            out.ddpy = -out.ddpy

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

        _reverse_strengths(out._data)

        out._data['reference_frame'] = {
            'proper': 'reverse', 'reverse': 'proper'}[self.reference_frame]

        return out

    ind_per_table = []

    def add_strengths(self, line=None):
        if line is None and hasattr(self,"_action"):
            line = self._action.line
        _add_strengths_to_twiss_res(self, line)
        return self

    @classmethod
    def concatenate(cls, tables_to_concat):

        # Check values_at compatibility
        assert len(set([tt.values_at for tt in tables_to_concat])) == 1, (
            'All tables must have the same values_at')

        # Check reference_frame compatibility
        assert len(set([tt.reference_frame for tt in tables_to_concat])) == 1, (
            'All tables must have the same reference_frame')

        # trim away common markers
        ind_per_table = []
        for ii, tt in enumerate(tables_to_concat):
            this_ind = [0, len(tt)]
            if ii > 0:
                if tt.name[0] in tables_to_concat[ii-1].name:
                    assert tt.name[0] == tables_to_concat[ii-1].name[ind_per_table[ii-1][1]-1]
                    ind_per_table[ii-1][1] -= 1
            if ii < len(tables_to_concat) - 1:
                if tt.name[-1] == '_end_point':
                    this_ind[1] -= 1

            ind_per_table.append(this_ind)

        n_elem = sum([ind[1] - ind[0] for ind in ind_per_table])

        new_data = {}
        for kk in tables_to_concat[0]._col_names:
            if kk == 'W_matrix':
                new_data[kk] = np.empty(
                    (n_elem, 6, 6), dtype=tables_to_concat[0][kk].dtype)
                continue
            dtype=tables_to_concat[0][kk].dtype
            if dtype.str.startswith('<U'):
                str_len = np.max([int(tables_to_concat[ii][kk].dtype.str.split('<U')[-1])
                                    for ii in range(len(tables_to_concat))])
                dtype = f'<U{str_len}'
            new_data[kk] = np.empty(n_elem, dtype=dtype)

        i_start = 0
        for ii, tt in enumerate(tables_to_concat):
            i_end = i_start + ind_per_table[ii][1] - ind_per_table[ii][0]
            for kk in tt._col_names:
                if kk == 'W_matrix':
                    new_data[kk][i_start:i_end] = (
                        tt[kk][ind_per_table[ii][0]:ind_per_table[ii][1], :, :])
                    continue
                new_data[kk][i_start:i_end] = (
                    tt[kk][ind_per_table[ii][0]:ind_per_table[ii][1]])
                if kk in CYCLICAL_QUANTITIES:
                    new_data[kk][i_start:i_end] -= new_data[kk][i_start]
                    if ii > 0:
                        new_data[kk][i_start:i_end] += new_data[kk][i_start-1]
                        new_data[kk][i_start:i_end] += (
                            tables_to_concat[ii-1][kk][-1]
                            - tables_to_concat[ii-1][kk][ind_per_table[ii-1][1]-1])

            i_start = i_end

        new_table = cls(new_data)
        new_table._data['values_at'] = tables_to_concat[0].values_at
        new_table._data['reference_frame'] = tables_to_concat[0].reference_frame
        new_table._data['particle_on_co'] = tables_to_concat[0].particle_on_co

        return new_table

    def zero_at(self, name):
        for kk in CYCLICAL_QUANTITIES:
            if kk in self:
                self[kk] -= self[kk, name]

    def target(self, tars=None, value=None, at=None, **kwargs):
        if value is None:
            value = self
        tarset = xt.TargetSet(tars=tars, value=value, at=at,
                              action=self._action, **kwargs)
        return tarset

    def plot(self,
            yl=None,
            yr=None,x='s',
            lattice=True,
            mask=None,
            labels=None,
            clist="k r b g c m",
            figure=None,
            figlabel=None,
            ax=None,
            axleft=None,
            axright=None,
            axlattice=None,
            hover=False,
            grid=True,
            figsize=(6.4*1.2, 4.8),
            lattice_only=False
            ):
        """
        Plot columns of the TwissTable

        Parameters:
        -----------
        yl: str
            space separated columns or expressions to plot on the left y-axis
        yr: str
            space separated columns or expressions to plot on the right y-axis
        x: str
            column to plot on the x-axis
        lattice: bool
            if True, the lattice is plotted
        mask: slice
            mask to select the elements to plot
        labels: str
            mask to select the elements to label
        clist: str
            colors to use
        ax: matplotlib axis
            axis to plot on
        figlabel: str
            label to use for the figure
        """

        if yl is None and yr is None:
            yl='betx bety'
            yr='dx dy'
        if yl is None:
            yl=""
        if yr is None:
            yr=""

        if not hasattr(self,"_action"):
            lattice=False

        if lattice and 'length' not in self.keys():
            self.add_strengths()

        if mask is not None:
            if isinstance(mask,str):
                idx=self.mask[mask]
            else:
                idx=mask
        else:
            idx=slice(None)

        self._is_s_begin=True

        if lattice_only:
            yl = ''
            yr = ''

        pl=TwissPlot(self,
                x=x,
                yl=yl,
                yr=yr,
                idx=idx,
                lattice=lattice,
                figure=figure,
                figlabel=figlabel,clist=clist,
                ax=ax,
                axleft=axleft,
                axright=axright,
                axlattice=axlattice,
                hover=hover,
                figsize=figsize,
                grid=grid,
                )

        if labels is not None:
            mask=self.mask[labels]
            labels=self[self._index][mask]
            xs=self[x][mask]
            pl.left.set_xticks(xs,labels)

        if lattice_only:
            ax1 = pl.lattice.twinx()
            ax1.yaxis.set_label_position("left")
            ax1.yaxis.set_ticks_position("left")
            ax1.set_autoscale_on(True)
            pl.left = ax1

        return pl

    def _compute_radiation_integrals(self, add_to_tw=False):

        angle_rad = self['angle_rad']
        rot_s_rad = self['rot_s_rad']
        x = self['x']
        y = self['y']
        kin_px = self['kin_px']
        kin_py = self['kin_py']
        delta = self['delta']
        length = self['length']

        betx = self['betx']             # Twiss beta function x
        alfx = self['alfx']             # Twiss alpha x
        gamx = self['gamx']             # Twiss gamma x
        bety = self['bety']             # Twiss beta function y
        alfy = self['alfy']             # Twiss alpha y
        gamy = self['gamy']             # Twiss gamma y
        dx = self['dx']                 # Dispersion x
        dy = self['dy']                 # Dispersion y
        dpx = self['dpx']               # Dispersion px
        dpy = self['dpy']               # Dispersion py

        mass0 = self.particle_on_co.mass0
        r0 = self.particle_on_co.get_classical_particle_radius0()
        gamma0 = self.gamma0

        dxprime = dpx * (1 - delta) - kin_px
        dyprime = dpy * (1 - delta) - kin_py

        # Curvature of the reference trajectory
        mask = length != 0
        kappa0_x = 0 * angle_rad
        kappa0_y = 0 * angle_rad
        kappa0_x[mask] = angle_rad[mask] * np.cos(rot_s_rad[mask]) / length[mask]
        kappa0_y[mask] = angle_rad[mask] * np.sin(rot_s_rad[mask]) / length[mask]
        kappa0 = np.sqrt(kappa0_x**2 + kappa0_y**2)

        # Field index
        k1 = 0 * angle_rad
        k1[mask] = self.k1l[mask] / length[mask]

        # Compute x', y', x'', y''
        ps = np.sqrt((1 + delta)**2 - kin_px**2 - kin_py**2)
        xp = kin_px / ps
        yp = kin_py / ps
        xp_ele = xp * 0
        yp_ele = yp * 0
        xp_ele[:-1] = (xp[:-1] + xp[1:]) / 2
        yp_ele[:-1] = (yp[:-1] + yp[1:]) / 2

        mask_length = length != 0
        xpp_ele = xp_ele * 0
        ypp_ele = yp_ele * 0
        xpp_ele[mask_length] = np.diff(xp, append=0)[mask_length] / length[mask_length]
        ypp_ele[mask_length] = np.diff(yp, append=0)[mask_length] / length[mask_length]

        # Curvature of the particle trajectory
        hhh = 1 + kappa0_x * x + kappa0_y * y
        hprime = kappa0_x * xp_ele + kappa0_y * yp_ele
        mask1 = xpp_ele**2 + hhh**2 != 0
        mask2 = xpp_ele**2 + hhh**2 != 0
        kappa_x = (-(hhh * (xpp_ele - hhh * kappa0_x) - 2 * hprime * xp_ele)[mask1]
                / (xp_ele**2 + hhh**2)[mask1]**(3/2))
        kappa_y = (-(hhh * (ypp_ele - hhh * kappa0_y) - 2 * hprime * yp_ele)[mask2]
                / (yp_ele**2 + hhh**2)[mask2]**(3/2))
        kappa = np.sqrt(kappa_x**2 + kappa_y**2)

        # Curly H
        Hx_rad = gamx * dx**2 + 2*alfx * dx * dxprime + betx * dxprime**2
        Hy_rad = gamy * dy**2 + 2*alfy * dy * dyprime + bety * dyprime**2

        # Integrands
        i1x_integrand = kappa0_x * dx
        i1y_integrand = kappa0_y * dy

        i2_integrand = kappa * kappa

        i3_integrand = np.abs(kappa * kappa * kappa)

        i4x_integrand = (kappa * kappa + 2 * k1) * kappa0_x * dx
        i4y_integrand = (kappa * kappa - 2 * k1) * kappa0_y * dy

        i5x_integrand = np.abs(kappa * kappa * kappa) * Hx_rad
        i5y_integrand = np.abs(kappa * kappa * kappa) * Hy_rad

        # Integrate
        i1x = np.sum(i1x_integrand * length)
        i1y = np.sum(i1y_integrand * length)
        i2 = np.sum(i2_integrand * length)
        i3 = np.sum(i3_integrand * length)
        i4x = np.sum(i4x_integrand * length)
        i4y = np.sum(i4y_integrand * length)
        i5x = np.sum(i5x_integrand * length)
        i5y = np.sum(i5y_integrand * length)

        # Emittances
        eq_gemitt_x = (55/(32 * 3**(1/2)) * hbar / electron_volt * clight
                    / mass0 * gamma0**2 * i5x / (i2 - i4x))
        eq_gemitt_y = (55/(32 * 3**(1/2)) * hbar / electron_volt * clight
                    / mass0 * gamma0**2 * i5y / (i2 - i4y))

        # Damping constants
        damping_constant_x_s = r0/3 * gamma0**3 * clight/self.circumference * (i2 - i4x)
        damping_constant_y_s = r0/3 * gamma0**3 * clight/self.circumference * (i2 - i4y)
        damping_constant_zeta_s = r0/3 * gamma0**3 * clight/self.circumference * (2*i2 + i4x + i4y)

        cols = {
            'rad_int_kappax': kappa_x,
            'rad_int_kappay': kappa_y,
            'rad_int_hx': Hx_rad,
            'rad_int_hy': Hy_rad,
            'rad_int_i1x_integrand': i1x_integrand,
            'rad_int_i1y_integrand': i1y_integrand,
            'rad_int_i2_integrand': i2_integrand,
            'rad_int_i3_integrand': i3_integrand,
            'rad_int_i4x_integrand': i4x_integrand,
            'rad_int_i4y_integrand': i4y_integrand,
            'rad_int_i5x_integrand': i5x_integrand,
            'rad_int_i5y_integrand': i5y_integrand,
            'rad_int_kappa0_x': kappa0_x,
            'rad_int_kappa0_y': kappa0_y,
            'rad_int_kappa0': kappa0,
            'rad_int_kappa_x': kappa_x,
            'rad_int_kappa_y': kappa_y,
            'rad_int_kappa': kappa,
        }

        scalars = {
            'rad_int_i1x': i1x,
            'rad_int_i1y': i1y,
            'rad_int_i2': i2,
            'rad_int_i3': i3,
            'rad_int_i4x': i4x,
            'rad_int_i4y': i4y,
            'rad_int_i5x': i5x,
            'rad_int_i5y': i5y,
            'rad_int_eq_gemitt_x': eq_gemitt_x,
            'rad_int_eq_gemitt_y': eq_gemitt_y,
            'rad_int_damping_constant_x_s': damping_constant_x_s,
            'rad_int_damping_constant_y_s': damping_constant_y_s,
            'rad_int_damping_constant_zeta_s': damping_constant_zeta_s,
        }

        out = Table({'name': self.name, 's': self.s, 'length':self.length} | cols)
        out._data.update(scalars)

        if add_to_tw:
            for ncc, cc in cols.items():
                self[ncc] = cc
            self._data.update(scalars)

        return out

def _complete_twiss_init(start, end, init_at, init,
                        line, reverse,
                        x, px, y, py, zeta, delta,
                        alfx, alfy, betx, bety, bets,
                        dx, dpx, dy, dpy, dzeta,
                        mux, muy, muzeta,
                        ax_chrom, bx_chrom, ay_chrom, by_chrom,
                        ddx, ddpx, ddy, ddpy
                        ):

    if isinstance(init, TwissInit) and init_at is not None:
        init.element_name = init_at

    if start is not None or end is not None:
        assert start is not None and end is not None, (
            'start and end must be provided together')
        if init is None:

            assert betx is not None and bety is not None, (
                'betx and bety or init must be provided when start '
                'and end are used')

            init = xt.TwissInit(
                element_name=init_at,
                x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
                betx=betx, alfx=alfx, bety=bety, alfy=alfy, bets=bets,
                dx=dx, dpx=dpx, dy=dy, dpy=dpy, dzeta=dzeta,
                mux=mux, muy=muy, muzeta=muzeta,
                ax_chrom=ax_chrom, bx_chrom=bx_chrom,
                ay_chrom=ay_chrom, by_chrom=by_chrom,
                ddpx=ddpx, ddx=ddx, ddpy=ddpy, ddy=ddy,
                )
        elif isinstance(init, TwissTable):
            init = init.get_twiss_init(at_element=init_at)
        else:
            assert x is None and px is None and y is None and py is None
            assert zeta is None and delta is None
            assert betx is None and alfx is None and bety is None and alfy is None
            assert bets is None
            assert dx is None and dpx is None and dy is None and dpy is None
            assert dzeta is None
            assert mux is None and muy is None and muzeta is None
            assert ax_chrom is None and bx_chrom is None
            assert ay_chrom is None and by_chrom is None
            assert ddpx is None and ddx is None and ddpy is None and ddy is None

    if init is not None and not isinstance(init, str):
        assert isinstance(init, TwissInit)
        init = init.copy() # To avoid changing the one provided
        if init._needs_complete():
            assert isinstance(start, str), (
                'start must be provided as name when an incomplete '
                'init is provided')
            init._complete(line=line,
                    element_name=(init.element_name or start))

        if init.reference_frame is None:
            init.reference_frame = {
                True: 'reverse', False: 'proper', None: None}[reverse]

        if reverse is not None:
            if init.reference_frame == 'proper':
                assert not(reverse), ('`init` needs to be given in the '
                    'proper reference frame when `reverse` is False')
            elif init is not None and init.reference_frame == 'reverse':
                assert reverse is True, ('`init` needs to be given in the '
                    'reverse reference frame when `reverse` is True')

    return init

def _complete_steps_r_matrix_with_default(steps_r_matrix):
    if steps_r_matrix is not None:
        steps_in = steps_r_matrix.copy()
        for nn in steps_in.keys():
            assert nn in list(DEFAULT_STEPS_R_MATRIX.keys()) + ['adapted'], (
                '`steps_r_matrix` can contain only ' +
                ' '.join(DEFAULT_STEPS_R_MATRIX.keys())
            )
        steps_r_matrix = DEFAULT_STEPS_R_MATRIX.copy()
        steps_r_matrix.update(steps_in)
    else:
        steps_r_matrix = DEFAULT_STEPS_R_MATRIX.copy()

    return steps_r_matrix

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

def _str_to_index(line, ele, allow_end_point=True):
    if allow_end_point and ele == '_end_point':
        return len(line._element_names_unique)
    if isinstance(ele, str):
        if ele not in line._element_names_unique:
            raise ValueError(f'Element {ele} not found in line')
        return line._element_names_unique.index(ele)
    else:
        return ele

def _build_sigma_table(Sigma, s=None, name=None):

    res_data = {}
    if s is not None:
        res_data['s'] = s.copy()
    if name is not None:
        res_data['name'] = name.copy()

    # Longitudinal plane is untested

    res_data['sigma_x'] = np.sqrt(Sigma[:, 0, 0])
    res_data['sigma_y'] = np.sqrt(Sigma[:, 2, 2])
    res_data['sigma_zeta'] = np.sqrt(Sigma[:, 4, 4])

    res_data['sigma_px'] = np.sqrt(Sigma[:, 1, 1])
    res_data['sigma_py'] = np.sqrt(Sigma[:, 3, 3])
    res_data['sigma_pzeta'] = np.sqrt(Sigma[:, 5, 5])


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


    return Table(res_data)

def compute_T_matrix_line(line, start, end, particle_on_co=None,
                            steps_t_matrix=None):

    steps_t_matrix = _complete_steps_r_matrix_with_default(steps_t_matrix)

    if particle_on_co is None:
        tw = line.twiss(reverse=False)
        particle_on_co = tw.get_twiss_init(start).particle_on_co

    R_plus = {}
    R_minus = {}
    p_plus = {}
    p_minus = {}

    for kk in ['x', 'px', 'y', 'py', 'zeta', 'delta']:

        p_plus[kk] = particle_on_co.copy()
        setattr(p_plus[kk], kk, getattr(particle_on_co, kk) + steps_t_matrix['d' + kk])
        R_plus[kk] = line.compute_one_turn_matrix_finite_differences(
                            start=start, end=end,
                            particle_on_co=p_plus[kk])['R_matrix']

        p_minus[kk] = particle_on_co.copy()
        setattr(p_minus[kk], kk, getattr(particle_on_co, kk) - steps_t_matrix['d' + kk])
        R_minus[kk] = line.compute_one_turn_matrix_finite_differences(
                            start=start, end=end,
                            particle_on_co=p_minus[kk])['R_matrix']

    TT = np.zeros((6, 6, 6))
    TT[:, :, 0] = 0.5 * (R_plus['x'] - R_minus['x']) / (
        p_plus['x']._xobject.x[0] - p_minus['x']._xobject.x[0])
    TT[:, :, 1] = 0.5 * (R_plus['px'] - R_minus['px']) / (
        p_plus['px']._xobject.px[0] - p_minus['px']._xobject.px[0])
    TT[:, :, 2] = 0.5 * (R_plus['y'] - R_minus['y']) / (
        p_plus['y']._xobject.y[0] - p_minus['y']._xobject.y[0])
    TT[:, :, 3] = 0.5 * (R_plus['py'] - R_minus['py']) / (
        p_plus['py']._xobject.py[0] - p_minus['py']._xobject.py[0])
    TT[:, :, 4] = 0.5 * (R_plus['zeta'] - R_minus['zeta']) / (
        p_plus['zeta']._xobject.zeta[0] - p_minus['zeta']._xobject.zeta[0])
    TT[:, :, 5] = 0.5 * (R_plus['delta'] - R_minus['delta']) / (
        (p_plus['delta']._xobject.ptau[0] - p_minus['delta']._xobject.ptau[0])
        / p_plus['delta']._xobject.beta0[0])

    return TT

def _multiturn_twiss(tw0, num_turns, kwargs):
    tw_curr = tw0
    twisses_to_merge = []
    line = kwargs['line']

    for i_turn in range(num_turns):

        tw_start_turn = tw_curr.rows[0]
        tw_start_turn.name[0] = f'_turn_{i_turn}'
        twisses_to_merge.append(tw_start_turn)
        twisses_to_merge.append(tw_curr)

        if i_turn == num_turns - 1:
            break # need n-1 twisses

        tini1 = tw_curr.get_twiss_init(-1)
        tini1.element_name = tw_curr.name[0]
        tw_curr = twiss_line(**kwargs,
            init=tini1, start=tw_curr.name[0],
            end=line._element_names_unique[-1])

    tw_mt = xt.TwissTable.concatenate(twisses_to_merge)

    return tw_mt

def _add_action_in_res(res, kwargs):
    if isinstance(res, xt.TwissInit):
        return res
    twiss_kwargs = kwargs.copy()
    action = xt.match.ActionTwiss(**twiss_kwargs)
    res._data['_action'] = action
    return res

def get_non_linear_chromaticity(line, delta0_range, num_delta, fit_order=3, **kwargs):

    assert 'method' not in kwargs.keys()
    kwargs['method'] = '4d'

    delta0 = np.linspace(delta0_range[0], delta0_range[1], num_delta)

    twiss = []
    for dd in delta0:
        tw = line.twiss(delta0=dd, **kwargs)
        twiss.append(tw)

    qx = np.array([tw.mux[-1] for tw in twiss])
    qy = np.array([tw.muy[-1] for tw in twiss])
    momentum_compaction_factor = np.array([
        tw.momentum_compaction_factor for tw in twiss])

    poly_qx_fit = np.polyfit(delta0, qx, deg=fit_order)
    poly_qy_fit = np.polyfit(delta0, qy, deg=fit_order)

    out_data = {}

    if fit_order >= 1:
        dqx = poly_qx_fit[-2]
        dqy = poly_qy_fit[-2]
        out_data['dqx'] = dqx
        out_data['dqy'] = dqy

    if fit_order >= 2:
        ddqx = poly_qx_fit[-3] * 2
        ddqy = poly_qy_fit[-3] * 2
        out_data['ddqx'] = ddqx
        out_data['ddqy'] = ddqy

    dnqx = np.array([poly_qx_fit[::-1][ii] * factorial(ii) for ii in range(fit_order+1)])
    dnqy = np.array([poly_qy_fit[::-1][ii] * factorial(ii) for ii in range(fit_order+1)])

    out_data['dnqx'] = dnqx
    out_data['dnqy'] = dnqy

    out_data['delta0'] = delta0
    out_data['qx'] = qx
    out_data['qy'] = qy
    out_data['momentum_compaction_factor'] = momentum_compaction_factor
    out_data['twiss'] = twiss

    out = xt.Table(data = out_data, index='delta0',
            col_names = ['delta0', 'qx', 'qy', 'momentum_compaction_factor'])

    return out

def _merit_function_co_t_rec(x, line, num_turns):
    p = line.build_particles(x=x[0], px=x[1], y=x[2], py=x[3], zeta=x[4], delta=x[5])
    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
    rec = line.record_last_track
    dx = rec.x[0, :] - rec.x[0, 0]
    dpx = rec.px[0, :] - rec.px[0, 0]
    dy = rec.y[0, :] - rec.y[0, 0]
    dpy = rec.py[0, :] - rec.py[0, 0]
    ddelta = rec.delta[0, :] - rec.delta[0, 0]

    out = np.array(list(dx) + list(dpx) + list(dy) + list(dpy) + list(ddelta))
    return out

def _find_closed_orbit_search_t_rev(line, num_turns_search_t_rev=None):

    if num_turns_search_t_rev is None:
        num_turns_search_t_rev = DEFAULT_NUM_TURNS_SEARCH_T_REV

    opt = xt.match.opt_from_callable(partial(_merit_function_co_t_rec,
                        line=line, num_turns=num_turns_search_t_rev),
                        x0=np.array(6*[0.]),
                        steps=[1e-9, 1e-10, 1e-9, 1e-10, 1e-4, 1e-7],
                        tar=np.array(5*num_turns_search_t_rev*[0.]),
                        tols=np.array(5*num_turns_search_t_rev*[1e-10]))
    opt.solve()
    x_sol = opt.get_knob_values()
    particle_on_co = line.build_particles(
        x=x_sol[0], px=x_sol[1], y=x_sol[2], py=x_sol[3], zeta=x_sol[4],
        delta=x_sol[5])

    return particle_on_co


def _reverse_strengths(out):
    ### Same convention as in MAD-X for reversing strengths
    for kk in SIGN_FLIP_FOR_ATTR_REVERSE:
        if kk in out:
            val=out[kk]#avoid passing by setitem
            np.negative(val,val)

def _W_phys2norm(x, px, y, py, zeta, pzeta, W_matrix, co_dict, nemitt_x=None, nemitt_y=None, nemitt_zeta=None):

    # Compute geometric emittances if normalized emittances are provided
    gemitt_x = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_x is None else (
        nemitt_x / co_dict['beta0'] / co_dict['gamma0'])
    gemitt_y = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_y is None else (
        nemitt_y / co_dict['beta0'] / co_dict['gamma0'])
    gemitt_zeta = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_zeta is None else (
        nemitt_zeta / co_dict['beta0'] / co_dict['gamma0'])

    # Prepaing co arrray and gemitt array:
    co = np.array([co_dict['x'], co_dict['px'], co_dict['y'], co_dict['py'],
                  co_dict['zeta'], co_dict['ptau'] / co_dict['beta0']])
    gemitt_values = np.array(
        [gemitt_x, gemitt_x, gemitt_y, gemitt_y, gemitt_zeta, gemitt_zeta])

    # Ensuring consistent dimensions
    for add_axis in range(-1, len(np.shape(x))-len(np.shape(co))):
        co = co[:, np.newaxis]
    for add_axis in range(-1, len(np.shape(x))-len(np.shape(gemitt_values))):
        gemitt_values = gemitt_values[:, np.newaxis]

    # substracting closed orbit
    XX = np.array([x, px, y, py, zeta, pzeta])
    XX -= co

    # Apply the inverse transformation matrix
    W_inv = np.linalg.inv(W_matrix)

    if len(np.shape(XX)) == 3:
        XX_norm = np.dot(W_inv, XX.reshape(6, x.shape[0]*x.shape[1]))
        XX_norm = XX_norm.reshape(6, x.shape[0], x.shape[1])
    else:
        XX_norm = np.dot(W_inv, XX)

    # Normalize the coordinates with the geometric emittances
    XX_norm /= np.sqrt(gemitt_values)

    return XX_norm

def _add_strengths_to_twiss_res(twiss_res, line):
    tt = line.get_table(attr=True).rows[list(twiss_res.name)]
    for kk in (NORMAL_STRENGTHS_FROM_ATTR + SKEW_STRENGTHS_FROM_ATTR
                + OTHER_FIELDS_FROM_ATTR + OTHER_FIELDS_FROM_TABLE):
        twiss_res._col_names.append(kk)
        twiss_res._data[kk] = tt[kk].copy()


