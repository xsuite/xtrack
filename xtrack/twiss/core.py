# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from contextlib import ExitStack

from .twiss_init import TwissInit
from .element_indexing import _str_to_index
from .twiss_table import TwissTable
from .input_normalization import (
    _normalize_twiss_inputs,
)
from .base_computation import (
    _clear_twiss_init_inputs,
    _complete_init_for_base_twiss,
    _compute_base_twiss,
    _compute_base_twiss_after_explicit_init_completion,
)
from .chromatic_functions import trapz
from .open_table_composition import (
    _combine_loop_around_twiss_tables,
    _combine_init_inside_range_twiss_tables,
)
from .open_propagation import (
    _plan_loop_around_twiss_parts,
    _plan_init_inside_range_twiss_parts,
)
from .computation_plan import _plan_twiss_computation
from .line_context import (
    _prepare_twiss_line_context,
)
from .finalize import _finalize_twiss_result
from .constants import (
    VARS_FOR_TWISS_INIT_GENERATION,
)

import xtrack as xt  # To avoid circular imports


def twiss_line(line, particle_ref=None, method=None,
        particle_on_co=None, R_matrix=None, W_matrix=None,
        delta0=None, zeta0=None, zeta_shift=None,
        nemitt_x=None, nemitt_y=None, step_W_sigma=None,
        delta_disp=None, delta_chrom=None, zeta_disp=None,
        co_guess=None, steps_R_matrix=None,
        co_search_settings=None,
        continue_on_closed_orbit_error=None,
        values_at_element_exit=None,
        radiation_method=None,
        radiation_analysis=None,
        radiation_integrals=None,
        spin=None,
        polarization_analysis=None,
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
        chrom=None,
        coupling_edw_teng=False,
        init_at=None,
        x=None, px=None, y=None, py=None, zeta=None, delta=None,
        betx=None, alfx=None, bety=None, alfy=None, bets=None,
        dx=None, dpx=None, dy=None, dpy=None, dzeta=None,
        mux=None, muy=None, muzeta=None,
        ax_chrom=None, bx_chrom=None, ay_chrom=None, by_chrom=None,
        ddx=None, ddpx=None, ddy=None, ddpy=None,
        spin_x=None, spin_y=None, spin_z=None,
        zero_at=None,
        co_search_at=None,
        include_collective=False,
        disable_apertures=None,
        _continue_if_lost=None,
        _keep_tracking_data=None,
        _keep_initial_particles=None,
        _initial_particles=None,
        _ebe_monitor=None,
        only_markers=None,
        # Deprecated
        at_s=None,
        at_elements=None,
        compute_chromatic_properties=None,
        r_sigma=None,
        freeze_longitudinal=None,
        freeze_energy=None,
        polarization=None,
        eneloss_and_damping=None,
        steps_r_matrix=None,
    ):
    """
    Compute the Twiss parameters of the beam line. If no initial conditions
    are provided, the periodic solution is computed.

    Parameters
    ----------

    method : {'6d', '4d'}, optional
        Method to be used for the computation. If '6d' the full 6D
        normal form is used. If '4d' the 4D normal form is used.
    start : str, optional
        Name of the element at which the computation starts. If not provided,
        the periodic solution is computed. Initial conditions must be provided if
        ``start`` is provided.
    end : str, optional
        Name of the element at which the computation stops.
    init : TwissInit object, optional
        Initial values for the Twiss parameters. If ``init="periodic"`` is
        passed, the periodic solution for the selected range is computed.
        Instead of passing ``init``, initial conditions can be provided directly
        as keyword arguments, e.g. ``line.twiss(betx=1, bety=2, x=1e-3)``.
        Accepted fields: ``x``, ``px``, ``y``, ``py``, ``zeta``, ``delta``, ``betx``,
        ``alfx``, ``bety``, ``alfy``, ``bets``, ``dx``, ``dpx``, ``dy``, ``dpy``,
        ``dzeta``, ``mux``, ``muy``, ``muzeta``, ``ax_chrom``, ``bx_chrom``,
        ``ay_chrom``, ``by_chrom``, ``ddx``, ``ddpx``, ``ddy``, ``ddpy``, ``spin_x``,
        ``spin_y``, ``spin_z``.
    init_at : str, optional
        Element name at which the initial conditions are defined. If not provided,
        the initial conditions are defined at ``start``.
    delta0 : float, optional
        Closed-orbit ``delta`` at the start of the beam line, used when solving
        the closed orbit in ``method='4d'``. Mutually exclusive with ``zeta0``.
        Cannot be used in 6d mode.
    zeta0 : float, optional
        Closed-orbit ``zeta`` at the start of the beam line, used when solving
        the closed orbit in ``method='4d'``. Mutually exclusive with ``delta0``.
        Cannot be used in 6d mode.
    zeta_shift : float, optional
        Offset applied to ``zeta`` during closed-orbit search (closed orbit is
        found for ``zeta[out] = zeta[in] - zeta_shift``). Default is 0.
    co_guess : xpart.Particles or dict, optional
        Initial guess for the closed orbit. If not provided, zero is assumed.
    co_search_at : str, optional
        Element name at which the closed orbit is searched. If not provided,
        the closed orbit is searched at the start of the line.
    strengths : bool, optional
        If True, the strengths of the magnetic elements are added to the table.
    include_collective : bool, optional
        If True, keep collective elements active during the twiss computation.
        Default is False.
    disable_apertures : bool, optional
        If True (default), aperture checks on tracked particles are disabled
        while computing twiss.
    reverse : bool, optional
        If True, the output is computed in the reversed reference frame, i.e.
        s = -s, x = -x, y = y, zeta = -zeta, px=px, py=-py, delta=delta.
        Default is False.
    chrom : bool, optional
        If True, compute chromatic properties. Default is None, which means
        chromatic properties are computed only for the periodic solution, but
        not for open twiss.
    radiation_analysis : bool, optional
        If True, the energy loss, radiation damping constants, and equilibrium
        emittances are computed. Default is False.
    radiation_method : {'full', 'kick_as_co', 'scale_as_co'}, optional
        Method to be used for the computation of twiss parameters in the presence
        of radiation. If 'full' the method described in E. Forest, "From tracking
        code to analysis" is used. If 'kick_as_co' all particles receive the same
        radiation kicks as the closed orbit. If 'scale_as_co' all particles
        momenta are scaled by radiation as much as the closed orbit.
    radiation_integrals : bool, optional
        If True, the radiation integrals are computed.
    spin : bool, optional
        If True, for periodic twiss compute spin closed solution (n0);
        for open twiss, propagate spin components.
    polarization_analysis : bool, optional
        If True, compute quantititis related to spin polarization.
    delta_chrom : float, optional
        Momentum deviation for the chromaticity computation.
    steps_R_matrix : dict, optional
        Steps to be used for the finite difference computation of the R matrix.
        If not provided, the default values are used.
    matrix_responsiveness_tol : float, optional
        Tolerance to be used to check the responsiveness of the R matrix.
        If not provided, the default value is used.
    matrix_stability_tol : float, optional
        Tolerance to be used to check the stability of the R matrix.
        If not provided, the default value is used.
    step_W_sigma : float, optional.
        Deviation in sigmas used for the propagation of the W matrix.
    nemitt_x : float, optional
        Horizontal emittance assumed for the computation of the deviation
        used for the propagation of the W matrix.
    nemitt_y : float, optional
        Vertical emittance assumed for the computation of the deviation
        used for the propagation of the W matrix.
    coupling_edw_teng : bool, optional
        If True, Edwards-Teng coupling quantities are computed. Default is
        False.
    zero_at : str, optional
        Element name at which the s coordinate and the phase advances are set to
        zero.
    compute_R_element_by_element : bool, optional
        If True, the element-by-element R matrices are computed and stored in
        the output table. Default is False.
    num_turns: int, optional
        If specified the periodic solution and the twiss table are computed
        on multiple turns.
    search_for_t_rev : bool, optional
        If True, the revolution period is searched for, otherwise the revolution
        period computed from the line length is assumed.
    num_turns_search_t_rev : int, optional
        Number of turns used for the search of the revolution period. Used only
        if ``search_for_t_rev`` is True.
    symplectify : bool, optional
        If True, the R matrix is symplectified before computing the linear normal
        form. Default is False.
    particle_on_co : xpart.Particles, optional
        Particle on the closed orbit. If not provided, the closed orbit is searched for.
    co_search_settings : dict, optional
        Settings to be used by the optimizer for the closed orbit search. If not
        provided, the default values are used.
    R_matrix : np.ndarray, optional
        R matrix to be used for the computation. If not provided, the R matrix is
        computed using finite differences.
    W_matrix : np.ndarray, optional
        W matrix to be used for the computation. If not provided, the W matrix is
        computed from the R matrix.
    use_full_inverse : bool, optional
        If True, the full inverse of the W matrix is used. If False, the inverse is
        computed from the symplectic condition.

    Returns
    -------

    twiss : xtrack.TwissTable

    Notes
    -----

    Output fields depending on selected options (for detailed definitions and
    explanations refer to the Xsuite Physics Guide (https://xsuite.readthedocs.io/en/latest/physicsguide.html):
    Fields marked as "ebe" are element-by-element quantities.

    Default output fields:
        - `name`: element name, when repeated elements are present "::1", "::2", ...
          suffixes are added to make the names unique. (ebe)
        - `env_name`: environment name of the element, i.e. name without suffix
          for repeated elements. (ebe)
        - `s`: element position [m] (ebe)
        - `x`, `px`, `y`, `py`, `zeta`, `delta`, `ptau`: coordinates
          of the closed orbit for the periodic twiss and of the beam trajectory
          for the open twiss. (ebe)
        - `betx`, `bety`, `alfx`, `alfy`, `gamx`, `gamy`: Twiss parameters.
          In the presence of linear coupling, these are respectively `betx1`,
          `bety2`, `alfx1`, `alfy2`, `gamx1`, `gamy2` in the Mais-Ripken sense. (ebe)
        - `dx`, `dpx`, `dy`, `dpy`: dispersion functions (ebe)
        - `ddx`, `ddpx`, `ddy`, `ddpy`: second-order dispersion functions (ebe)
        - `dx_zeta`, `dpx_zeta`, `dy_zeta`, `dpy_zeta`: crab dispersion functions (ebe)
        - `bets0`: longitudinal beta function at start ring.
        - `W_matrix`: linear normal-form matrix. (ebe)
        - `kin_px`, `kin_py`, `kin_ps`: kinetic momenta (different from `px`, `py`
          which are canonical momenta). (ebe)
        - `kin_xp`, `kin_yp`: transverse slopes kin_px/kin_ps, kin_py/kin_ps. (ebe)
        - `mux`, `muy`, `muzeta`: phase advances in units of 2 pi. (ebe)
        - `nux`, `nuy`, `nuzeta`: damping exponents. (ebe)
        - `betx1`, `bety1`, `betx2`, `bety2`, `alfx1`, `alfy1`, `alfx2`,
          `alfy2`, `gamx1`, `gamy1`, `gamx2`, `gamy2`: Mais-Ripken coupled optics
          functions (ebe)
        - `wx_chrom`, `wy_chrom`, `bx_chrom`, `by_chrom`, `ax_chrom`, `ay_chrom`:
          chromatic functions, see physics guide for definitions (ebe)
        - `particle_on_co`: particle on closed orbit or reference trajecory, placed
          at the first element in the selected range.
        - `reference_frame`: reference frame used for the output (can be `proper`
          or `reversed`)
        - `periodic`: True if periodic twiss, False if open twiss
        - `method`: method used for the computation (`4d` or `6d`)
    Output fields present only for periodic twiss:
        - `qx`, `qy`: transverse tunes
        - `qs`: synchrotron tune (present only when method is `6d`)
        - `dqx`, `dqy`: linear chromaticities
        - `ddqx`, `ddqy`: second-order chromaticities
        - `line_length`: length of the beam line
        - `p0c`, `gamma0`, `beta0`: reference momentum and relativistic factors
        -  `t_rev0`: reference revolution period
        - `slip_factor`: slip factor, i.e. eta = -(dfrev / frev) / ddelta
        - `momentum_compaction_factor`: momentum compaction factor (d C / C) / ddelta
          where C the closed orbit path length
        - `slip_factor_dzeta_ddelta`: d (zeta) / ddelta
        - `bets0`: longitudinal beta function at start of the ring.
        - `c_minus`, `c_minus_re_0`, `c_minus_im_0`: closest tune approach coefficient
          (absolute, real and imaginary parts). See physics guide for definitions.
        - `c_minus_re`, `c_minus_im`, `c_r1`, `c_r2`, `c_phi1`, `c_phi2`:
          element-by-element coupling coefficients. See physics guide for
          definitions. (ebe)
        - `R_matrix`: one-turn transfer matrix
        - `steps_R_matrix`: steps used for the finite-difference computation of
          the R matrix
        - `R_matrix_ebe`: element-by-element transfer matrices, from the start of
          the line to the selected element. (ebe)
        - `eigenvalues`, `rotation_matrix`: additional linear-normal-form data
        - `dmux`, `dmuy`: phase-advance derivatives vs delta
        - `dzeta`: longitudinal dispersion vs delta
    Output fields present when `strengths=True` (or `radiation_integrals=True`):
        - `k0l`–`k5l`, `k0sl`–`k5sl`: normal/skew multipole integrated strengths
        - `angle`, `rot_s_rad`, `hkick`, `vkick`, `ks`, `bs`, `length`,
          `element_type`, `isthick`, `parent_name`, `prototype`: element properties
    Output fields present when `radiation_analysis=True`:
        - `energy_loss`: energy loss per turn [eV]
        - `damping_constants_turns`, `damping_constants_s`: damping constants in
          1/turn or 1/s.
        - `partition_numbers`: radiation partition numbers
        - `eq_gemitt_x`, `eq_gemitt_y`, `eq_gemitt_zeta`: equilibrium geometric
          emittances.
        - `eq_nemitt_x`, `eq_nemitt_y`, `eq_nemitt_zeta`: equilibrium normalized
          emittances.
    Output fields present when `radiation_integrals=True`:
        - `rad_int_i1x`, `rad_int_i1y`, `rad_int_i2`, `rad_int_i3`, `rad_int_i4`,
          `rad_int_i4x`, `rad_int_i4y`, `rad_int_i5x`, `rad_int_i5y`: radiation
          integrals (see physics guide for definitions)
        - `rad_int_i1x_integrand`, `rad_int_i1y_integrand`, `rad_int_l2_integrand`,
          `rad_int_i3_integrand`, `rad_int_i4_integrand`, `rad_int_i4x_integrand`,
          `rad_int_i4y_integrand`, `rad_int_i5x_integrand`, `rad_int_i5y_integrand`:
          integrands of the radiation integrals (ebe)
        - `rad_int_curly_hx`, `rad_int_curly_hy`: curly-H functions (see physics
          guide for definitions) (ebe)
        - `rad_int_eq_gemitt_x`, `rad_int_eq_gemitt_y`: geometric equilibrium
          emittances from radiation integrals.
        - `rad_int_energy_loss`: energy loss per turn from radiation integrals [eV]
        - `rad_int_sigma_delta`: equilibrium momentum spread from radiation
          integrals.
        - `rad_int_damping_constant_x_s`, `rad_int_damping_constant_y_s`,
          `rad_int_damping_constant_zeta_s`: damping constants from radiation
          integrals
        - `rad_int_kappa0_x`, `rad_int_kappa0_y`, `rad_int_kappa0`: reference
          curvature used in the computation (ebe)
        - `rad_int_kappa_x`, `rad_int_kappa_y`, `rad_int_kappa`: closed orbit
          curvature used in the computation (ebe)
        - `rad_int_iv_x`, `rad_int_iv_y`, `rad_int_iv_z`: velocity direction
          cosines (ebe)
    Output fields present when `spin=True`:
        - `spin_x`, `spin_y`, `spin_z`: spin components of the closed spin solution
          (n0) for periodic twiss, or propagated spin components for open twiss. (ebe)
    Output fields present when `polarization_analysis=True`:
        - `spin_tune_fractional`: fractional spin tune
        - `spin_polarization_eq`: equilibrium polarization in the linear approximation
        - `spin_polarization_inf_no_depol`: infinite-time polarization without
          depolarization effects
        - `spin_t_pol_buildup_s`: polarization buildup time in seconds
        - `spin_t_pol_component_s`: polarization component of the buildup time in seconds
        - `spin_t_depol_component_s`: depolarization component of the buildup time in seconds
        - `spin_n_matrix`: invariant spin field matrix in local frame (ebe)
        - `spin_eigenvectors`: eigenvectors of the spin one-turn matrix (ebe)
        - `spin_dn_ddelta_x`, `spin_dn_ddelta_y`, `spin_dn_ddelta_z`: derivatives of
          the invariant spin field w.r.t. delta (ebe)
        - `spin_n0_iv`, `spin_n0_ib`: projections of equilibrium spin along the
          closed orbit velocity and magnetic field directions (ebe)
        - `spin_int_kappa3_n0_ib`, `spin_int_kappa3_dn_ddelta_ib`,
          `spin_int_kappa3_11_18_dn_ddelta_sq`: integrals involved in polarization
          computations
    Output fields present when `coupling_edw_teng=True`:
        - `r11_edw_teng`, `r12_edw_teng`, `r21_edw_teng`, `r22_edw_teng`:
          Elements of the Edwards-Teng coupling matrix (ebe)
    Output fields present when `search_for_t_rev=True`:
        - `t_rev`: measured revolution period [s]

    """
    normalized_kwargs, input_kwargs = _normalize_twiss_inputs(
        twiss_kwargs=locals().copy(), twiss_init_cls=TwissInit)

    with ExitStack() as twiss_context:
        prepared_data = _prepare_twiss_line_context(
            twiss_context=twiss_context,
            data=normalized_kwargs)
        prepared_data['kwargs'] = normalized_kwargs
        return _compute_twiss_with_prepared_line_context(
            data=prepared_data, input_kwargs=input_kwargs)


def _compute_twiss_with_prepared_line_context(data, input_kwargs):

    data = data.copy()
    data['zero_at_requested'] = data['zero_at']
    data['zero_at'] = None

    computation_plan = _plan_twiss_computation(data, data['init'])
    data['twiss_computation_plan'] = computation_plan
    twiss_res = _compute_composed_twiss_before_init_completion(
        data=data, computation_plan=computation_plan)

    if twiss_res is None:
        data['init'], data['completed_init'] = _complete_init_for_base_twiss(
            data=data)
        _clear_twiss_init_inputs(data)

        twiss_res = _compute_composed_twiss_after_init_completion(
            data=data, computation_plan=computation_plan)

        if twiss_res is None:
            twiss_res = _compute_base_twiss_after_explicit_init_completion(
                data=data)

    return _finalize_twiss_result(
        twiss_res, input_kwargs, zero_at=data['zero_at_requested'])


def _compute_composed_twiss_before_init_completion(
        data, computation_plan):

    composed_twiss_res = None
    route = computation_plan.route

    if route in (
            'periodic_one_turn_from_start', 'open_one_turn_from_start'):
        call_kwargs = data.copy()
        composed_twiss_res = _compute_one_turn_twiss_from_plan(
            kwargs=call_kwargs,
            computation_plan=computation_plan,
        )

    elif route == 'full_periodic_range':
        call_kwargs = data.copy()
        periodic_kwargs = _prepare_kwargs_for_full_periodic_twiss(call_kwargs)
        full_periodic_init = _acquire_full_periodic_twiss_init(
            kwargs=periodic_kwargs,
            acquisition_plan=computation_plan.init_acquisition,
            start=data['start'],
        )
        composed_twiss_res = _propagate_full_periodic_init_over_range(
            kwargs=call_kwargs,
            init=full_periodic_init,
            open_plan=computation_plan.open_propagation,
        )
        if data['zero_at_requested'] is None:
            composed_twiss_res.zero_at(data['start'])

    elif route != 'base':
        raise RuntimeError(f'Unexpected Twiss computation route: {route}')

    return composed_twiss_res


def _compute_composed_twiss_after_init_completion(
        data, computation_plan):

    composed_twiss_res = None
    open_plan = _open_propagation_plan_after_init_completion(
        data=data, computation_plan=computation_plan)

    if open_plan is not None:
        call_kwargs = data.copy()

        if open_plan.crosses_line_boundary:
            composed_twiss_res = _handle_loop_around(
                call_kwargs, open_plan=open_plan)

        elif not open_plan.init_is_at_boundary:
            composed_twiss_res = _handle_init_inside_range(
                call_kwargs, open_plan=open_plan)

    return composed_twiss_res


def _open_propagation_plan_after_init_completion(data, computation_plan):

    init = data['init']
    open_plan = None

    if not data['periodic'] and not isinstance(init, str):
        open_plan = computation_plan.open_propagation

    return open_plan


def _propagate_full_periodic_init_over_range(kwargs, init, open_plan):

    range_kwargs = kwargs.copy()
    range_kwargs['init'] = init
    range_kwargs['init_at'] = None

    if open_plan.crosses_line_boundary:
        return _handle_loop_around(range_kwargs, open_plan=open_plan)

    if not open_plan.init_is_at_boundary:
        return _handle_init_inside_range(range_kwargs, open_plan=open_plan)

    assert len(open_plan.pieces) == 1
    return _compute_twiss_segment_for_piece(
        kwargs=range_kwargs, piece=open_plan.pieces[0], init=init)




def _compute_twiss_segment_for_piece(kwargs, piece, init):

    return _compute_twiss_segment(
        kwargs, start=piece.start, end=piece.end, init=init)


def _handle_loop_around(kwargs, open_plan=None):

    kwargs = kwargs.copy()

    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')

    line = kwargs['line']
    reverse = kwargs['reverse']

    if open_plan is not None and len(open_plan.pieces) == 3:
        twiss_tables, completed_init = _execute_three_piece_loop_around_plan(
            kwargs=kwargs, open_plan=open_plan, init=init)
        return _combine_loop_around_twiss_tables(
            twiss_tables, init, completed_init)

    plan = _plan_loop_around_twiss_parts(
        line=line, start=start, end=end, init=init, reverse=reverse,
        open_plan=open_plan)
    tw1, tw2, completed_init = _execute_loop_around_twiss_plan(
        kwargs=kwargs, plan=plan, init=init)

    return _combine_loop_around_twiss_tables(
        [tw1, tw2], init, completed_init)


def _execute_three_piece_loop_around_plan(kwargs, open_plan, init):

    first_piece, second_piece, third_piece = open_plan.pieces

    if first_piece.role == 'start_to_init':
        first_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=first_piece, init=init)
        second_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=second_piece, init=init)
        first_side_table = _combine_init_inside_range_twiss_tables(
            first_table, second_table, init)
        boundary_init = second_table.get_twiss_init('_end_point')
        boundary_init.element_name = third_piece.start
        third_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=third_piece, init=boundary_init)
        loop_tables = (first_side_table, third_table)
        completed_init = first_side_table.completed_init

    elif first_piece.role == 'start_to_line_boundary':
        second_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=second_piece, init=init)
        third_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=third_piece, init=init)
        second_side_table = _combine_init_inside_range_twiss_tables(
            second_table, third_table, init)
        boundary_init = second_table.get_twiss_init(second_piece.start)
        boundary_init.element_name = first_piece.end
        first_table = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=first_piece, init=boundary_init)
        loop_tables = (first_table, second_side_table)
        completed_init = second_side_table.completed_init

    else:
        raise RuntimeError('Unexpected three-piece loop-around Twiss plan')

    return loop_tables, completed_init


def _execute_loop_around_twiss_plan(kwargs, plan, init):

    if plan.init_piece_role == 'first_table_piece':
        tw1 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.first_table_piece, init=init)
        twini_2 = tw1.get_twiss_init(at_element=plan.transfer_init_at)
        twini_2.element_name = plan.transfer_init_element_name
        tw2 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.second_table_piece, init=twini_2)
        completed_init = tw1.completed_init
    elif plan.init_piece_role == 'second_table_piece':
        tw2 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.second_table_piece, init=init)
        twini_1 = tw2.get_twiss_init(at_element=plan.transfer_init_at)
        twini_1.element_name = plan.transfer_init_element_name
        tw1 = _compute_twiss_segment_for_piece(
            kwargs=kwargs, piece=plan.first_table_piece, init=twini_1)
        completed_init = tw2.completed_init
    else:
        raise RuntimeError('Unexpected loop-around Twiss plan init piece')

    return tw1, tw2, completed_init


def _handle_init_inside_range(kwargs, open_plan=None):

    kwargs = kwargs.copy()
    line = kwargs['line']
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    init = kwargs.pop('init')
    reverse = kwargs.pop('reverse')

    _assert_init_inside_range_is_supported(
        line=line, start=start, end=end, init=init, reverse=reverse)

    plan = _plan_init_inside_range_twiss_parts(
        line=line, start=start, end=end, init=init, reverse=reverse,
        open_plan=open_plan)
    tw1, tw2 = _execute_init_inside_range_twiss_plan(
        kwargs=kwargs, plan=plan, init=init, reverse=reverse)

    return _combine_init_inside_range_twiss_tables(tw1, tw2, init)


def _assert_init_inside_range_is_supported(line, start, end, init, reverse):

    ele_name_init = init.element_name
    ele_init = line.get(ele_name_init)
    if isinstance(ele_init, xt.Replica):
        ele_init = ele_init.resolve()
    if not isinstance(ele_init, xt.Marker):
        raise ValueError(
            'The element at the initial position is not a Marker. '
            'This is not yet supported')

    if reverse:
        assert _str_to_index(line, ele_name_init) <= _str_to_index(line, start)
        assert _str_to_index(line, ele_name_init) >= _str_to_index(line, end)
    else:
        assert _str_to_index(line, ele_name_init) >= _str_to_index(line, start)
        assert _str_to_index(line, ele_name_init) <= _str_to_index(line, end)


def _execute_init_inside_range_twiss_plan(kwargs, plan, init, reverse):

    return tuple(
        _compute_twiss_segment(
            kwargs,
            start=piece.start,
            end=piece.end,
            init=init,
            reverse=reverse)
        for piece in plan.pieces)


def _compute_twiss_segment(kwargs, **overrides):

    segment_kwargs = _kwargs_for_preflighted_twiss_segment(kwargs)
    segment_kwargs.update(overrides)

    return _compute_base_twiss(segment_kwargs)


def _kwargs_for_preflighted_twiss_segment(kwargs):

    segment_kwargs = kwargs.copy()
    segment_kwargs['disable_apertures'] = False
    segment_kwargs['freeze_longitudinal'] = False
    segment_kwargs['freeze_energy'] = False
    segment_kwargs['at_s'] = None

    return segment_kwargs




def _prepare_kwargs_for_full_periodic_twiss(kwargs):

    kwargs = kwargs.copy()
    kwargs.pop('init')
    kwargs.pop('start')
    kwargs.pop('end')
    kwargs.pop('init_at')

    return kwargs


def _acquire_full_periodic_twiss_init(kwargs, acquisition_plan, start):
    """Compute the full periodic Twiss and extract the requested init."""

    assert acquisition_plan.source == 'full_periodic_solution'
    assert acquisition_plan.scope == 'full_line'
    assert acquisition_plan.computation_direction == 'forward'

    tw = _compute_twiss_segment(kwargs) # Periodic twiss of the full line

    return tw.get_twiss_init(acquisition_plan.init_at or start)


def _compute_one_turn_twiss_from_plan(kwargs, computation_plan):

    kwargs = kwargs.copy()
    kwargs.pop('start')
    route = computation_plan.route
    propagation_plan = computation_plan.one_turn_propagation

    if route == 'periodic_one_turn_from_start':
        return _compute_periodic_one_turn_twiss_from_start(
            kwargs=kwargs, plan=propagation_plan)

    if route == 'open_one_turn_from_start':
        return _compute_open_one_turn_twiss_from_start(
            kwargs=kwargs, plan=propagation_plan)

    raise RuntimeError(f'Unexpected one-turn Twiss route: {route}')


def _compute_periodic_one_turn_twiss_from_start(kwargs, plan):

    tw = _compute_twiss_segment(kwargs)
    t1 = tw.rows[plan.start:]
    t2 = tw.rows[:plan.start]
    out = xt.TwissTable.concatenate([t1, t2])
    out.zero_at(out.name[0])
    out.name[-1] = '_end_point'
    out['periodic'] = True
    out['completed_init'] = tw.completed_init
    return out


def _compute_open_one_turn_twiss_from_start(kwargs, plan):

    kwargs = kwargs.copy()
    kwargs.pop('end')

    t1o = _compute_twiss_segment_for_piece(
        kwargs=kwargs, piece=plan.first_piece, init=kwargs['init'])
    init_part2 = t1o.get_twiss_init(plan.transfer_init_at)
    init_part2.element_name = plan.transfer_init_element_name

    for kk in VARS_FOR_TWISS_INIT_GENERATION:
        kwargs.pop(kk, None)
    kwargs.pop('init')
    t2o = _compute_twiss_segment_for_piece(
        kwargs=kwargs, piece=plan.second_piece, init=init_part2)
    # remove repeated element
    t2o = t2o.rows[:-1]
    t2o.name[-1] = '_end_point'
    out = xt.TwissTable.concatenate([t1o, t2o])
    out['completed_init'] = t1o.completed_init
    return out
