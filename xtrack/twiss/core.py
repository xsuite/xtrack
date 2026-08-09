# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from contextlib import ExitStack

from .twiss_table import TwissTable
from .input_normalization import (
    _normalize_twiss_inputs,
    _normalize_twiss_inputs_after_line_context,
)
from .chromatic_functions import trapz
from .line_context import (
    _prepare_twiss_line_context,
)
from .propagation import _propagate_twiss_from_init
from .base_result import (
    _add_periodic_solution_data_to_base_twiss,
    _add_chromatic_functions_to_twiss_result,
    _add_radiation_analysis_to_twiss_result,
    _apply_4d_longitudinal_result_convention,
    _set_twiss_result_values_at,
    _add_strengths_and_radiation_integrals_to_twiss_result,
    _add_spin_polarization_to_twiss_result,
    _add_edwards_teng_coupling_to_twiss_result,
    _add_base_twiss_metadata,
    _reverse_twiss_result_if_needed,
    _add_measured_revolution_period_if_requested,
    _extend_base_twiss_to_multiple_turns,
    _select_twiss_result_at_elements,
    _add_periodicity_and_completed_init_to_twiss_result,
    _align_open_twiss_phases_with_init,
)
from .twiss_init import (
    TwissInit,
    _build_twiss_init_from_inputs,
    _clear_twiss_init_input_fields,
    _compute_periodic_twiss_init,
)
from .element_indexing import _str_to_index
from .finalize import _finalize_twiss_result

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
    # Normalize the public API once before entering temporary line contexts.
    normalized_kwargs, input_kwargs = _normalize_twiss_inputs(
        twiss_kwargs=locals().copy(), twiss_init_cls=TwissInit)

    with ExitStack() as twiss_context:
        # For the twiss calculation we need to alter line.config and
        # line.tracker.track_flags. This context manager takes care of restoring
        # the original values after the twiss calculation is done or in case of
        # an exception.

        # Configure line.config and line.tracker.track_flags
        data = _prepare_twiss_line_context(
            twiss_context=twiss_context,
            data=normalized_kwargs)
        data['kwargs'] = normalized_kwargs

        # Further input normalization
        data = _normalize_twiss_inputs_after_line_context(data)
        data['zero_at_requested'] = data['zero_at']
        data['zero_at'] = None

        # Determine the route (periodic / open / periodic_one_turn_custom_start
        # / open_one_turn_custom_start / open_init_from_full_periodic)
        route = _select_twiss_route(data)

        if route == 'periodic':
            # Standard periodic Twiss, for the full line or a closed range.
            data['completed_init'] = data['init']
            _clear_twiss_init_input_fields(data)
            data.update(_compute_periodic_twiss_init(data))
            twiss_res = _compute_base_twiss(data)

        elif route == 'open':
            # Standard open Twiss from supplied init data.
            data['init'], data['completed_init'] = (
                _build_twiss_init_from_inputs(data))
            _clear_twiss_init_input_fields(data)

            crosses_line_boundary, init_is_at_boundary = (
                _get_open_twiss_range_flags(data))
            if not crosses_line_boundary and init_is_at_boundary:
                twiss_res = _compute_base_twiss(data)
            else:
                twiss_res = _handle_init_inside_range_and_line_wrap(
                    data, crosses_line_boundary)

        elif route == 'periodic_one_turn_custom_start':
            # Compute a full periodic table, then rotate it to the requested start.
            requested_start = data['start']
            one_turn_kwargs = data.copy()
            one_turn_kwargs['start'] = None
            one_turn_kwargs['completed_init'] = one_turn_kwargs['init']
            _clear_twiss_init_input_fields(one_turn_kwargs)
            one_turn_kwargs.update(
                _compute_periodic_twiss_init(one_turn_kwargs))

            full_twiss = _compute_base_twiss(one_turn_kwargs)
            first_part = full_twiss.rows[requested_start:]
            second_part = full_twiss.rows[:requested_start]
            twiss_res = TwissTable.concatenate([first_part, second_part])
            twiss_res.zero_at(twiss_res.name[0])
            twiss_res.name[-1] = '_end_point'
            twiss_res['periodic'] = True
            twiss_res['completed_init'] = full_twiss.completed_init

        elif route == 'open_one_turn_custom_start':
            # A start without an end requests a wrapped open range of one turn.
            data['end'] = data['start']
            data['init'], data['completed_init'] = (
                _build_twiss_init_from_inputs(data))
            _clear_twiss_init_input_fields(data)
            twiss_res = _handle_init_inside_range_and_line_wrap(
                data,
                crosses_line_boundary=True,
                one_turn_from_start=True)

        elif route == 'open_init_from_full_periodic':
            # Compute a forward full-line periodic table and take the open-range
            # init from the requested location.
            periodic_kwargs = data.copy()
            periodic_kwargs.update(
                init=None,
                start=None,
                end=None,
                init_at=None,
                periodic=True,
                periodic_mode='periodic',
                completed_init=None,
            )
            _clear_twiss_init_input_fields(periodic_kwargs)
            periodic_kwargs.update(
                _compute_periodic_twiss_init(periodic_kwargs))
            full_periodic_twiss = _compute_base_twiss(periodic_kwargs)

            data['init'] = full_periodic_twiss.get_twiss_init(
                data['init_at'] or data['start'])
            data['init_at'] = None
            data['init'], data['completed_init'] = (
                _build_twiss_init_from_inputs(data))
            _clear_twiss_init_input_fields(data)

            crosses_line_boundary, init_is_at_boundary = (
                _get_open_twiss_range_flags(data))
            if not crosses_line_boundary and init_is_at_boundary:
                twiss_res = _compute_base_twiss(data)
            else:
                twiss_res = _handle_init_inside_range_and_line_wrap(
                    data, crosses_line_boundary)
            if data['zero_at_requested'] is None:
                twiss_res.zero_at(data['start'])

        else:
            raise RuntimeError(f'Unexpected Twiss route: {route}')

        # All table-producing routes share the same public result finalization.
        return _finalize_twiss_result(
            twiss_res, input_kwargs, zero_at=data['zero_at_requested'])


def _compute_base_twiss(data, **overrides):
    """Propagate from a concrete init and finish one non-composed result."""

    data = data.copy()
    # The public line context is already active. These options only control
    # context setup and must not be applied again for individual segments.
    data['disable_apertures'] = False
    data['freeze_longitudinal'] = False
    data['freeze_energy'] = False
    data['at_s'] = None
    data.update(overrides)

    if 'init' in overrides and 'completed_init' not in overrides:
        data['completed_init'] = data['init'].copy()
    assert isinstance(data['init'], TwissInit), (
        '_compute_base_twiss requires a concrete TwissInit')

    if data['reverse']:
        if data['start'] is not None and data['end'] is not None:
            assert (_str_to_index(data['line'], data['start'])
                    >= _str_to_index(data['line'], data['end'])), (
                'start must be smaller than end in reverse mode')
        data['start'], data['end'] = data['end'], data['start']
    elif data['start'] is not None and data['end'] is not None:
        assert (_str_to_index(data['line'], data['start'])
                <= _str_to_index(data['line'], data['end'])), (
            'start must be larger than end in forward mode')

    if data['only_twiss_init']:
        assert data['periodic'], (
            '``only_twiss_init`` can only be used in periodic mode')
        if data['reverse']:
            return data['init'].reverse()
        return data['init']

    if data['only_markers'] and data['radiation_analysis']:
        raise NotImplementedError(
            '``only_markers`` not implemented for ``radiation_analysis``')

    twiss_res = _propagate_twiss_from_init(
        line=data['line'],
        init=data['init'],
        start=data['start'],
        end=data['end'],
        nemitt_x=data['nemitt_x'],
        nemitt_y=data['nemitt_y'],
        step_W_sigma=data['step_W_sigma'],
        delta_disp=data['delta_disp'],
        use_full_inverse=data['use_full_inverse'],
        hide_thin_groups=data['hide_thin_groups'],
        only_markers=data['only_markers'],
        only_orbit=data['only_orbit'],
        spin=data['spin'],
        compute_lattice_functions=data['compute_lattice_functions'],
        continue_if_lost=data['_continue_if_lost'],
        keep_tracking_data=data['_keep_tracking_data'],
        keep_initial_particles=data['_keep_initial_particles'],
        initial_particles=data['_initial_particles'],
        ebe_monitor=data['_ebe_monitor'])

    if (data['periodic']
            and not data['skip_global_quantities']
            and not data['only_orbit']):
        _add_periodic_solution_data_to_base_twiss(data, twiss_res)

    _add_chromatic_functions_to_twiss_result(data, twiss_res)
    _add_radiation_analysis_to_twiss_result(data, twiss_res)
    _apply_4d_longitudinal_result_convention(data, twiss_res)
    twiss_res = _set_twiss_result_values_at(data, twiss_res)
    _add_strengths_and_radiation_integrals_to_twiss_result(data, twiss_res)
    _add_spin_polarization_to_twiss_result(data, twiss_res)
    _add_edwards_teng_coupling_to_twiss_result(data, twiss_res)
    _add_base_twiss_metadata(data, twiss_res)

    twiss_res = _reverse_twiss_result_if_needed(data, twiss_res)
    if not data['periodic'] and not data['only_orbit']:
        _align_open_twiss_phases_with_init(data, twiss_res)
    _add_measured_revolution_period_if_requested(data, twiss_res)

    if data['num_turns'] > 1:
        twiss_res = _extend_base_twiss_to_multiple_turns(data, twiss_res)

    twiss_res = _select_twiss_result_at_elements(data, twiss_res)
    _add_periodicity_and_completed_init_to_twiss_result(data, twiss_res)

    return twiss_res


def _select_twiss_route(data):

    if data['start'] is not None and data['end'] is None:
        if data['periodic']:
            return 'periodic_one_turn_custom_start'
        return 'open_one_turn_custom_start'

    if (data['init'] == 'full_periodic'
            and (data['start'] is not None or data['end'] is not None)):
        return 'open_init_from_full_periodic'

    if data['periodic']:
        return 'periodic'
    return 'open'


def _get_open_twiss_range_flags(data):

    start = data['start']
    end = data['end']
    if start is None or end is None:
        crosses_line_boundary = False
    else:
        direction_sign = -1 if data['reverse'] else 1
        crosses_line_boundary = (
            direction_sign * _str_to_index(data['line'], start)
            > direction_sign * _str_to_index(data['line'], end))

    init_is_at_boundary = data['init'].element_name in (start, end)
    return crosses_line_boundary, init_is_at_boundary


def _handle_init_inside_range_and_line_wrap(
        kwargs, crosses_line_boundary, one_turn_from_start=False):

    if not crosses_line_boundary:
        kwargs = kwargs.copy()
        line = kwargs['line']
        start = kwargs.pop('start')
        end = kwargs.pop('end')
        init = kwargs.pop('init')
        reverse = kwargs.pop('reverse')

        # Bidirectional propagation from an interior init is supported at
        # markers.
        init_element_name = init.element_name
        init_element = line.get(init_element_name)
        if isinstance(init_element, xt.Replica):
            init_element = init_element.resolve()
        if not isinstance(init_element, xt.Marker):
            raise ValueError(
                'The element at the initial position is not a Marker. '
                'This is not yet supported')

        if reverse:
            assert (_str_to_index(line, init_element_name)
                    <= _str_to_index(line, start))
            assert (_str_to_index(line, init_element_name)
                    >= _str_to_index(line, end))
        else:
            assert (_str_to_index(line, init_element_name)
                    >= _str_to_index(line, start))
            assert (_str_to_index(line, init_element_name)
                    <= _str_to_index(line, end))

        # Propagate both sides from the same init, then restore one continuous
        # table.
        first_table, second_table = tuple(
            _compute_base_twiss(
                kwargs,
                start=piece_start,
                end=piece_end,
                init=init,
                reverse=reverse)
            for piece_start, piece_end in (
                (start, init_element_name),
                (init_element_name, end),
            ))

        return _combine_init_inside_range_twiss_tables(
            first_table, second_table, init)

    kwargs = kwargs.copy()
    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    line = kwargs['line']
    reverse = kwargs['reverse']

    # Confirm that the requested traversal crosses the physical line boundary.
    if one_turn_from_start:
        assert start == end
    elif not reverse:
        assert _str_to_index(line, end) < _str_to_index(line, start), (
            'This function should not have been called')
    else:
        assert _str_to_index(line, end) > _str_to_index(line, start), (
            'This function should not have been called')

    if reverse:
        line_boundary_end = line._element_names_unique[0]
        line_boundary_start = line._element_names_unique[-1]
    else:
        line_boundary_end = line._element_names_unique[-1]
        line_boundary_start = line._element_names_unique[0]

    if one_turn_from_start:
        # A start without an end requests one complete turn. Propagate across
        # the physical line boundary, then replace the repeated start row with
        # the conventional final _end_point row.
        first_table = _compute_base_twiss(
            kwargs, start=start, end=line_boundary_end, init=init)
        boundary_init = first_table.get_twiss_init('_end_point')
        boundary_init.element_name = line_boundary_start
        second_table = _compute_base_twiss(
            kwargs,
            start=line_boundary_start,
            end=start,
            init=boundary_init)
        second_table = second_table.rows[:-1]
        second_table.name[-1] = '_end_point'
        twiss_res = TwissTable.concatenate([first_table, second_table])
        twiss_res['completed_init'] = first_table.completed_init
        return twiss_res

    init_element_name = init.element_name
    init_index = _str_to_index(line, init_element_name)
    start_index = _str_to_index(line, start)
    end_index = _str_to_index(line, end)

    if init_element_name not in (start, end):
        # Build the side containing the init in both directions, then transfer
        # its boundary conditions across the physical end of the line.
        init_is_after_start = (
            (not reverse and init_index >= start_index)
            or (reverse and init_index <= start_index))

        if init_is_after_start:
            first_table = _compute_base_twiss(
                kwargs, start=start, end=init_element_name, init=init)
            second_table = _compute_base_twiss(
                kwargs, start=init_element_name, end=line_boundary_end,
                init=init)
            first_side_table = _combine_init_inside_range_twiss_tables(
                first_table, second_table, init)
            boundary_init = second_table.get_twiss_init('_end_point')
            boundary_init.element_name = line_boundary_start
            third_table = _compute_base_twiss(
                kwargs, start=line_boundary_start, end=end,
                init=boundary_init)
            twiss_tables = (first_side_table, third_table)
            completed_init = first_side_table.completed_init

        else:
            second_table = _compute_base_twiss(
                kwargs, start=line_boundary_start, end=init_element_name,
                init=init)
            third_table = _compute_base_twiss(
                kwargs, start=init_element_name, end=end, init=init)
            second_side_table = _combine_init_inside_range_twiss_tables(
                second_table, third_table, init)
            boundary_init = second_table.get_twiss_init(line_boundary_start)
            boundary_init.element_name = line_boundary_end
            first_table = _compute_base_twiss(
                kwargs, start=start, end=line_boundary_end,
                init=boundary_init)
            twiss_tables = (first_table, second_side_table)
            completed_init = second_side_table.completed_init

    else:
        # With a boundary init, propagate its side first and transfer across.
        if not reverse and init_index >= start_index:
            init_is_in_first_piece = True
        elif not reverse and init_index <= end_index:
            init_is_in_first_piece = False
        elif reverse and init_index <= start_index:
            init_is_in_first_piece = True
        elif reverse and init_index >= end_index:
            init_is_in_first_piece = False
        else:
            raise RuntimeError(
                'Boundary conditions not at start or end of the specified range')

        if init_is_in_first_piece:
            first_table = _compute_base_twiss(
                kwargs, start=start, end=line_boundary_end, init=init)
            boundary_init = first_table.get_twiss_init('_end_point')
            boundary_init.element_name = line_boundary_start
            second_table = _compute_base_twiss(
                kwargs, start=line_boundary_start, end=end,
                init=boundary_init)
            completed_init = first_table.completed_init
        else:
            second_table = _compute_base_twiss(
                kwargs, start=line_boundary_start, end=end, init=init)
            boundary_init = second_table.get_twiss_init(line_boundary_start)
            boundary_init.element_name = line_boundary_end
            first_table = _compute_base_twiss(
                kwargs, start=start, end=line_boundary_end,
                init=boundary_init)
            completed_init = second_table.completed_init

        twiss_tables = (first_table, second_table)

    # Assemble the output in traversal order and align it with the supplied init.
    twiss_res = TwissTable.concatenate(twiss_tables)
    twiss_res.s -= twiss_res['s', init_element_name] - init.s
    twiss_res['completed_init'] = completed_init

    if 'mux' in twiss_res.keys():
        twiss_res.mux -= twiss_res['mux', init_element_name] - init.mux
        twiss_res.muy -= twiss_res['muy', init_element_name] - init.muy
        twiss_res.muzeta -= (
            twiss_res['muzeta', init_element_name] - init.muzeta)
    if 'dzeta' in twiss_res.keys():
        twiss_res.dzeta -= twiss_res['dzeta', init_element_name] - init.dzeta

    _remove_unsupported_phase_derivative_columns(twiss_res)
    twiss_res._data['loop_around'] = True
    _copy_common_metadata_from_tables(
        twiss_res=twiss_res, twiss_tables=twiss_tables)

    return twiss_res


def _combine_init_inside_range_twiss_tables(first_table, second_table, init):

    init_element_name = init.element_name
    twiss_res = TwissTable.concatenate([first_table, second_table])
    twiss_res['completed_init'] = first_table.completed_init

    twiss_res.s -= twiss_res['s', init_element_name] - init.s
    twiss_res.mux -= twiss_res['mux', init_element_name] - init.mux
    twiss_res.muy -= twiss_res['muy', init_element_name] - init.muy
    twiss_res.muzeta -= (
        twiss_res['muzeta', init_element_name] - init.muzeta)
    if 'dzeta' in twiss_res:
        twiss_res.dzeta -= twiss_res['dzeta', init_element_name] - init.dzeta

    _remove_unsupported_phase_derivative_columns(twiss_res)
    _copy_common_metadata_from_tables(
        twiss_res=twiss_res, twiss_tables=(first_table, second_table))

    return twiss_res


def _remove_unsupported_phase_derivative_columns(twiss_res):

    for column_name in ['dmux', 'dmuy']:
        if column_name in twiss_res.keys():
            twiss_res._data.pop(column_name)
            twiss_res._col_names.remove(column_name)


def _copy_common_metadata_from_tables(twiss_res, twiss_tables):

    for field_name in ['method', 'radiation_method', 'reference_frame']:
        values = [table[field_name] for table in twiss_tables]
        if all(value == values[0] for value in values[1:]):
            twiss_res._data[field_name] = values[0]
        else:
            twiss_res._data[field_name] = tuple(values)
