# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from contextlib import ExitStack

from .twiss_table import TwissTable
from .twiss_defaults_and_input_preparation import (
    _element_ref_to_index,
    _normalize_twiss_inputs,
)
from .chromatic_functions import trapz
from .handle_init_inside_range_and_line_wrap import (
    _compute_twiss_handling_init_inside_range_and_line_wrap,
)
from .optics_propagation import _propagate_twiss_from_init
from . import twiss_postprocess_and_complem_results as twpc
from .multiturn import (
    _extend_twiss_result_to_multiple_turns,
    _kwargs_for_multiturn_continuation,
)
from .twiss_init import (
    TwissInit,
    _build_twiss_init_from_inputs,
    _clear_twiss_init_input_fields,
    _compute_periodic_twiss_init,
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
        steps_r_matrix=None, *,
        with_progress=True,
        chi=None, charge_ratio=None, mass_ratio=None,
    ):
    """
    Compute the Twiss parameters of the beam line. If no initial conditions
    are provided, the periodic solution is computed.

    Parameters
    ----------

    method : {'6d', '4d'}, optional
        Method to be used for the computation. If '6d' the full 6D
        normal form is used. If '4d' the 4D normal form is used.
    particle_ref : xpart.Particles, optional
        Reference particle used to search for the closed orbit. If not provided,
        ``line.particle_ref`` is used.
    chi : float, optional
        Relative charge-to-mass ratio ``q / q0 * mass0 / mass``. The particle
        used for the Twiss calculation is a copy; ``particle_ref`` is not
        modified. If provided alone, its ``charge_ratio`` is preserved and
        ``mass_ratio`` is adjusted consistently.
    charge_ratio : float, optional
        Relative charge ``q / q0``. If provided alone, the ``mass_ratio`` of
        ``particle_ref`` is preserved and ``chi`` is adjusted consistently.
    mass_ratio : float, optional
        Relative rest mass ``mass / mass0``. If provided alone, the
        ``charge_ratio`` of ``particle_ref`` is preserved and ``chi`` is
        adjusted consistently.
    with_progress : bool, optional
        Whether to show progress when temporary slicing is needed for ``at_s``.
        Defaults to ``True``.
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
        If True, the Edwards-Teng coupling matrix is reconstructed. The
        Edwards-Teng Twiss functions, normalization `g_edw_teng`, and linear
        coupling RDTs are computed directly from `W_matrix` independently of
        this option. Edwards-Teng quantities are exact for `method='4d'`. For
        `method='6d'`, they can become inaccurate when the transverse 4D
        transfer matrix has a significant symplectic deviation. Default is
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
        - `betx_edw_teng`, `bety_edw_teng`, `alfx_edw_teng`, `alfy_edw_teng`:
          Edwards-Teng modal Twiss functions. Exact for `method='4d'`; for
          `method='6d'`, they can become inaccurate when the transverse 4D
          transfer matrix has a significant symplectic deviation. (ebe)
        - `g_edw_teng`: Edwards-Teng coupling normalization. Exact for
          `method='4d'`; for `method='6d'`, it can become inaccurate when the
          transverse 4D transfer matrix has a significant symplectic deviation.
          (ebe)
        - `f1001`, `f1010`, `f0110`, `f0101`: matrix-equivalent linear coupling
          RDTs obtained directly from `W_matrix`. Exact for `method='4d'`; for
          `method='6d'`, they can become inaccurate when the transverse 4D
          transfer matrix has a significant symplectic deviation. (ebe)
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
          elements of the Edwards-Teng coupling matrix. Exact for
          `method='4d'`; for `method='6d'`, they can become inaccurate when the
          transverse 4D transfer matrix has a significant symplectic deviation.
          (ebe)
    Output fields present when `search_for_t_rev=True`:
        - `t_rev`: measured revolution period [s]

    """
    # Normalize all inputs (handle defaults and deprecated arguments).
    # Prepare dictionaries describing modifications to be made to line.config
    # and line.tracker.track_flags.
    (twiss_config, input_kwargs, track_flag_updates, line_config_updates
     ) = _normalize_twiss_inputs(
        twiss_kwargs=locals().copy())

    with ExitStack() as twiss_context:
        # Apply the requested temporary line state (line.config and
        # line.tracker.track_flags). The `with ExitStack()` context manager
        # ensures that the original line state is restored at the end or if
        # an exception is raised.
        _apply_twiss_line_context(
            twiss_context=twiss_context,
            line=twiss_config['line'],
            track_flag_updates=track_flag_updates,
            line_config_updates=line_config_updates,
            freeze_longitudinal=twiss_config['freeze_longitudinal'],
            freeze_energy=twiss_config['freeze_energy'])

        # Determine the route (periodic / open / periodic_one_turn_custom_start
        # / open_one_turn_custom_start / open_init_from_full_periodic)
        route = _select_twiss_route(twiss_config)

        if route == 'periodic':
            # Standard periodic Twiss, for the full line or a closed range.
            result_init = twiss_config['init']
            _clear_twiss_init_input_fields(twiss_config)
            twiss_config.update(_compute_periodic_twiss_init(twiss_config))
            twiss_res = _compute_base_twiss(twiss_config)

        elif route == 'open':
            # Standard open Twiss from supplied init data.
            twiss_config['init'] = _build_twiss_init_from_inputs(twiss_config)
            result_init = twiss_config['init'].copy()
            _clear_twiss_init_input_fields(twiss_config)

            crosses_line_boundary, init_is_at_boundary = (
                _get_open_twiss_range_flags(twiss_config))
            if not crosses_line_boundary and init_is_at_boundary:
                twiss_res = _compute_base_twiss(twiss_config)
            else:
                twiss_res = (
                    _compute_twiss_handling_init_inside_range_and_line_wrap(
                        twiss_config, crosses_line_boundary,
                        compute_base_twiss=_compute_base_twiss))

        elif route == 'periodic_one_turn_custom_start':
            # Compute a full periodic table, then rotate it to the requested start.
            requested_start = twiss_config['start']
            one_turn_kwargs = twiss_config.copy()
            one_turn_kwargs['start'] = None
            result_init = one_turn_kwargs['init']
            _clear_twiss_init_input_fields(one_turn_kwargs)
            one_turn_kwargs.update(
                _compute_periodic_twiss_init(one_turn_kwargs))

            full_twiss = _compute_base_twiss(one_turn_kwargs)
            first_part = full_twiss.rows[requested_start:]
            second_part = full_twiss.rows[:requested_start]
            twiss_res = TwissTable.concatenate([first_part, second_part])
            twiss_res.zero_at(twiss_res.name[0])
            twiss_res.name[-1] = '_end_point'

        elif route == 'open_one_turn_custom_start':
            # A start without an end requests a wrapped open range of one turn.
            twiss_config['end'] = twiss_config['start']
            twiss_config['init'] = _build_twiss_init_from_inputs(twiss_config)
            result_init = twiss_config['init'].copy()
            _clear_twiss_init_input_fields(twiss_config)
            twiss_res = (
                _compute_twiss_handling_init_inside_range_and_line_wrap(
                    twiss_config,
                    crosses_line_boundary=True,
                    one_turn_from_start=True,
                    compute_base_twiss=_compute_base_twiss))

        elif route == 'open_init_from_full_periodic':
            # Compute a forward full-line periodic table and take the open-range
            # init from the requested location.
            periodic_kwargs = twiss_config.copy()
            periodic_kwargs.update(
                init=None,
                start=None,
                end=None,
                init_at=None,
                periodic=True,
                periodic_mode='periodic',
            )
            _clear_twiss_init_input_fields(periodic_kwargs)
            periodic_kwargs.update(
                _compute_periodic_twiss_init(periodic_kwargs))
            full_periodic_twiss = _compute_base_twiss(periodic_kwargs)

            twiss_config['init'] = full_periodic_twiss.get_twiss_init(
                twiss_config['init_at'] or twiss_config['start'])
            twiss_config['init_at'] = None
            twiss_config['init'] = _build_twiss_init_from_inputs(twiss_config)
            result_init = twiss_config['init'].copy()
            _clear_twiss_init_input_fields(twiss_config)

            crosses_line_boundary, init_is_at_boundary = (
                _get_open_twiss_range_flags(twiss_config))
            if not crosses_line_boundary and init_is_at_boundary:
                twiss_res = _compute_base_twiss(twiss_config)
            else:
                twiss_res = (
                    _compute_twiss_handling_init_inside_range_and_line_wrap(
                        twiss_config, crosses_line_boundary,
                        compute_base_twiss=_compute_base_twiss))
            if twiss_config['zero_at'] is None:
                twiss_res.zero_at(twiss_config['start'])

        else:
            raise RuntimeError(f'Unexpected Twiss route: {route}')

        if isinstance(twiss_res, TwissInit):
            return twiss_res

        # Multi-turn case (calls twiss_line recursively)
        if twiss_config['num_turns'] > 1:
            multiturn_kwargs = _kwargs_for_multiturn_continuation(
                input_kwargs, twiss_config)
            twiss_res = _extend_twiss_result_to_multiple_turns(
                twiss_res=twiss_res,
                num_turns=twiss_config['num_turns'],
                kwargs=multiturn_kwargs)

        if twiss_config['at_elements'] is not None:
            twiss_res = twiss_res.rows[twiss_config['at_elements']]

        twiss_res['periodic'] = twiss_config['periodic']
        twiss_res['completed_init'] = result_init
        twiss_res._sort_col_names()

        if twiss_config['zero_at'] is not None:
            twiss_res.zero_at(twiss_config['zero_at'])

        twiss_res._data['_action'] = xt.match.ActionTwiss(**input_kwargs)

        return twiss_res


def _compute_base_twiss(twiss_config):
    """Propagate from a concrete init and finish one non-composed result."""

    twiss_config = twiss_config.copy()

    assert isinstance(twiss_config['init'], TwissInit), (
        '_compute_base_twiss requires a concrete TwissInit')

    line = twiss_config['line']
    start = twiss_config['start']
    end = twiss_config['end']
    reverse = twiss_config['reverse']

    # validate `start`` and `end`` and handle `reverse``
    if start is not None and end is not None:
        start_index = _element_ref_to_index(line, start)
        end_index = _element_ref_to_index(line, end)
        if reverse:
            assert start_index >= end_index, (
                'start must be at or after end in reverse mode')
        else:
            assert start_index <= end_index, (
                'start must be at or before end in forward mode')

    if reverse:
        start, end = end, start
        twiss_config['start'] = start
        twiss_config['end'] = end

    if twiss_config['only_twiss_init']:
        assert twiss_config['periodic'], (
            '``only_twiss_init`` can only be used in periodic mode')
        if reverse:
            return twiss_config['init'].reverse()

        return twiss_config['init']

    twiss_res = _propagate_twiss_from_init(
        line=line,
        init=twiss_config['init'],
        start=start,
        end=end,
        nemitt_x=twiss_config['nemitt_x'],
        nemitt_y=twiss_config['nemitt_y'],
        step_W_sigma=twiss_config['step_W_sigma'],
        delta_disp=twiss_config['delta_disp'],
        use_full_inverse=twiss_config['use_full_inverse'],
        hide_thin_groups=twiss_config['hide_thin_groups'],
        only_markers=twiss_config['only_markers'],
        only_orbit=twiss_config['only_orbit'],
        spin=twiss_config['spin'],
        compute_lattice_functions=twiss_config['compute_lattice_functions'],
        continue_if_lost=twiss_config['_continue_if_lost'],
        keep_tracking_data=twiss_config['_keep_tracking_data'],
        keep_initial_particles=twiss_config['_keep_initial_particles'],
        initial_particles=twiss_config['_initial_particles'],
        ebe_monitor=twiss_config['_ebe_monitor'])

    if (twiss_config['periodic']
            and not twiss_config['skip_global_quantities']
            and not twiss_config['only_orbit']):
        twpc._add_periodic_solution_data_to_twiss_result(
            twiss_config, twiss_res)

    twpc._add_chromatic_functions_to_twiss_result(twiss_config, twiss_res)
    twpc._add_radiation_analysis_to_twiss_result(twiss_config, twiss_res)
    twpc._apply_4d_longitudinal_result_convention(twiss_config, twiss_res)
    twiss_res = twpc._set_twiss_result_values_at(twiss_config, twiss_res)
    twpc._add_strengths_and_radiation_integrals_to_twiss_result(
        twiss_config, twiss_res)
    twpc._add_spin_polarization_to_twiss_result(twiss_config, twiss_res)
    twpc._add_edwards_teng_coupling_to_twiss_result(twiss_config, twiss_res)
    twpc._add_base_twiss_metadata(twiss_config, twiss_res)

    twiss_res = twpc._reverse_twiss_result_if_needed(twiss_config, twiss_res)
    if not twiss_config['periodic'] and not twiss_config['only_orbit']:
        twpc._align_open_twiss_phases_with_init(twiss_config, twiss_res)
    twpc._add_measured_revolution_period_if_requested(twiss_config, twiss_res)

    return twiss_res


def _select_twiss_route(twiss_config):

    if twiss_config['start'] is not None and twiss_config['end'] is None:
        if twiss_config['periodic']:
            return 'periodic_one_turn_custom_start'
        return 'open_one_turn_custom_start'

    if (twiss_config['init'] == 'full_periodic'
            and (twiss_config['start'] is not None or twiss_config['end'] is not None)):
        return 'open_init_from_full_periodic'

    if twiss_config['periodic']:
        return 'periodic'
    return 'open'


def _get_open_twiss_range_flags(twiss_config):

    start = twiss_config['start']
    end = twiss_config['end']
    if start is None or end is None:
        crosses_line_boundary = False
    else:
        direction_sign = -1 if twiss_config['reverse'] else 1
        crosses_line_boundary = (
            direction_sign * _element_ref_to_index(
                twiss_config['line'], start)
            > direction_sign * _element_ref_to_index(
                twiss_config['line'], end))

    init_is_at_boundary = twiss_config['init'].element_name in (start, end)
    return crosses_line_boundary, init_is_at_boundary


def _apply_twiss_line_context(
        twiss_context, line, track_flag_updates, line_config_updates, *,
        freeze_longitudinal, freeze_energy):
    """Apply and automatically restore the normalized temporary line state."""

    if freeze_longitudinal:
        twiss_context.enter_context(xt.freeze_longitudinal(line))
    elif freeze_energy:
        if not line._energy_is_frozen():
            twiss_context.enter_context(xt.line._preserve_config(line))
            line.freeze_energy(force=True)  # force is needed for collective lines

    if track_flag_updates:
        twiss_context.enter_context(xt.line._preserve_track_flags(line))
        for flag_name, value in track_flag_updates.items():
            setattr(line.tracker.track_flags, flag_name, value)

    if line_config_updates:
        twiss_context.enter_context(xt.line._preserve_config(line))
        for config_name, value in line_config_updates.items():
            setattr(line.config, config_name, value)
