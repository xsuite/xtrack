# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as clight

import xobjects as xo
from ..general import _print
from .closed_orbit import ClosedOrbitSearchError, find_closed_orbit_line
from .twiss_init import TwissInit, _complete_twiss_init
from .element_indexing import _str_to_index
from .twiss_table import TwissTable
from .transfer_matrices import _complete_steps_r_matrix_with_default
from .input_normalization import (
    _apply_twiss_defaults,
    _handle_deprecated_twiss_kwargs,
)
from .ring_quantities import _add_ring_quantities
from .periodic_solution import _find_periodic_solution
from .chromatic_functions import _get_chromatic_functions, trapz
from .extra_markers import _build_auxiliary_tracker_with_extra_markers
from .open_twiss import _twiss_open
from .open_propagation import (
    _plan_loop_around_twiss_parts,
    _plan_init_inside_range_twiss_parts,
    _plan_open_one_turn_twiss,
)
from .finalize import _finalize_twiss_result
from .spin import _get_spin_polarization
from .non_linear_chromaticity import get_non_linear_chromaticity
from .coupling_edw_teng import _get_coupling_elements_edwards_teng
from .radiation import (
    _get_eneloss_and_damping_rates,
    _get_equilibrium_emittance_full,
    _get_equilibrium_emittance_kick_as_co,
)
from .strengths import _add_strengths_to_twiss_res
from .constants import (
    DEFAULT_MATRIX_RESPONSIVENESS_TOL,
    DEFAULT_MATRIX_STABILITY_TOL,
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
    (chrom, step_W_sigma, polarization_analysis, radiation_analysis,
        steps_R_matrix) = _handle_deprecated_twiss_kwargs(
        at_s=at_s,
        at_elements=at_elements,
        compute_chromatic_properties=compute_chromatic_properties,
        r_sigma=r_sigma,
        freeze_energy=freeze_energy,
        freeze_longitudinal=freeze_longitudinal,
        polarization=polarization,
        eneloss_and_damping=eneloss_and_damping,
        steps_r_matrix=steps_r_matrix,
        chrom=chrom,
        step_W_sigma=step_W_sigma,
        polarization_analysis=polarization_analysis,
        radiation_analysis=radiation_analysis,
        steps_R_matrix=steps_R_matrix,
    )

    input_kwargs = locals().copy()

    (step_W_sigma, nemitt_x, nemitt_y, delta_disp, delta_chrom, zeta_disp,
        zeta_shift, values_at_element_exit, continue_on_closed_orbit_error,
        freeze_longitudinal, radiation_method, spin, polarization_analysis,
        radiation_integrals, radiation_analysis, symplectify, reverse,
        strengths, hide_thin_groups, search_for_t_rev, num_turns_search_t_rev,
        only_twiss_init, only_markers, only_orbit, compute_R_element_by_element,
        compute_lattice_functions, chrom, num_turns,
        disable_apertures) = _apply_twiss_defaults(
        step_W_sigma=step_W_sigma,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        delta_disp=delta_disp,
        delta_chrom=delta_chrom,
        zeta_disp=zeta_disp,
        zeta_shift=zeta_shift,
        values_at_element_exit=values_at_element_exit,
        continue_on_closed_orbit_error=continue_on_closed_orbit_error,
        freeze_longitudinal=freeze_longitudinal,
        radiation_method=radiation_method,
        spin=spin,
        polarization_analysis=polarization_analysis,
        radiation_integrals=radiation_integrals,
        radiation_analysis=radiation_analysis,
        symplectify=symplectify,
        reverse=reverse,
        strengths=strengths,
        hide_thin_groups=hide_thin_groups,
        search_for_t_rev=search_for_t_rev,
        num_turns_search_t_rev=num_turns_search_t_rev,
        only_twiss_init=only_twiss_init,
        only_markers=only_markers,
        only_orbit=only_orbit,
        compute_R_element_by_element=compute_R_element_by_element,
        compute_lattice_functions=compute_lattice_functions,
        chrom=chrom,
        num_turns=num_turns,
        disable_apertures=disable_apertures,
    )

    if only_markers:
        raise NotImplementedError('``only_markers`` not supported anymore')

    if polarization_analysis:
        spin = True
        radiation_integrals = True # some quantities are needed for polarization
                                   # could be decoupled in the future
    if spin:
        assert reverse is False

    if isinstance(init, TwissInit):
        init = init.copy()

    kwargs = locals().copy()
    zero_at_requested = zero_at
    zero_at = None

    with ExitStack() as twiss_context:
        if disable_apertures:
            if not (line.tracker.track_flags.XS_FLAG_IGNORE_GLOBAL_APERTURE
                    and line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE):
                twiss_context.enter_context(
                    xt.line._preserve_track_flags(line))
                line.tracker.track_flags.XS_FLAG_IGNORE_GLOBAL_APERTURE = True
                line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE = True

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

        if (init is not None and init not in ['periodic', 'periodic_symmetric']
            or betx is not None or bety is not None):
            periodic = False
            periodic_mode = None
        else:
            periodic = True
            periodic_mode = init or 'periodic'
            assert x is None, '``x`` not supported for periodic twiss'
            assert px is None, '``px`` not supported for periodic twiss'
            assert y is None, '``y`` not supported for periodic twiss'
            assert py is None, '``py`` not supported for periodic twiss'
            assert zeta is None, '``zeta`` not supported for periodic twiss'
            assert delta is None, '``delta`` not supported for periodic twiss'

        freeze_longitudinal, freeze_energy = _enter_twiss_freeze_context(
            twiss_context=twiss_context,
            line=line,
            freeze_longitudinal=freeze_longitudinal,
            freeze_energy=freeze_energy,
        )

        if method == '4d' and not line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK:
            twiss_context.enter_context(xt.line._preserve_track_flags(line))
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True

        if at_s is not None:
            if reverse:
                raise NotImplementedError('``at_s`` not implemented for ``reverse``=True')
            if np.isscalar(at_s):
                at_s = [at_s]
            assert at_elements is None
            (auxtracker, names_inserted_markers
                ) = _build_auxiliary_tracker_with_extra_markers(
                tracker=line.tracker, at_s=at_s, marker_prefix='inserted_twiss_marker',
                algorithm='insert')
            line = auxtracker.line
            at_elements = names_inserted_markers
            at_s = None
            strengths = True

        if radiation_method is None and line._radiation_model is not None:
            if line._radiation_model in ('quantum', 'quantum-kick'):
                raise ValueError(
                    'twiss cannot be called when the radiation model is stochastic')
            if method == '4d':
                raise RuntimeError('4d twiss cannot be called when radiation is present')
            radiation_method = 'kick_as_co'

        if radiation_method is not None and radiation_method != 'full':
            assert isinstance(line._context, xo.ContextCpu), (
                'Twiss with radiation computation is only supported on CPU')
            assert not line._context.openmp_enabled, (
                'Twiss with radiation computation is not supported with OpenMP'
                ' parallelization')
            assert radiation_method in ['full', 'kick_as_co', 'scale_as_co']
            assert freeze_longitudinal is False
            if (radiation_method == 'kick_as_co' and (
                not line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST)):
                twiss_context.enter_context(xt.line._preserve_track_flags(line))
                line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = True
            elif (radiation_method == 'scale_as_co' and (
                not hasattr(line.config, 'XTRACK_SYNRAD_SCALE_SAME_AS_FIRST') or
                not line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST)):
                twiss_context.enter_context(xt.line._preserve_config(line))
                line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = True

        if radiation_method == 'kick_as_co':
            assert line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST

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

        composed_twiss_res = None
        if start is not None and end is None:
            kwargs = _kwargs_for_composed_twiss_call(kwargs, locals().copy())
            composed_twiss_res = _compute_one_turn_twiss_from_start(
                kwargs=kwargs,
                line=line,
                start=start,
                init=init,
                betx=betx,
                bety=bety,
            )
        elif init == 'full_periodic' and (start is not None or end is not None):
            kwargs = _kwargs_for_composed_twiss_call(kwargs, locals().copy())
            kwargs = _prepare_kwargs_for_full_periodic_twiss(kwargs)
            full_periodic_init = _compute_full_periodic_twiss_init(
                kwargs=kwargs, start=start, init_at=init_at)
            composed_twiss_res = _compute_twiss_segment(
                kwargs,
                start=start,
                end=end,
                init=full_periodic_init,
            )
            if zero_at_requested is None:
                composed_twiss_res.zero_at(start)

        if composed_twiss_res is not None:
            return _finalize_twiss_result(
                composed_twiss_res, input_kwargs, zero_at=zero_at_requested)

        init, completed_init = _complete_init_for_base_twiss(
            start=start, end=end, init_at=init_at, init=init,
            line=line, reverse=reverse,
            x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
            alfx=alfx, alfy=alfy, betx=betx, bety=bety, bets=bets,
            dx=dx, dpx=dpx, dy=dy, dpy=dpy, dzeta=dzeta,
            mux=mux, muy=muy, muzeta=muzeta,
            ax_chrom=ax_chrom, bx_chrom=bx_chrom, ay_chrom=ay_chrom, by_chrom=by_chrom,
            ddx=ddx, ddpx=ddpx, ddy=ddy, ddpy=ddpy,
            spin_x=spin_x, spin_y=spin_y, spin_z=spin_z
        )

        # clean quantities embedded in init
        init_at=None
        x=None; px=None; y=None; py=None; zeta=None; delta=None
        alfx=None; alfy=None; betx=None; bety=None; bets=None
        dx=None; dpx=None; dy=None; dpy=None; dzeta=None
        mux=None; muy=None; muzeta=None
        ax_chrom=None; bx_chrom=None; ay_chrom=None; by_chrom=None
        ddx=None; ddpx=None; ddy=None; ddpy=None
        spin_x=None; spin_y=None; spin_z=None

        composed_twiss_res = None

        # Twiss goes through the start of the line
        rv = (-1 if reverse else 1)
        if not periodic and (
            rv * _str_to_index(line, start) > rv * _str_to_index(line, end)):

            kwargs = _kwargs_for_composed_twiss_call(kwargs, locals().copy())
            composed_twiss_res = _handle_loop_around(kwargs)

        # init is not at the boundary
        elif (not periodic and not isinstance(init, str)
                and init.element_name != start
                and init.element_name != end):

            kwargs = _kwargs_for_composed_twiss_call(kwargs, locals().copy())
            composed_twiss_res = _handle_init_inside_range(kwargs)

        if composed_twiss_res is not None:
            return _finalize_twiss_result(
                composed_twiss_res, input_kwargs, zero_at=zero_at_requested)

        base_twiss = _TwissBaseComputation(locals().copy())
        base_twiss.prepare_for_propagation_from_init()

        if only_twiss_init:
            assert periodic, '``only_twiss_init`` can only be used in periodic mode'
            return base_twiss.init_for_only_twiss_init()

        if only_markers and radiation_analysis:
            raise NotImplementedError(
                '``only_markers`` not implemented for ``radiation_analysis``')

        twiss_res = base_twiss.propagate_from_init()
        twiss_res = base_twiss.finish_result(twiss_res)

        return _finalize_twiss_result(twiss_res, input_kwargs, zero_at=zero_at_requested)

@dataclass(frozen=True)
class _PeriodicTwissInitData:
    init: object
    R_matrix: object
    steps_R_matrix: object
    eigenvalues: object
    Rot: object
    RR_ebe: object

    @classmethod
    def from_periodic_solution(cls, periodic_solution):
        init, R_matrix, steps_R_matrix, eigenvalues, Rot, RR_ebe = (
            periodic_solution)
        return cls(
            init=init,
            R_matrix=R_matrix,
            steps_R_matrix=steps_R_matrix,
            eigenvalues=eigenvalues,
            Rot=Rot,
            RR_ebe=RR_ebe,
        )


class _TwissBaseComputation:
    def __init__(self, data):
        self.__dict__.update(data)
        self.periodic_init_data = None

    def prepare_for_propagation_from_init(self):

        self._prepare_range_and_line()
        self._validate_base_request()
        self._acquire_periodic_init_if_needed()

    def _prepare_range_and_line(self):

        self.start, self.end = _apply_base_twiss_reverse_range(
            line=self.line, start=self.start, end=self.end,
            reverse=self.reverse)

        (self.matrix_responsiveness_tol, self.matrix_stability_tol,
            self.use_full_inverse) = _prepare_base_twiss_matrix_settings(
                line=self.line,
                radiation_method=self.radiation_method,
                matrix_responsiveness_tol=self.matrix_responsiveness_tol,
                matrix_stability_tol=self.matrix_stability_tol,
                use_full_inverse=self.use_full_inverse)

        self.line, self.particle_ref = _prepare_base_twiss_line_and_particle_ref(
            line=self.line,
            particle_ref=self.particle_ref,
            particle_on_co=self.particle_on_co,
            co_guess=self.co_guess,
            include_collective=self.include_collective)

        self.method = _validate_base_twiss_method(self.method)

    def _validate_base_request(self):

        _validate_base_twiss_boundary_init(start=self.start, init=self.init)
        _validate_base_twiss_init_mode(init=self.init)
        _validate_base_twiss_open_momentum_offsets(
            periodic=self.periodic, delta0=self.delta0, zeta0=self.zeta0)

    def _acquire_periodic_init_if_needed(self):

        if not self.periodic:
            self.skip_global_quantities = True
            return

        self.periodic_init_data = _compute_periodic_twiss_init_and_data(
            line=self.line,
            particle_on_co=self.particle_on_co,
            particle_ref=self.particle_ref,
            method=self.method,
            co_search_settings=self.co_search_settings,
            continue_on_closed_orbit_error=self.continue_on_closed_orbit_error,
            delta0=self.delta0,
            zeta0=self.zeta0,
            zeta_shift=self.zeta_shift,
            steps_R_matrix=self.steps_R_matrix,
            W_matrix=self.W_matrix,
            R_matrix=self.R_matrix,
            co_guess=self.co_guess,
            delta_disp=self.delta_disp,
            symplectify=self.symplectify,
            matrix_responsiveness_tol=self.matrix_responsiveness_tol,
            matrix_stability_tol=self.matrix_stability_tol,
            start=self.start,
            end=self.end,
            num_turns=self.num_turns,
            co_search_at=self.co_search_at,
            search_for_t_rev=self.search_for_t_rev,
            spin=self.spin,
            num_turns_search_t_rev=self.num_turns_search_t_rev,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            compute_R_element_by_element=self.compute_R_element_by_element,
            only_markers=self.only_markers,
            only_orbit=self.only_orbit,
            periodic_mode=self.periodic_mode,
            include_collective=self.include_collective,
            initial_particles=self._initial_particles,
        )
        self.init = self.periodic_init_data.init
        self.R_matrix = self.periodic_init_data.R_matrix
        self.steps_R_matrix = self.periodic_init_data.steps_R_matrix
        self.eigenvalues = self.periodic_init_data.eigenvalues
        self.Rot = self.periodic_init_data.Rot
        self.RR_ebe = self.periodic_init_data.RR_ebe

    def init_for_only_twiss_init(self):

        if self.reverse:
            return self.init.reverse()
        return self.init

    def propagate_from_init(self):

        return _propagate_twiss_from_init(
            line=self.line,
            init=self.init,
            start=self.start,
            end=self.end,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            delta_disp=self.delta_disp,
            use_full_inverse=self.use_full_inverse,
            hide_thin_groups=self.hide_thin_groups,
            only_markers=self.only_markers,
            only_orbit=self.only_orbit,
            spin=self.spin,
            compute_lattice_functions=self.compute_lattice_functions,
            continue_if_lost=self._continue_if_lost,
            keep_tracking_data=self._keep_tracking_data,
            keep_initial_particles=self._keep_initial_particles,
            initial_particles=self._initial_particles,
            ebe_monitor=self._ebe_monitor)

    def finish_result(self, twiss_res):

        self.add_periodic_solution_data_to(twiss_res)
        self.add_chromatic_functions_to(twiss_res)
        self.add_radiation_analysis_to(twiss_res)
        _apply_4d_longitudinal_result_convention(
            twiss_res=twiss_res, method=self.method)
        twiss_res = self.set_values_at(twiss_res)
        self.add_strengths_and_radiation_integrals_to(twiss_res)
        self.add_spin_polarization_to(twiss_res)
        self.add_edwards_teng_coupling_to(twiss_res)
        self.add_metadata_to(twiss_res)
        twiss_res = self.reverse_result_if_needed(twiss_res)
        self.align_open_phases_with_init(twiss_res)
        self.add_measured_revolution_period_if_requested(twiss_res)
        twiss_res = self.extend_to_multiple_turns_if_needed(twiss_res)
        twiss_res = self.select_at_elements(twiss_res)
        self.add_periodicity_and_completed_init_to(twiss_res)

        return twiss_res

    def add_periodic_solution_data_to(self, twiss_res):

        if self.skip_global_quantities or self.only_orbit:
            return

        _add_periodic_solution_data_to_base_twiss(
            line=self.line,
            twiss_res=twiss_res,
            method=self.method,
            R_matrix=self.R_matrix,
            steps_R_matrix=self.steps_R_matrix,
            RR_ebe=self.RR_ebe,
            eigenvalues=self.eigenvalues,
            Rot=self.Rot)

    def add_chromatic_functions_to(self, twiss_res):

        _add_chromatic_functions_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            init=self.init,
            chrom=self.chrom,
            periodic=self.periodic,
            only_orbit=self.only_orbit,
            delta_chrom=self.delta_chrom,
            delta0=self.delta0,
            zeta0=self.zeta0,
            steps_R_matrix=self.steps_R_matrix,
            matrix_responsiveness_tol=self.matrix_responsiveness_tol,
            matrix_stability_tol=self.matrix_stability_tol,
            symplectify=self.symplectify,
            method=self.method,
            use_full_inverse=self.use_full_inverse,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            delta_disp=self.delta_disp,
            zeta_disp=self.zeta_disp,
            start=self.start,
            end=self.end,
            num_turns=self.num_turns,
            hide_thin_groups=self.hide_thin_groups,
            only_markers=self.only_markers,
            periodic_mode=self.periodic_mode,
            include_collective=self.include_collective)

    def add_radiation_analysis_to(self, twiss_res):

        _add_radiation_analysis_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            radiation_analysis=self.radiation_analysis,
            only_orbit=self.only_orbit,
            method=self.method,
            steps_R_matrix=self.steps_R_matrix,
            matrix_responsiveness_tol=self.matrix_responsiveness_tol,
            start=self.start,
            end=self.end,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            step_W_sigma=self.step_W_sigma,
            zeta_shift=self.zeta_shift,
            only_markers=self.only_markers,
            radiation_method=self.radiation_method)

    def set_values_at(self, twiss_res):

        return _set_twiss_result_values_at(
            twiss_res=twiss_res,
            values_at_element_exit=self.values_at_element_exit)

    def add_strengths_and_radiation_integrals_to(self, twiss_res):

        _add_strengths_and_radiation_integrals_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            strengths=self.strengths,
            radiation_integrals=self.radiation_integrals)

    def add_spin_polarization_to(self, twiss_res):

        _add_spin_polarization_to_twiss_result(
            line=self.line,
            twiss_res=twiss_res,
            method=self.method,
            polarization_analysis=self.polarization_analysis)

    def add_edwards_teng_coupling_to(self, twiss_res):

        _add_edwards_teng_coupling_to_twiss_result(
            twiss_res=twiss_res,
            coupling_edw_teng=self.coupling_edw_teng,
            periodic=self.periodic,
            reverse=self.reverse)

    def add_metadata_to(self, twiss_res):

        _add_base_twiss_metadata(
            line=self.line,
            twiss_res=twiss_res,
            method=self.method,
            radiation_method=self.radiation_method)

    def reverse_result_if_needed(self, twiss_res):

        return _reverse_twiss_result_if_needed(
            twiss_res=twiss_res, reverse=self.reverse)

    def align_open_phases_with_init(self, twiss_res):

        if not self.periodic and not self.only_orbit:
            _align_open_twiss_phases_with_init(
                twiss_res=twiss_res, init=self.init, reverse=self.reverse)

    def add_measured_revolution_period_if_requested(self, twiss_res):

        _add_measured_revolution_period_if_requested(
            twiss_res=twiss_res,
            search_for_t_rev=self.search_for_t_rev)

    def extend_to_multiple_turns_if_needed(self, twiss_res):

        if self.num_turns <= 1:
            return twiss_res

        kwargs = _kwargs_for_composed_twiss_call(self.kwargs, self.__dict__)
        return _extend_twiss_result_to_multiple_turns(
            twiss_res=twiss_res, num_turns=self.num_turns, kwargs=kwargs)

    def select_at_elements(self, twiss_res):

        return _select_twiss_result_at_elements(
            twiss_res=twiss_res, at_elements=self.at_elements)

    def add_periodicity_and_completed_init_to(self, twiss_res):

        _add_periodicity_and_completed_init_to_twiss_result(
            twiss_res=twiss_res,
            periodic=self.periodic,
            completed_init=self.completed_init)


def _compute_twiss_segment_for_piece(kwargs, piece, init):

    return _compute_twiss_segment(
        kwargs, start=piece.start, end=piece.end, init=init)


def _handle_loop_around(kwargs):

    kwargs = kwargs.copy()

    init = kwargs.pop('init')
    start = kwargs.pop('start')
    end = kwargs.pop('end')

    line = kwargs['line']
    reverse = kwargs['reverse']

    plan = _plan_loop_around_twiss_parts(
        line=line, start=start, end=end, init=init, reverse=reverse)
    tw1, tw2, completed_init = _execute_loop_around_twiss_plan(
        kwargs=kwargs, plan=plan, init=init)

    return _combine_loop_around_twiss_tables(tw1, tw2, init, completed_init)


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


def _combine_loop_around_twiss_tables(tw1, tw2, init, completed_init):

    ele_name_init = init.element_name

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
    line = kwargs['line']
    start = kwargs.pop('start')
    end = kwargs.pop('end')
    init = kwargs.pop('init')
    reverse = kwargs.pop('reverse')

    _assert_init_inside_range_is_supported(
        line=line, start=start, end=end, init=init, reverse=reverse)

    plan = _plan_init_inside_range_twiss_parts(
        line=line, start=start, end=end, init=init, reverse=reverse)
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


def _combine_init_inside_range_twiss_tables(tw1, tw2, init):

    ele_name_init = init.element_name

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


def _enter_twiss_freeze_context(
        twiss_context, line, freeze_longitudinal, freeze_energy):

    if freeze_longitudinal:
        twiss_context.enter_context(xt.freeze_longitudinal(line))
        freeze_longitudinal = False
    elif freeze_energy:
        if not line._energy_is_frozen():
            twiss_context.enter_context(xt.line._preserve_config(line))
            line.freeze_energy(force=True) # need to force for collective lines
            freeze_energy = False

    return freeze_longitudinal, freeze_energy


def _kwargs_for_composed_twiss_call(kwargs, loc):
    """Temporary kwargs refresh for composed Twiss calls.

    This keeps existing recursive segment calls correct while the normalized
    segment engine is being extracted. It should shrink or disappear once those
    composed paths stop calling the public ``twiss_line`` wrapper.
    """

    out = kwargs.copy()

    for kk in kwargs.keys():
        if kk in loc:
            out[kk] = loc[kk]

    out.pop('input_kwargs', None)

    return out


def _compute_twiss_segment(kwargs, **overrides):

    segment_kwargs = _kwargs_for_preflighted_twiss_segment(kwargs)
    segment_kwargs.update(overrides)

    return twiss_line(**segment_kwargs)


def _kwargs_for_preflighted_twiss_segment(kwargs):

    segment_kwargs = kwargs.copy()
    segment_kwargs['disable_apertures'] = False
    segment_kwargs['freeze_longitudinal'] = False
    segment_kwargs['freeze_energy'] = False
    segment_kwargs['at_s'] = None

    return segment_kwargs


def _apply_base_twiss_reverse_range(line, start, end, reverse):

    if reverse:
        if start is not None and end is not None:
            assert (_str_to_index(line, start) >= _str_to_index(line, end)), (
                'start must be smaller than end in reverse mode')
        return end, start

    if start is not None and end is not None:
        assert _str_to_index(line, start) <= _str_to_index(line, end), (
            'start must be larger than end in forward mode')

    return start, end


def _validate_base_twiss_boundary_init(start, init):

    if start is not None and init is None:
        assert init is not None, (
            'init must be provided if start and end are used')


def _prepare_base_twiss_matrix_settings(
        line, radiation_method, matrix_responsiveness_tol,
        matrix_stability_tol, use_full_inverse):

    if matrix_responsiveness_tol is None:
        matrix_responsiveness_tol = line.matrix_responsiveness_tol
    if matrix_stability_tol is None:
        matrix_stability_tol = line.matrix_stability_tol

    if (line._radiation_model is not None
            and radiation_method != 'kick_as_co'):
        matrix_stability_tol = None
        if use_full_inverse is None:
            use_full_inverse = True

    return matrix_responsiveness_tol, matrix_stability_tol, use_full_inverse


def _prepare_base_twiss_line_and_particle_ref(
        line, particle_ref, particle_on_co, co_guess, include_collective):

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
            "Either ``particle_ref`` or ``co_guess`` must be provided")

    return line, particle_ref


def _validate_base_twiss_method(method):

    if method is None:
        method = '6d'

    assert method in ['6d', '4d'], 'Method must be ``6d`` or ``4d``'

    return method


def _validate_base_twiss_init_mode(init):

    if isinstance(init, str):
        if init in ['preserve', 'preserve_start', 'preserve_end']:
            raise ValueError(f'init={init} not anymore supported')
        assert init == 'periodic' or 'full_periodic'


def _validate_base_twiss_open_momentum_offsets(periodic, delta0, zeta0):

    if not periodic:
        if delta0 is not None or zeta0 is not None:
            raise ValueError(
                'delta0 and zeta0 cannot be provided for open twiss')


def _complete_init_for_base_twiss(
        start, end, init_at, init, line, reverse,
        x, px, y, py, zeta, delta,
        alfx, alfy, betx, bety, bets,
        dx, dpx, dy, dpy, dzeta,
        mux, muy, muzeta,
        ax_chrom, bx_chrom, ay_chrom, by_chrom,
        ddx, ddpx, ddy, ddpy,
        spin_x, spin_y, spin_z):

    init = _complete_twiss_init(
        start=start, end=end, init_at=init_at, init=init,
        line=line, reverse=reverse,
        x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
        alfx=alfx, alfy=alfy, betx=betx, bety=bety, bets=bets,
        dx=dx, dpx=dpx, dy=dy, dpy=dpy, dzeta=dzeta,
        mux=mux, muy=muy, muzeta=muzeta,
        ax_chrom=ax_chrom, bx_chrom=bx_chrom, ay_chrom=ay_chrom,
        by_chrom=by_chrom,
        ddx=ddx, ddpx=ddpx, ddy=ddy, ddpy=ddpy,
        spin_x=spin_x, spin_y=spin_y, spin_z=spin_z
    )
    completed_init = (init.copy() if hasattr(init, 'copy') else init)

    return init, completed_init


def _compute_periodic_twiss_init_and_data(
        line, particle_on_co, particle_ref, method, co_search_settings,
        continue_on_closed_orbit_error, delta0, zeta0, zeta_shift,
        steps_R_matrix, W_matrix, R_matrix, co_guess, delta_disp,
        symplectify, matrix_responsiveness_tol, matrix_stability_tol,
        start, end, num_turns, co_search_at, search_for_t_rev, spin,
        num_turns_search_t_rev, nemitt_x, nemitt_y, step_W_sigma,
        compute_R_element_by_element, only_markers, only_orbit,
        periodic_mode, include_collective, initial_particles):

    assert not initial_particles

    steps_R_matrix = _complete_steps_r_matrix_with_default(steps_R_matrix)

    periodic_solution = _find_periodic_solution(
        line=line, particle_on_co=particle_on_co,
        particle_ref=particle_ref, method=method,
        co_search_settings=co_search_settings,
        continue_on_closed_orbit_error=continue_on_closed_orbit_error,
        delta0=delta0, zeta0=zeta0, zeta_shift=zeta_shift,
        steps_R_matrix=steps_R_matrix,
        W_matrix=W_matrix, R_matrix=R_matrix,
        co_guess=co_guess,
        delta_disp=delta_disp, symplectify=symplectify,
        matrix_responsiveness_tol=matrix_responsiveness_tol,
        matrix_stability_tol=matrix_stability_tol,
        start=start, end=end,
        num_turns=num_turns,
        co_search_at=co_search_at,
        search_for_t_rev=search_for_t_rev,
        spin=spin,
        num_turns_search_t_rev=num_turns_search_t_rev,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        step_W_sigma=step_W_sigma,
        compute_R_element_by_element=compute_R_element_by_element,
        only_markers=only_markers,
        only_orbit=only_orbit,
        periodic_mode=periodic_mode,
        include_collective=include_collective,
    )

    return _PeriodicTwissInitData.from_periodic_solution(periodic_solution)


def _propagate_twiss_from_init(
        line, init, start, end, nemitt_x, nemitt_y, step_W_sigma,
        delta_disp, use_full_inverse, hide_thin_groups, only_markers,
        only_orbit, spin, compute_lattice_functions, continue_if_lost,
        keep_tracking_data, keep_initial_particles, initial_particles,
        ebe_monitor):

    return _twiss_open(
        line=line,
        init=init,
        start=start, end=end,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        step_W_sigma=step_W_sigma,
        delta_disp=delta_disp,
        use_full_inverse=use_full_inverse,
        hide_thin_groups=hide_thin_groups,
        only_markers=only_markers,
        only_orbit=only_orbit,
        spin=spin,
        compute_lattice_functions=compute_lattice_functions,
        _continue_if_lost=continue_if_lost,
        _keep_tracking_data=keep_tracking_data,
        _keep_initial_particles=keep_initial_particles,
        _initial_particles=initial_particles,
        _ebe_monitor=ebe_monitor)


def _add_periodic_solution_data_to_base_twiss(
        line, twiss_res, method, R_matrix, steps_R_matrix, RR_ebe,
        eigenvalues, Rot):

    twiss_res._data['R_matrix'] = R_matrix
    twiss_res._data['steps_R_matrix'] = steps_R_matrix
    twiss_res._data['steps_r_matrix'] = steps_R_matrix # deprecated
    twiss_res._data['R_matrix_ebe'] = RR_ebe

    _add_ring_quantities(line=line, twiss_res=twiss_res, method=method)

    twiss_res._data['eigenvalues'] = eigenvalues.copy()
    twiss_res._data['rotation_matrix'] = Rot.copy()


def _add_chromatic_functions_to_twiss_result(
        line, twiss_res, init, chrom, periodic, only_orbit, delta_chrom,
        delta0, zeta0, steps_R_matrix, matrix_responsiveness_tol,
        matrix_stability_tol, symplectify, method, use_full_inverse,
        nemitt_x, nemitt_y, step_W_sigma, delta_disp, zeta_disp,
        start, end, num_turns, hide_thin_groups, only_markers,
        periodic_mode, include_collective):

    if only_orbit:
        return

    if not (chrom is True or (chrom is None and periodic)):
        return

    cols_chrom, scalars_chrom = _get_chromatic_functions(
        line=line,
        init=init,
        delta_chrom=delta_chrom,
        delta0=delta0,
        zeta0=zeta0,
        steps_R_matrix=steps_R_matrix,
        matrix_responsiveness_tol=matrix_responsiveness_tol,
        matrix_stability_tol=matrix_stability_tol,
        symplectify=symplectify,
        method=method,
        use_full_inverse=use_full_inverse,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        on_momentum_twiss_res=twiss_res,
        step_W_sigma=step_W_sigma,
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


def _add_radiation_analysis_to_twiss_result(
        line, twiss_res, radiation_analysis, only_orbit, method,
        steps_R_matrix, matrix_responsiveness_tol, start, end, nemitt_x,
        nemitt_y, step_W_sigma, zeta_shift, only_markers, radiation_method):

    if not radiation_analysis or only_orbit:
        return

    assert 'R_matrix' in twiss_res._data
    if method == '4d':
        raise ValueError('method="4d" not supported for radiation_analysis=True')

    with xt.line._preserve_config(line):
        with xt.line._preserve_track_flags(line):
            line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = False
            line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = False
            _, RR, _, _, _, RR_ebe = _find_periodic_solution(
                line=line,
                particle_ref=None,
                method='6d',
                particle_on_co=twiss_res.particle_on_co,
                co_search_settings=None,
                continue_on_closed_orbit_error=None,
                co_guess=None,
                steps_R_matrix=steps_R_matrix,
                symplectify=False,
                matrix_responsiveness_tol=matrix_responsiveness_tol,
                matrix_stability_tol=None,
                start=start, end=end,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                step_W_sigma=step_W_sigma,
                delta0=None, zeta0=None, zeta_shift=zeta_shift,
                W_matrix=None, R_matrix=None,
                delta_disp=None,
                compute_R_element_by_element=True,
                only_markers=only_markers,
                factor_adapt_steps=0.03 # 10 times smaller than for optics
                                        # to campture small damping effects
                )

    eneloss_damp_res = _get_eneloss_and_damping_rates(
            particle_on_co=twiss_res.particle_on_co, R_matrix=RR,
            W_matrix=twiss_res.W_matrix,
            px_co=twiss_res.px, py_co=twiss_res.py,
            ptau_co=twiss_res.ptau, t_rev0=twiss_res.t_rev0,
            line=line, radiation_method=radiation_method)
    twiss_res._data.update(eneloss_damp_res)

    for kk in ['angle_rad', 'angle', 'rot_s_rad', 'length', 'radiation_flag']:
        if kk not in twiss_res._data:
            aa = line.attr[kk]
            twiss_res[kk] = np.concatenate([aa, [aa[0]*0]])

    # Equilibrium emittances
    if radiation_method == 'kick_as_co':
        eq_emitts = _get_equilibrium_emittance_kick_as_co(
            twiss_res=twiss_res,
            damping_constants_turns=eneloss_damp_res['damping_constants_turns'],
            radiation_method=radiation_method)
        twiss_res._data.update(eq_emitts)
    elif radiation_method == 'full':
        eq_emitts = _get_equilibrium_emittance_full(
            twiss_res=twiss_res,
            R_matrix_ebe=RR_ebe,
            radiation_method=radiation_method)
        twiss_res._data.update(eq_emitts)


def _apply_4d_longitudinal_result_convention(twiss_res, method):

    if method == '4d' and 'muzeta' in twiss_res._data:
        twiss_res.muzeta[:] = 0
        if 'qs' in twiss_res._data:
            twiss_res._data['qs'] = 0


def _set_twiss_result_values_at(twiss_res, values_at_element_exit):

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

    return twiss_res


def _add_strengths_and_radiation_integrals_to_twiss_result(
        line, twiss_res, strengths, radiation_integrals):

    if strengths or radiation_integrals:
        _add_strengths_to_twiss_res(twiss_res, line)

    if radiation_integrals:
        twiss_res._get_radiation_integrals(add_to_tw=True)


def _add_spin_polarization_to_twiss_result(
        line, twiss_res, method, polarization_analysis):

    if polarization_analysis:
        _get_spin_polarization(twiss_res, line, method)


def _add_edwards_teng_coupling_to_twiss_result(
        twiss_res, coupling_edw_teng, periodic, reverse):

    if not coupling_edw_teng:
        return

    if not periodic:
        raise ValueError(
            'Computing Edwards-Teng coupling elements is only supported for periodic lines.'
        )
    if reverse:
        raise NotImplementedError(
            'Computing Edwards-Teng coupling elements in reverse mode is not '
            'yet implemented.'
        )

    coupling_result = _get_coupling_elements_edwards_teng(
        W_matrix=twiss_res['W_matrix'],
        mux=twiss_res['mux'],
        muy=twiss_res['muy'],
        qx=twiss_res['qx'],
        qy=twiss_res['qy']
    )
    for kk in coupling_result:
        twiss_res[kk] = coupling_result[kk]


def _add_base_twiss_metadata(line, twiss_res, method, radiation_method):

    twiss_res._data['method'] = method
    twiss_res._data['radiation_method'] = radiation_method
    twiss_res._data['reference_frame'] = 'proper'
    twiss_res._data['line_config'] = dict(line.config.copy())


def _reverse_twiss_result_if_needed(twiss_res, reverse):

    if reverse:
        return twiss_res.reverse()

    return twiss_res


def _add_measured_revolution_period_if_requested(twiss_res, search_for_t_rev):

    if not search_for_t_rev:
        return

    # Recompute t_rev0 to support case with only_orbit=True
    line_length = twiss_res.s[-1]
    beta0 = twiss_res.particle_on_co.beta0[0]
    t_rev_0 = line_length/clight/beta0
    twiss_res._data['t_rev'] = t_rev_0 - (
        twiss_res.zeta[-1] - twiss_res.zeta[0])/(beta0*clight)
    twiss_res._data['T_rev'] = twiss_res._data['t_rev'] # deprecated


def _extend_twiss_result_to_multiple_turns(twiss_res, num_turns, kwargs):

    kwargs.pop('num_turns')
    kwargs.pop('init')
    kwargs.pop('start')
    kwargs.pop('end')

    tw_mt = _multiturn_twiss(tw0=twiss_res, num_turns=num_turns,
                             kwargs=kwargs)
    tw_mt._data['_tw0'] = twiss_res
    return tw_mt


def _select_twiss_result_at_elements(twiss_res, at_elements):

    if at_elements is not None:
        return twiss_res.rows[at_elements]

    return twiss_res


def _add_periodicity_and_completed_init_to_twiss_result(
        twiss_res, periodic, completed_init):

    twiss_res['periodic'] = periodic
    twiss_res['completed_init'] = completed_init
    twiss_res._sort_col_names()


def _align_open_twiss_phases_with_init(twiss_res, init, reverse):

    # Start phase advance with provided init
    if ((twiss_res._orientation == 'forward' and not reverse)
            or (twiss_res._orientation == 'backward' and reverse)):
        twiss_res.muzeta += init.muzeta - twiss_res.muzeta[0]
        if 'dzeta' in twiss_res._data:
            twiss_res.dzeta += init.dzeta - twiss_res.dzeta[0]
        if 'mux' in twiss_res._data:
            twiss_res.mux += init.mux - twiss_res.mux[0]
            twiss_res.muy += init.muy - twiss_res.muy[0]
    elif ((twiss_res._orientation == 'forward' and reverse)
        or (twiss_res._orientation == 'backward' and not reverse)):
        twiss_res.muzeta += init.muzeta - twiss_res.muzeta[-1]
        if 'dzeta' in twiss_res._data:
            twiss_res.dzeta += init.dzeta - twiss_res.dzeta[-1]
        if 'mux' in twiss_res._data:
            twiss_res.mux += init.mux - twiss_res.mux[-1]
            twiss_res.muy += init.muy - twiss_res.muy[-1]


def _prepare_kwargs_for_full_periodic_twiss(kwargs):

    kwargs = kwargs.copy()
    kwargs.pop('init')
    kwargs.pop('start')
    kwargs.pop('end')
    kwargs.pop('init_at')

    return kwargs


def _compute_full_periodic_twiss_init(kwargs, start, init_at):
    """Compute the full periodic Twiss and extract the requested init."""

    tw = _compute_twiss_segment(kwargs) # Periodic twiss of the full line

    return tw.get_twiss_init(init_at or start)


def _compute_one_turn_twiss_from_start(kwargs, line, start, init, betx, bety):

    kwargs = kwargs.copy()
    kwargs.pop('start')

    if (init is None or init == 'periodic') and betx is None and bety is None:
        return _compute_periodic_one_turn_twiss_from_start(
            kwargs=kwargs, start=start)

    return _compute_open_one_turn_twiss_from_start(
        kwargs=kwargs, line=line, start=start)


def _compute_periodic_one_turn_twiss_from_start(kwargs, start):

    tw = _compute_twiss_segment(kwargs)
    t1 = tw.rows[start:]
    t2 = tw.rows[:start]
    out = xt.TwissTable.concatenate([t1, t2])
    out.zero_at(out.name[0])
    out.name[-1] = '_end_point'
    out['periodic'] = True
    out['completed_init'] = tw.completed_init
    return out


def _compute_open_one_turn_twiss_from_start(kwargs, line, start):

    kwargs = kwargs.copy()
    kwargs.pop('end')
    plan = _plan_open_one_turn_twiss(line=line, start=start,
                                     reverse=kwargs['reverse'])

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


def _multiturn_twiss(tw0, num_turns, kwargs):

    twisses_to_merge = _compute_multiturn_twiss_parts(
        tw0=tw0, num_turns=num_turns, kwargs=kwargs)

    return _combine_multiturn_twiss_tables(twisses_to_merge)


def _compute_multiturn_twiss_parts(tw0, num_turns, kwargs):

    tw_curr = tw0
    twisses_to_merge = []

    for i_turn in range(num_turns):

        twisses_to_merge.append(
            _multiturn_start_row(tw_curr=tw_curr, i_turn=i_turn))
        twisses_to_merge.append(tw_curr)

        if i_turn == num_turns - 1:
            break # need n-1 twisses

        tw_curr = _continue_multiturn_twiss(tw_curr=tw_curr, kwargs=kwargs)

    return twisses_to_merge


def _multiturn_start_row(tw_curr, i_turn):

    tw_start_turn = tw_curr.rows[0]
    tw_start_turn.name[0] = f'_turn_{i_turn}'
    return tw_start_turn


def _continue_multiturn_twiss(tw_curr, kwargs):

    line = kwargs['line']
    tini1 = tw_curr.get_twiss_init(-1)
    tini1.element_name = tw_curr.name[0]

    return twiss_line(
        **kwargs, init=tini1, start=tw_curr.name[0],
        end=line._element_names_unique[-1])


def _combine_multiturn_twiss_tables(twisses_to_merge):

    return xt.TwissTable.concatenate(twisses_to_merge)
