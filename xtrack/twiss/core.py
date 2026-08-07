# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from warnings import warn

import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0
from scipy.constants import e as qe
from scipy.constants import electron_volt

if hasattr(np, 'trapezoid'): # numpy >= 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz

import xobjects as xo
from .. import linear_normal_form as lnf
from ..general import _print, DEPRECATION_INFO_PREP_1_0
from .closed_orbit import ClosedOrbitSearchError, find_closed_orbit_line
from .twiss_init import TwissInit, _W_phys2norm
from .element_indexing import _str_to_index
from .twiss_table import TwissTable
from .transfer_matrices import _complete_steps_r_matrix_with_default
from .beam_covariance import _build_sigma_table
from .trajectory_curvatures import _get_trajectory_curvatures
from .spin import _get_spin_polarization
from .non_linear_chromaticity import get_non_linear_chromaticity
from .strengths import _add_strengths_to_twiss_res
from .constants import (
    AT_TURN_FOR_TWISS,
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
    if at_s is not None:
        warn('`at_s` keyword is deprecated and will be removed in future versions. \n'
        'The same functionality can be achieved making a shallow copy of the line '
        '(e.g. `line_copy = line.copy(shallow=True)`), using the`line.cut_at_s(...)` '
        ' functionality and then calling line_copy.twiss(...) on the cut line.'
        + DEPRECATION_INFO_PREP_1_0,
        FutureWarning)

    if at_elements is not None:
        warn('`at_elements` keyword is deprecated and will be removed in future versions. \n'
        'The same functionality can be achieved by selecting the desired names after computing '
        'the twiss, e.g. `line.twiss(...).rows[["ele1", "ele2", "ele3"]]`. '
        'Regular expressions are also supported for the selection of element names, '
        'e.g. `line.twiss(...).rows["quad.*"]`.'
        + DEPRECATION_INFO_PREP_1_0,
        FutureWarning)

    if compute_chromatic_properties is not None:
        warn('The `compute_chromatic_properties` keyword is deprecated and will be removed in future versions. \n'
             'Please use `chrom` instead, which has the same behavior.'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)
        chrom = compute_chromatic_properties

    if r_sigma:
        warn('The `r_sigma` keyword is deprecated and will be removed in future versions. \n'
             'Please use `step_W_sigma` instead, which has the same behavior.'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)
        step_W_sigma = r_sigma

    if freeze_energy:
        warn('The `freeze_energy` keyword is deprecated and will be removed in future versions. \n'
             'You can use twiss(method="4d", ...) to suppress the energy kick from RF cavities'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)

    if freeze_longitudinal:
        warn('The `freeze_longitudinal` keyword is deprecated and will be removed in future versions. \n'
             'You can use twiss(method="4d", ...) to suppress the energy kick from RF cavities'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)

    if polarization:
        warn('The `polarization` keyword is deprecated and will be removed in future versions. \n'
             'Please use `polarization_analysis` instead, which has the same behavior.'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)
        polarization_analysis = polarization

    if eneloss_and_damping:
        warn('The `eneloss_and_damping` keyword is deprecated and will be removed in future versions. \n'
             'Please use `radiation_analysis` instead, which has the same behavior.'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)
        radiation_analysis = eneloss_and_damping

    if steps_r_matrix is not None:
        warn('The `steps_r_matrix` keyword is deprecated and will be removed in future versions. \n'
             'Please use `steps_R_matrix` instead, which has the same behavior.'
             + DEPRECATION_INFO_PREP_1_0,
             FutureWarning)
        steps_R_matrix = steps_r_matrix

    input_kwargs = locals().copy()

    # defaults
    step_W_sigma=(step_W_sigma or 0.01)
    nemitt_x=(nemitt_x or 1e-6)
    nemitt_y=(nemitt_y or 1e-6)
    delta_disp=(delta_disp or 1e-5)
    delta_chrom=(delta_chrom or 5e-5)
    zeta_disp=(zeta_disp or 1e-3)
    zeta_shift=(zeta_shift or 0.0)
    values_at_element_exit=(values_at_element_exit or False)
    continue_on_closed_orbit_error=(continue_on_closed_orbit_error or False)
    freeze_longitudinal=(freeze_longitudinal or False)
    radiation_method=(radiation_method or None)
    spin=(spin or False)
    polarization_analysis=(polarization_analysis or False)
    radiation_integrals=(radiation_integrals or False)
    radiation_analysis=(radiation_analysis or False)
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
    chrom=(chrom if chrom is not None else None)
    num_turns = (num_turns or 1)
    disable_apertures = (disable_apertures if disable_apertures is not None else True)

    if disable_apertures:
        if not (line.tracker.track_flags.XS_FLAG_IGNORE_GLOBAL_APERTURE
                and line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE):
            with xt.line._preserve_track_flags(line):
                line.tracker.track_flags.XS_FLAG_IGNORE_GLOBAL_APERTURE = True
                line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE = True
                out = twiss_line(**input_kwargs)
                return _add_action_in_res(out, input_kwargs)

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
        assert x is None, '``x`` not supported for periodic twiss'
        assert px is None, '``px`` not supported for periodic twiss'
        assert y is None, '``y`` not supported for periodic twiss'
        assert py is None, '``py`` not supported for periodic twiss'
        assert zeta is None, '``zeta`` not supported for periodic twiss'
        assert delta is None, '``delta`` not supported for periodic twiss'

    if freeze_longitudinal:
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        kwargs.pop('freeze_longitudinal')

        with xt.freeze_longitudinal(line):
            return _add_action_in_res(twiss_line(**kwargs), input_kwargs)
    elif freeze_energy:
        if not line._energy_is_frozen():
            kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
            kwargs.pop('freeze_energy')
            with xt.line._preserve_config(line):
                line.freeze_energy(force=True) # need to force for collective lines
                return _add_action_in_res(
                    twiss_line(freeze_energy=False, **kwargs), input_kwargs)

    if method == '4d' and not line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK:
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        with xt.line._preserve_track_flags(line):
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True
            return _add_action_in_res(twiss_line(**kwargs), input_kwargs)

    if at_s is not None:
        if reverse:
            raise NotImplementedError('``at_s`` not implemented for ``reverse``=True')
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
        kwargs.pop('strengths')
        res = twiss_line(line=auxtracker.line,
                        at_elements=names_inserted_markers,
                        matrix_responsiveness_tol=matrix_responsiveness_tol,
                        matrix_stability_tol=matrix_stability_tol,
                        strengths=True,
                        **kwargs)
        return _add_action_in_res(res, input_kwargs)

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
        kwargs = _updated_kwargs_from_locals(kwargs, locals().copy())
        assert radiation_method in ['full', 'kick_as_co', 'scale_as_co']
        assert freeze_longitudinal is False
        if (radiation_method == 'kick_as_co' and (
            not line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST)):
            with xt.line._preserve_track_flags(line):
                line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = True
                return _add_action_in_res(twiss_line(**kwargs), input_kwargs)
        elif (radiation_method == 'scale_as_co' and (
            not hasattr(line.config, 'XTRACK_SYNRAD_SCALE_SAME_AS_FIRST') or
            not line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST)):
            with xt.line._preserve_config(line):
                line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = True
                return _add_action_in_res(twiss_line(**kwargs), input_kwargs)

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

    init = _complete_twiss_init(
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
    completed_init = (init.copy() if hasattr(init, 'copy') else init)

    # clean quantities embedded in init
    init_at=None
    x=None; px=None; y=None; py=None; zeta=None; delta=None
    alfx=None; alfy=None; betx=None; bety=None; bets=None
    dx=None; dpx=None; dy=None; dpy=None; dzeta=None
    mux=None; muy=None; muzeta=None
    ax_chrom=None; bx_chrom=None; ay_chrom=None; by_chrom=None
    ddx=None; ddpx=None; ddy=None; ddpy=None
    spin_x=None; spin_y=None; spin_z=None

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
            "Either ``particle_ref`` or ``co_guess`` must be provided")

    if method is None:
        method = '6d'

    assert method in ['6d', '4d'], 'Method must be ``6d`` or ``4d``'

    if isinstance(init, str):
        if init in ['preserve', 'preserve_start', 'preserve_end']:
            raise ValueError(f'init={init} not anymore supported')
        assert init == 'periodic' or 'full_periodic'

    if not periodic:
        if delta0 is not None or zeta0 is not None:
            raise ValueError(
                'delta0 and zeta0 cannot be provided for open twiss')

    if periodic:

        assert not _initial_particles

        steps_R_matrix = _complete_steps_r_matrix_with_default(steps_R_matrix)

        init, R_matrix, steps_R_matrix, eigenvalues, Rot, RR_ebe = _find_periodic_solution(
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
            nemitt_x=nemitt_x, nemitt_y=nemitt_y, step_W_sigma=step_W_sigma,
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
        assert periodic, '``only_twiss_init`` can only be used in periodic mode'
        if reverse:
            return init.reverse()
        else:
            return init

    if only_markers and radiation_analysis:
        raise NotImplementedError(
            '``only_markers`` not implemented for ``radiation_analysis``')

    twiss_res = _twiss_open(
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
        _continue_if_lost=_continue_if_lost,
        _keep_tracking_data=_keep_tracking_data,
        _keep_initial_particles=_keep_initial_particles,
        _initial_particles=_initial_particles,
        _ebe_monitor=_ebe_monitor)

    if not skip_global_quantities and not only_orbit:
        twiss_res._data['R_matrix'] = R_matrix
        twiss_res._data['steps_R_matrix'] = steps_R_matrix
        twiss_res._data['steps_r_matrix'] = steps_R_matrix # deprecated
        twiss_res._data['R_matrix_ebe'] = RR_ebe

        _get_global_quantities(line=line, twiss_res=twiss_res, method=method)

        twiss_res._data['eigenvalues'] = eigenvalues.copy()
        twiss_res._data['rotation_matrix'] = Rot.copy()

    if (not only_orbit and (
        (chrom is True)
        or (chrom is None and periodic))):

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



    if radiation_analysis and not only_orbit:
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
                    nemitt_x=nemitt_x, nemitt_y=nemitt_y, step_W_sigma=step_W_sigma,
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
            eq_emitts = _get_equilibrium_emittance_full(twiss_res=twiss_res,
                        R_matrix_ebe=RR_ebe,
                        radiation_method=radiation_method)
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
        twiss_res._get_radiation_integrals(add_to_tw=True)

    if polarization_analysis:
        _get_spin_polarization(twiss_res, line, method)

    if coupling_edw_teng:
        if not periodic:
            raise ValueError(
                'Computing Edwards-Teng coupling elements is only supported for periodic lines.'
            )
        if reverse:
            raise NotImplementedError(
                'Computing Edwards-Teng coupling elements in reverse mode is not '
                'yet implemented.'
            )
        delta = twiss_res['delta']
        betx1, betx2 = twiss_res['betx1'], twiss_res['betx2']
        bety1, bety2 = twiss_res['bety1'], twiss_res['bety2']
        alfx1, alfx2 = twiss_res['alfx1'], twiss_res['alfx2']
        alfy1, alfy2 = twiss_res['alfy1'], twiss_res['alfy2']
        coupling_result = _get_coupling_elements_edwards_teng(
            W_matrix=twiss_res['W_matrix'],
            mux=twiss_res['mux'],
            muy=twiss_res['muy'],
            qx=twiss_res['qx'],
            qy=twiss_res['qy']
        )
        for kk in coupling_result:
            twiss_res[kk] = coupling_result[kk]

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

    if search_for_t_rev:
        # Recompute t_rev0 to support case with only_orbit=True
        line_length = twiss_res.s[-1]
        beta0 = twiss_res.particle_on_co.beta0[0]
        t_rev_0 = line_length/clight/beta0
        twiss_res._data['t_rev'] = t_rev_0 - (
            twiss_res.zeta[-1] - twiss_res.zeta[0])/(beta0*clight)
        twiss_res._data['T_rev'] = twiss_res._data['t_rev'] # deprecated

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

    # Sort col names
    twiss_res._sort_col_names()

    return _add_action_in_res(twiss_res, input_kwargs)

def _twiss_open(
        line,
        init,
        start,
        end,
        nemitt_x,
        nemitt_y,
        step_W_sigma,
        delta_disp,
        use_full_inverse,
        hide_thin_groups=False,
        only_markers=False,
        only_orbit=False,
        spin=False,
        compute_lattice_functions=True,
        _continue_if_lost=False,
        _keep_tracking_data=False,
        _keep_initial_particles=False,
        _initial_particles=None,
        _ebe_monitor=None,
):
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
            '``init`` must be given at the start or end of the specified element range.')

    ctx2np = line._context.nparray_from_context_array

    gemitt_x = nemitt_x/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    gemitt_y = nemitt_y/particle_on_co._xobject.beta0[0]/particle_on_co._xobject.gamma0[0]
    scale_transverse_x = np.sqrt(gemitt_x)*step_W_sigma
    scale_transverse_y = np.sqrt(gemitt_y)*step_W_sigma
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
        part_for_twiss.ax = particle_on_co._xobject.ax[0]
        part_for_twiss.ay = particle_on_co._xobject.ay[0]
        if spin:
            part_for_twiss.spin_x = particle_on_co._xobject.spin_x[0]
            part_for_twiss.spin_y = particle_on_co._xobject.spin_y[0]
            part_for_twiss.spin_z = particle_on_co._xobject.spin_z[0]

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

    with xt.line._preserve_config(line):
        if spin:
            # Spin is behind the same compile flag as synchrotron radiation
            line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
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
    kin_xp_co = line.record_last_track.kin_xp[0, i_start:i_stop+1].copy()
    kin_yp_co = line.record_last_track.kin_yp[0, i_start:i_stop+1].copy()
    if spin:
        spin_x_co = line.record_last_track.spin_x[0, i_start:i_stop+1].copy()
        spin_y_co = line.record_last_track.spin_y[0, i_start:i_stop+1].copy()
        spin_z_co = line.record_last_track.spin_z[0, i_start:i_stop+1].copy()

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
        'kin_xp': kin_xp_co,
        'kin_yp': kin_yp_co,
        'kin_xprime': kin_xp_co,
        'kin_yprime': kin_yp_co,
        'env_name': name_co_env,
    })
    if spin:
        twiss_res_element_by_element.update({
            'spin_x': spin_x_co,
            'spin_y': spin_y_co,
            'spin_z': spin_z_co,
        })

    if not only_orbit and compute_lattice_functions:
        lattice_functions, i_replace = _get_lattice_functions(Ws, use_full_inverse, s_co)
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

    line_length = line.tracker._tracker_data_base.line_length
    twiss_res._data['line_length'] = line_length
    twiss_res._data['circumference'] = line_length # deprecated
    twiss_res._data['_orientation'] = twiss_orientation

    return twiss_res


def _get_lattice_functions(Ws, use_full_inverse, s_co):

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
        (betx, alfx, gamx, bety, alfy, gamy, bety1, betx2, alfy1, alfx2, gamy1,
        gamx2) = _extract_twiss_parameters_with_inverse(Ws)
    else:
        betx = Ws[:, 0, 0]**2 + Ws[:, 0, 1]**2
        bety = Ws[:, 2, 2]**2 + Ws[:, 2, 3]**2

        gamx = Ws[:, 1, 0]**2 + Ws[:, 1, 1]**2
        gamy = Ws[:, 3, 2]**2 + Ws[:, 3, 3]**2

        alfx = -Ws[:, 0, 0] * Ws[:, 1, 0] - Ws[:, 0, 1] * Ws[:, 1, 1]
        alfy = -Ws[:, 2, 2] * Ws[:, 3, 2] - Ws[:, 2, 3] * Ws[:, 3, 3]

        bety1 = Ws[:, 2, 0]**2 + Ws[:, 2, 1]**2
        betx2 = Ws[:, 0, 2]**2 + Ws[:, 0, 3]**2

        alfx2 = -Ws[:, 0, 2] * Ws[:, 1, 2] - Ws[:, 0, 3] * Ws[:, 1, 3]
        alfy1 = -Ws[:, 2, 0] * Ws[:, 3, 0] - Ws[:, 2, 1] * Ws[:, 3, 1]

        gamx2 = Ws[:, 1, 2]**2 + Ws[:, 1, 3]**2
        gamy1 = Ws[:, 3, 0]**2 + Ws[:, 3, 1]**2

    betx1 = betx
    bety2 = bety

    alfx1 = alfx
    alfy2 = alfy

    gamx1 = gamx
    gamy2 = gamy


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
        'alfx1': alfx1,
        'alfy1': alfy1,
        'alfx2': alfx2,
        'alfy2': alfy2,
        'gamx1': gamx1,
        'gamy1': gamy1,
        'gamx2': gamx2,
        'gamy2': gamy2,
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


def _get_coupling_elements_edwards_teng(
        W_matrix: np.ndarray,
        mux: np.ndarray,
        muy: np.ndarray,
        qx: float = None,
        qy: float = None,
):
    """Compute coupling matrix elements using the Edwards-Teng method.

    """

    # This computes edwards-teng parameters from full one-turn W matrix at all locations
    edw_teng_cols = _edwards_teng_from_one_turn_at_all_locations(W_matrix, qx, qy)
    #
    # The following instead computes from the one-turn R matrix at one location
    # and then propagates along the ring (observed to be less precise)

    # # R matrix of the full ring (4D)
    # Rot = np.zeros(shape=(6, 6), dtype=np.float64)
    # Rot[0:2,0:2] = lnf.Rot2D(2 * np.pi * qx)
    # Rot[2:4,2:4] = lnf.Rot2D(2 * np.pi * qy)
    # WW0 = W_matrix[0, :, :]
    # WW0_inv = lnf.S.T @ WW0.T @ lnf.S
    # RR = WW0 @ Rot @ WW0_inv

    # # Edwards-Teng initial conditions
    # edw_teng_init = _get_edwards_teng_initial(RR)

    # # Edwards-Teng parameters along the ring
    # edw_teng_cols = _propagate_edwards_teng(
    #     WW=W_matrix, mux=mux, muy=muy,
    #     RR_ET0=edw_teng_init['RR_ET0'],
    #     betx0=edw_teng_init['betx0'],
    #     alfx0=edw_teng_init['alfx0'],
    #     bety0=edw_teng_init['bety0'],
    #     alfy0=edw_teng_init['alfy0']
    # )

    # Coupling RDTs from Edwards-Teng parameters
    rdts = _get_coupling_rdts(edw_teng_cols['r11'], edw_teng_cols['r12'],
                                  edw_teng_cols['r21'], edw_teng_cols['r22'],
                                  edw_teng_cols['betx'], edw_teng_cols['bety'],
                                  edw_teng_cols['alfx'], edw_teng_cols['alfy'])

    out = {
        'r11_edw_teng': edw_teng_cols['r11'],
        'r12_edw_teng': edw_teng_cols['r12'],
        'r21_edw_teng': edw_teng_cols['r21'],
        'r22_edw_teng': edw_teng_cols['r22'],
        'betx_edw_teng': edw_teng_cols['betx'],
        'alfx_edw_teng': edw_teng_cols['alfx'],
        'bety_edw_teng': edw_teng_cols['bety'],
        'alfy_edw_teng': edw_teng_cols['alfy'],
    }
    out.update(rdts)

    return out

def _get_coupling_rdts(r11, r12, r21, r22, betx, bety, alfx, alfy):

    '''
    Developed by CERN OMC team.
    Ported from:
    https://pypi.org/project/optics-functions/
    https://github.com/pylhc/optics_functions

    Based on Calaga, Tomas, https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.8.034001
    '''

    n = len(r11)
    assert len(r12) == n
    assert len(r21) == n
    assert len(r22) == n
    gx, r, inv_gy = np.zeros((n, 2, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))

    # Eq. (16)  C = 1 / (1 + |R|) * -J R J
    # rs form after -J R^T J
    r[:, 0, 0] = r22
    r[:, 0, 1] = -r12
    r[:, 1, 0] = -r21
    r[:, 1, 1] = r11

    r *= 1 / np.sqrt(1 + np.linalg.det(r)[:, None, None])

    # Cbar = Gx * C * Gy^-1,   Eq. (5)
    sqrt_betax = np.sqrt(betx)
    sqrt_betay = np.sqrt(bety)

    gx[:, 0, 0] = 1 / sqrt_betax
    gx[:, 1, 0] = alfx * gx[:, 0, 0]
    gx[:, 1, 1] = sqrt_betax

    inv_gy[:, 1, 1] = 1 / sqrt_betay
    inv_gy[:, 1, 0] = -alfy * inv_gy[:, 1, 1]
    inv_gy[:, 0, 0] = sqrt_betay

    c = np.matmul(gx, np.matmul(r, inv_gy))
    gamma = np.sqrt(1 - np.linalg.det(c))

    # Eq. (9) and Eq. (10)
    denom = 1 / (4 * gamma)
    f1001 = denom * (+c[:, 0, 1] - c[:, 1, 0] + (c[:, 0, 0] + c[:, 1, 1]) * 1j)
    f1010 = denom * (-c[:, 0, 1] - c[:, 1, 0] + (c[:, 0, 0] - c[:, 1, 1]) * 1j)
    f0110 = np.conj(f1001)

    # To be consistent with RDT definition in the Xsuite physics manual
    # (checked against tracking):
    f1001 = -np.conj(f1001)
    f1010 = -np.conj(f1010)
    f0110 = -np.conj(f0110)

    return {'f1001': f1001, 'f1010': f1010, 'f0110': f0110}

def _get_edwards_teng_initial(RR):

    AA = RR[:2, :2]
    BB = RR[:2, 2:4]
    CC = RR[2:4, :2]
    DD = RR[2:4, 2:4]

    if np.linalg.norm(BB) < 1e-10 and np.linalg.norm(CC) < 1e-10:
        RR_ET0 = np.zeros((2, 2))
    else:
        tr = np.linalg.trace
        b_pl_c = CC + _conj_mat(BB)
        det_bc = np.linalg.det(b_pl_c)
        tr_a_m_tr_d = tr(AA) - tr(DD)
        coeff = - (0.5 * tr_a_m_tr_d
            + np.sign(tr_a_m_tr_d) * np.sqrt(det_bc + 0.25 * tr_a_m_tr_d**2))
        RR_ET0 = 1/coeff * b_pl_c

    EE = AA - BB@RR_ET0
    FF = DD + RR_ET0@BB

    quarter = 0.25
    two = 2.0

    sinmu2 = -EE[0,1]*EE[1,0] - quarter*(EE[0,0] - EE[1,1])**2
    sinmux = np.sign(EE[0,1]) * np.sqrt(abs(sinmu2))
    betx0 = EE[0,1] / sinmux
    alfx0 = (EE[0,0] - EE[1,1]) / (two * sinmux)

    sinmu2 = -FF[0,1]*FF[1,0] - quarter*(FF[0,0] - FF[1,1])**2
    sinmuy = np.sign(FF[0,1]) * np.sqrt(abs(sinmu2))
    bety0 = FF[0,1] / sinmuy
    alfy0 = (FF[0,0] - FF[1,1]) / (two * sinmuy)

    edw_teng_init = {
        'RR_ET0': RR_ET0,
        'betx0': betx0,
        'alfx0': alfx0,
        'bety0': bety0,
        'alfy0': alfy0
    }

    return edw_teng_init

def _conj_mat(mm):
    a = mm[0,0]
    b = mm[0,1]
    c = mm[1,0]
    d = mm[1,1]
    return np.array([[d, -b], [-c, a]])


def _edwards_teng_from_one_turn_at_all_locations(WW, qx, qy):

    # R matrix of the full ring (4D)
    Rot = np.zeros(shape=(6, 6), dtype=np.float64)
    Rot[0:2,0:2] = lnf.Rot2D(2 * np.pi * qx)
    Rot[2:4,2:4] = lnf.Rot2D(2 * np.pi * qy)

    n_elem = WW.shape[0]

    betx = np.zeros(n_elem)
    alfx = np.zeros(n_elem)
    bety = np.zeros(n_elem)
    alfy = np.zeros(n_elem)
    r11 = np.zeros(n_elem)
    r12 = np.zeros(n_elem)
    r21 = np.zeros(n_elem)
    r22 = np.zeros(n_elem)

    for ii in range(n_elem):

        WW0 = WW[ii, :, :]
        WW0_inv = lnf.S.T @ WW0.T @ lnf.S
        RR = WW0 @ Rot @ WW0_inv

        # Edwards-Teng initial conditions
        edw_teng_init = _get_edwards_teng_initial(RR)

        RR_ET=edw_teng_init['RR_ET0']

        r11[ii] = RR_ET[0, 0]
        r12[ii] = RR_ET[0, 1]
        r21[ii] = RR_ET[1, 0]
        r22[ii] = RR_ET[1, 1]
        betx[ii] = edw_teng_init['betx0']
        alfx[ii] = edw_teng_init['alfx0']
        bety[ii] = edw_teng_init['bety0']
        alfy[ii] = edw_teng_init['alfy0']

    out_dict = {
        'betx': betx,
        'alfx': alfx,
        'bety': bety,
        'alfy': alfy,
        'r11': r11,
        'r12': r12,
        'r21': r21,
        'r22': r22
    }

    return out_dict

def _propagate_edwards_teng(WW, mux, muy, RR_ET0, betx0, alfx0, bety0, alfy0):

    lnf = xt.linear_normal_form
    SS2D = lnf.S[:2, :2]

    RR_ET = RR_ET0.copy()

    n_elem = len(mux)
    betx = np.zeros(n_elem)
    alfx = np.zeros(n_elem)
    bety = np.zeros(n_elem)
    alfy = np.zeros(n_elem)
    r11 = np.zeros(n_elem)
    r12 = np.zeros(n_elem)
    r21 = np.zeros(n_elem)
    r22 = np.zeros(n_elem)

    betx[0] = betx0
    alfx[0] = alfx0
    bety[0] = bety0
    alfy[0] = alfy0
    r11[0] = RR_ET[0, 0]
    r12[0] = RR_ET[0, 1]
    r21[0] = RR_ET[1, 0]
    r22[0] = RR_ET[1, 1]

    for ii in range(n_elem - 1):

        # Build 2D R matrix of the element
        WW1 = WW[ii, :, :]
        WW2 = WW[ii+1, :, :]
        WW1_inv = lnf.S.T @ WW1.T @ lnf.S
        Rot_e_ii = np.zeros((6,6), dtype=np.float64)
        Rot_e_ii[0:2,0:2] = lnf.Rot2D(2*np.pi*(mux[ii+1] - mux[ii]))
        Rot_e_ii[2:4,2:4] = lnf.Rot2D(2*np.pi*(muy[ii+1] - muy[ii]))
        RRe_ii = WW2 @ Rot_e_ii @ WW1_inv

        # Blocks of the R matrix of the element
        AA = RRe_ii[:2, :2]
        BB = RRe_ii[:2, 2:4]
        CC = RRe_ii[2:4, :2]
        DD = RRe_ii[2:4, 2:4]

        # Propagate EE, FF and RR_ET through the element
        # Bases on MAD-X implementation (see madx/src/twiss.f90, subroutine twcptk)

        if np.allclose(BB, 0, atol=1e-12) and np.allclose(CC, 0, atol=1e-12):
            # Case in which the matrix is block diagonal (no coupling in the element)
            EE = AA
            FF = DD
            EEBAR = SS2D @ EE.T @ SS2D.T
            edet = EE[0,0]*EE[1,1] - EE[0,1]*EE[1,0]
            CCDD = -FF @ RR_ET
            RR_ET = -CCDD @ EEBAR / edet
        else:
            RR_ET_BAR = SS2D @ RR_ET.T @ SS2D.T
            EE = AA - BB @ RR_ET
            edet = EE[0,0]*EE[1,1] - EE[0,1]*EE[1,0]
            EEBAR = SS2D @ EE.T @ SS2D.T
            CCDD = CC - DD @ RR_ET
            FF = DD + CC @ RR_ET_BAR
            RR_ET = -CCDD @ EEBAR / edet

        # Propagate Edwards-Teng Twiss parameters through the element
        # Based on MAD-X implementation (see madx/src/twiss.f90, subroutine twcptk_twiss)

        betx1 = betx[ii]
        alfx1 = alfx[ii]
        bety1 = bety[ii]
        alfy1 = alfy[ii]

        Rx11 = EE[0,0]
        Rx12 = EE[0,1]
        Rx21 = EE[1,0]
        Rx22 = EE[1,1]
        detx = Rx11 * Rx22 - Rx12 * Rx21
        tempb = Rx11 * betx1 - Rx12 * alfx1
        tempa = Rx21 * betx1 - Rx22 * alfx1
        alfx2 = - (tempa * tempb + Rx12 * Rx22) / (detx*betx1)
        betx2 =   (tempb * tempb + Rx12 * Rx12) / (detx*betx1)

        Ry11 = FF[0,0]
        Ry12 = FF[0,1]
        Ry21 = FF[1,0]
        Ry22 = FF[1,1]
        dety = Ry11 * Ry22 - Ry12 * Ry21
        tempb = Ry11 * bety1 - Ry12 * alfy1
        tempa = Ry21 * bety1 - Ry22 * alfy1
        alfy2 = - (tempa * tempb + Ry12 * Ry22) / (dety*bety1)
        bety2 =   (tempb * tempb + Ry12 * Ry12) / (dety*bety1)

        betx[ii+1] = betx2
        alfx[ii+1] = alfx2
        r11[ii+1] = RR_ET[0, 0]
        r12[ii+1] = RR_ET[0, 1]
        r21[ii+1] = RR_ET[1, 0]
        r22[ii+1] = RR_ET[1, 1]
        bety[ii+1] = bety2
        alfy[ii+1] = alfy2

    out_dict = {
        'betx': betx,
        'alfx': alfx,
        'bety': bety,
        'alfy': alfy,
        'r11': r11,
        'r12': r12,
        'r21': r21,
        'r22': r22
    }

    return out_dict


def _get_global_quantities(line, twiss_res, method):

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
            # Proceedings of the 2018 CERN–Accelerator–School
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


def _get_eneloss_and_damping_rates(particle_on_co, R_matrix,
                                       px_co, py_co, ptau_co, W_matrix,
                                       t_rev0, line, radiation_method):
    diff_ptau = np.diff(ptau_co)
    mask_loss = diff_ptau < 0
    eloss_turn = -sum(diff_ptau[mask_loss]) * particle_on_co._xobject.p0c[0]

    # Get eigenvalues
    w0, v0 = np.linalg.eig(R_matrix)

    # Sort eigenvalues
    modes = lnf.sort_modes(v0, w0)
    eigenvals = np.array([w0[ii] for ii in modes])

    # Damping constants and partition numbers
    energy0 = particle_on_co.mass0 * particle_on_co._xobject.gamma0[0]

    damping_constants_turns = -np.log(np.abs(eigenvals))
    damping_constants_s = damping_constants_turns / t_rev0

    # https://cds.cern.ch/record/175614 , Eq. 4.24
    partition_numbers = (
        damping_constants_turns * 2
        / (-np.sum(diff_ptau[mask_loss] / (1 + ptau_co[:-1][mask_loss]))))

    eneloss_damp_res = {
        'eneloss_turn': eloss_turn, # deprecated
        'energy_loss': eloss_turn,
        'damping_constants_turns': damping_constants_turns,
        'damping_constants_s':damping_constants_s,
        'partition_numbers': partition_numbers,
    }

    return eneloss_damp_res

def _extract_sr_distribution_properties(twiss_res):

    radiation_flag = twiss_res['radiation_flag']
    if np.any(
            (radiation_flag == 2)
            | (radiation_flag == 3)):
        raise ValueError('Incompatible radiation flag')

    hx, hy, kappa0_x, kappa0_y = _get_trajectory_curvatures(twiss_res)
    hh = np.sqrt(hx**2 + hy**2)

    ptau_co = twiss_res['ptau']
    dl = twiss_res['length'] * (twiss_res['radiation_flag'] == 1)

    pco = twiss_res['particle_on_co']
    mass0 = pco.mass0
    q0 = pco.q0
    gamma0 = pco._xobject.gamma0[0]
    beta0 = pco._xobject.beta0[0]

    gamma = gamma0 * (1 + beta0 * ptau_co)

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
        'hx': hx, 'hy': hy,
        'h0x': kappa0_x, 'h0y': kappa0_y,
        'E_crit_J': E_crit_J, 'n_dot': n_dot,
        'E_sq_ave_J': E_sq_ave_J, 'E_ave_J': E_ave_J,
        'n_dot_delta_kick_sq_ave': n_dot_delta_kick_sq_ave,
        'dl_radiation': dl,
    }

    return res

def _get_equilibrium_emittance_kick_as_co(twiss_res,
                                  damping_constants_turns,
                                  radiation_method):

    assert radiation_method == 'kick_as_co'

    sr_distrib_properties = _extract_sr_distribution_properties(twiss_res)

    pco = twiss_res['particle_on_co']
    beta0 = pco._xobject.beta0[0]
    gamma0 = pco._xobject.gamma0[0]

    kin_px_co = twiss_res['kin_px']
    kin_py_co = twiss_res['kin_py']
    ptau_co = twiss_res['ptau']
    W_matrix = twiss_res['W_matrix']

    n_dot_delta_kick_sq_ave = sr_distrib_properties['n_dot_delta_kick_sq_ave'][:-1]
    dl = sr_distrib_properties['dl_radiation'][:-1]

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

def _get_equilibrium_emittance_full(twiss_res, R_matrix_ebe,
                                        radiation_method):

    kin_px_co = twiss_res['kin_px']
    kin_py_co = twiss_res['kin_py']
    ptau_co = twiss_res['ptau']

    sr_distrib_properties = _extract_sr_distribution_properties(twiss_res)

    n_dot_delta_kick_sq_ave = sr_distrib_properties['n_dot_delta_kick_sq_ave'][:-1]
    dl = sr_distrib_properties['dl_radiation'][:-1]

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
    WW, _, Rot, lam_eig = lnf.get_linear_normal_form(RR)
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

    CC_split, _, RRR, reig = lnf.get_linear_normal_form(Rot)
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

    beta0 = twiss_res.particle_on_co._xobject.beta0[0]
    gamma0 = twiss_res.particle_on_co._xobject.gamma0[0]

    eq_nemitt_x = float(eq_gemitt_x * (beta0 * gamma0))
    eq_nemitt_y = float(eq_gemitt_y * (beta0 * gamma0))
    eq_nemitt_zeta = float(eq_gemitt_zeta * (beta0 * gamma0))

    Sigma_norm = np.zeros_like(EE_norm, dtype=complex)
    for ii in range(6):
        for jj in range(6):
            Sigma_norm[ii, jj] = EE_norm[ii, jj]/(1 - lam_eig_full[ii, ii]*lam_eig_full[jj, jj])

    Sigma_at_start = (BB @ Sigma_norm @ BB.T).real

    Sigma = RR_ebe @ Sigma_at_start @ np.transpose(RR_ebe, axes=(0,2,1))

    eq_sigma_tab = _build_sigma_table(Sigma=Sigma, s=None, name=twiss_res['name'],)

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
        'hx_rad': sr_distrib_properties['hx'],
        'hy_rad': sr_distrib_properties['hy'],
        'h0x_rad': sr_distrib_properties['h0x'],
        'h0y_rad': sr_distrib_properties['h0y'],
    }

    return res


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

    auxline = xt.Line(elements=tracker.line._element_dict.copy(),
                      element_names=list(tracker.line.element_names).copy())
    if tracker.line.particle_ref is not None:
        auxline.particle_ref = tracker.line.particle_ref.copy()

    insertions = []
    names_inserted_markers = []
    for ii, ss in enumerate(at_s):
        nn = marker_prefix + f'{ii}'
        insertions.append(auxline.env.new(nn, 'Marker', at=ss))
        names_inserted_markers.append(nn)
    auxline.insert(insertions)

    auxtracker = xt.Tracker(
        _buffer=tracker._buffer,
        io_buffer=tracker.io_buffer,
        line=auxline,
        particles_monitor_class=None,
    )
    auxtracker.line.config = tracker.line.config.copy()
    auxtracker.line._extra_config = tracker.line._extra_config.copy()

    return auxtracker, names_inserted_markers


def _complete_twiss_init(start, end, init_at, init,
                        line, reverse,
                        x, px, y, py, zeta, delta,
                        alfx, alfy, betx, bety, bets,
                        dx, dpx, dy, dpy, dzeta,
                        mux, muy, muzeta,
                        ax_chrom, bx_chrom, ay_chrom, by_chrom,
                        ddx, ddpx, ddy, ddpy,
                        spin_x, spin_y, spin_z
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
                spin_x=spin_x, spin_y=spin_y, spin_z=spin_z
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
                assert not(reverse), ('``init`` needs to be given in the '
                    'proper reference frame when ``reverse`` is False')
            elif init is not None and init.reference_frame == 'reverse':
                assert reverse is True, ('``init`` needs to be given in the '
                    'reverse reference frame when ``reverse`` is True')

    return init

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

    alfy1 = -EE[0, :, 2, 3]
    alfx2 = -EE[1, :, 0, 1]

    gamy1 = EE[0, :, 3, 3]
    gamx2 = EE[1, :, 1, 1]

    sign_x = np.sign(betx)
    sign_y = np.sign(bety)
    betx *= sign_x
    alfx *= sign_x
    gamx *= sign_x
    bety *= sign_y
    alfy *= sign_y
    gamy *= sign_y

    return betx, alfx, gamx, bety, alfy, gamy, bety1, betx2, alfy1, alfx2, gamy1, gamx2

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
