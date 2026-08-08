# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from warnings import warn

from ..general import DEPRECATION_INFO_PREP_1_0, _print


def _normalize_twiss_inputs(twiss_kwargs, twiss_init_cls):

    twiss_kwargs = twiss_kwargs.copy()

    (twiss_kwargs['chrom'],
     twiss_kwargs['step_W_sigma'],
     twiss_kwargs['polarization_analysis'],
     twiss_kwargs['radiation_analysis'],
     twiss_kwargs['steps_R_matrix']) = _handle_deprecated_twiss_kwargs(
        at_s=twiss_kwargs['at_s'],
        at_elements=twiss_kwargs['at_elements'],
        compute_chromatic_properties=twiss_kwargs['compute_chromatic_properties'],
        r_sigma=twiss_kwargs['r_sigma'],
        freeze_energy=twiss_kwargs['freeze_energy'],
        freeze_longitudinal=twiss_kwargs['freeze_longitudinal'],
        polarization=twiss_kwargs['polarization'],
        eneloss_and_damping=twiss_kwargs['eneloss_and_damping'],
        steps_r_matrix=twiss_kwargs['steps_r_matrix'],
        chrom=twiss_kwargs['chrom'],
        step_W_sigma=twiss_kwargs['step_W_sigma'],
        polarization_analysis=twiss_kwargs['polarization_analysis'],
        radiation_analysis=twiss_kwargs['radiation_analysis'],
        steps_R_matrix=twiss_kwargs['steps_R_matrix'],
    )

    input_kwargs = twiss_kwargs.copy()

    (twiss_kwargs['step_W_sigma'],
     twiss_kwargs['nemitt_x'],
     twiss_kwargs['nemitt_y'],
     twiss_kwargs['delta_disp'],
     twiss_kwargs['delta_chrom'],
     twiss_kwargs['zeta_disp'],
     twiss_kwargs['zeta_shift'],
     twiss_kwargs['values_at_element_exit'],
     twiss_kwargs['continue_on_closed_orbit_error'],
     twiss_kwargs['freeze_longitudinal'],
     twiss_kwargs['radiation_method'],
     twiss_kwargs['spin'],
     twiss_kwargs['polarization_analysis'],
     twiss_kwargs['radiation_integrals'],
     twiss_kwargs['radiation_analysis'],
     twiss_kwargs['symplectify'],
     twiss_kwargs['reverse'],
     twiss_kwargs['strengths'],
     twiss_kwargs['hide_thin_groups'],
     twiss_kwargs['search_for_t_rev'],
     twiss_kwargs['num_turns_search_t_rev'],
     twiss_kwargs['only_twiss_init'],
     twiss_kwargs['only_markers'],
     twiss_kwargs['only_orbit'],
     twiss_kwargs['compute_R_element_by_element'],
     twiss_kwargs['compute_lattice_functions'],
     twiss_kwargs['chrom'],
     twiss_kwargs['num_turns'],
     twiss_kwargs['disable_apertures']) = _apply_twiss_defaults(
        step_W_sigma=twiss_kwargs['step_W_sigma'],
        nemitt_x=twiss_kwargs['nemitt_x'],
        nemitt_y=twiss_kwargs['nemitt_y'],
        delta_disp=twiss_kwargs['delta_disp'],
        delta_chrom=twiss_kwargs['delta_chrom'],
        zeta_disp=twiss_kwargs['zeta_disp'],
        zeta_shift=twiss_kwargs['zeta_shift'],
        values_at_element_exit=twiss_kwargs['values_at_element_exit'],
        continue_on_closed_orbit_error=twiss_kwargs['continue_on_closed_orbit_error'],
        freeze_longitudinal=twiss_kwargs['freeze_longitudinal'],
        radiation_method=twiss_kwargs['radiation_method'],
        spin=twiss_kwargs['spin'],
        polarization_analysis=twiss_kwargs['polarization_analysis'],
        radiation_integrals=twiss_kwargs['radiation_integrals'],
        radiation_analysis=twiss_kwargs['radiation_analysis'],
        symplectify=twiss_kwargs['symplectify'],
        reverse=twiss_kwargs['reverse'],
        strengths=twiss_kwargs['strengths'],
        hide_thin_groups=twiss_kwargs['hide_thin_groups'],
        search_for_t_rev=twiss_kwargs['search_for_t_rev'],
        num_turns_search_t_rev=twiss_kwargs['num_turns_search_t_rev'],
        only_twiss_init=twiss_kwargs['only_twiss_init'],
        only_markers=twiss_kwargs['only_markers'],
        only_orbit=twiss_kwargs['only_orbit'],
        compute_R_element_by_element=twiss_kwargs['compute_R_element_by_element'],
        compute_lattice_functions=twiss_kwargs['compute_lattice_functions'],
        chrom=twiss_kwargs['chrom'],
        num_turns=twiss_kwargs['num_turns'],
        disable_apertures=twiss_kwargs['disable_apertures'],
    )

    if twiss_kwargs['only_markers']:
        raise NotImplementedError('``only_markers`` not supported anymore')

    if twiss_kwargs['polarization_analysis']:
        twiss_kwargs['spin'] = True
        # Some quantities are needed for polarization. This could be decoupled
        # in the future.
        twiss_kwargs['radiation_integrals'] = True
    if twiss_kwargs['spin']:
        assert twiss_kwargs['reverse'] is False

    if isinstance(twiss_kwargs['init'], twiss_init_cls):
        twiss_kwargs['init'] = twiss_kwargs['init'].copy()

    return twiss_kwargs, input_kwargs


def _normalize_twiss_inputs_after_line_context(data):
    """Finish normalization that depends on the prepared line and range."""

    data = data.copy()

    # Resolve symbolic init locations and TwissTable inputs after range names.
    import xtrack as xt  # Local import avoids circular imports.
    from .twiss_table import TwissTable

    init_at = data['init_at']
    if isinstance(init_at, xt.match._LOC):
        if init_at.name == 'START':
            data['init_at'] = data['start']
        elif init_at.name == 'END':
            data['init_at'] = data['end']

    if isinstance(data['init'], TwissTable):
        if data['init_at'] is None:
            data['init_at'] = data['start']
        data['init'] = data['init'].get_twiss_init(
            at_element=data['init_at'])
        data['init_at'] = None

    if data['matrix_responsiveness_tol'] is None:
        data['matrix_responsiveness_tol'] = (
            data['line'].matrix_responsiveness_tol)
    if data['matrix_stability_tol'] is None:
        data['matrix_stability_tol'] = data['line'].matrix_stability_tol
    if (data['line']._radiation_model is not None
            and data['radiation_method'] != 'kick_as_co'):
        data['matrix_stability_tol'] = None
        if data['use_full_inverse'] is None:
            data['use_full_inverse'] = True

    if data['particle_ref'] is None:
        if data['particle_on_co'] is not None:
            data['particle_ref'] = data['particle_on_co'].copy()
        elif data['co_guess'] is None and hasattr(data['line'], 'particle_ref'):
            data['particle_ref'] = data['line'].particle_ref

    if data['line'].iscollective and not data['include_collective']:
        _print(
            'The line has collective elements.\n'
            'In the twiss computation collective elements are'
            ' replaced by drifts')
        data['line'] = data['line']._get_non_collective_line()

    if data['particle_ref'] is None and data['co_guess'] is None:
        raise ValueError(
            "Either ``particle_ref`` or ``co_guess`` must be provided")

    if data['method'] is None:
        data['method'] = '6d'
    assert data['method'] in ['6d', '4d'], (
        'Method must be ``6d`` or ``4d``')

    if isinstance(data['init'], str):
        if data['init'] in ['preserve', 'preserve_start', 'preserve_end']:
            raise ValueError(f"init={data['init']} not anymore supported")
        assert data['init'] in ('periodic', 'full_periodic')
    if (not data['periodic']
            and (data['delta0'] is not None or data['zeta0'] is not None)):
        raise ValueError(
            'delta0 and zeta0 cannot be provided for open twiss')

    return data


def _handle_deprecated_twiss_kwargs(
        *,
        at_s,
        at_elements,
        compute_chromatic_properties,
        r_sigma,
        freeze_energy,
        freeze_longitudinal,
        polarization,
        eneloss_and_damping,
        steps_r_matrix,
        chrom,
        step_W_sigma,
        polarization_analysis,
        radiation_analysis,
        steps_R_matrix,
):
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

    return (
        chrom,
        step_W_sigma,
        polarization_analysis,
        radiation_analysis,
        steps_R_matrix,
    )


def _apply_twiss_defaults(
        *,
        step_W_sigma,
        nemitt_x,
        nemitt_y,
        delta_disp,
        delta_chrom,
        zeta_disp,
        zeta_shift,
        values_at_element_exit,
        continue_on_closed_orbit_error,
        freeze_longitudinal,
        radiation_method,
        spin,
        polarization_analysis,
        radiation_integrals,
        radiation_analysis,
        symplectify,
        reverse,
        strengths,
        hide_thin_groups,
        search_for_t_rev,
        num_turns_search_t_rev,
        only_twiss_init,
        only_markers,
        only_orbit,
        compute_R_element_by_element,
        compute_lattice_functions,
        chrom,
        num_turns,
        disable_apertures,
):
    return (
        (step_W_sigma or 0.01),
        (nemitt_x or 1e-6),
        (nemitt_y or 1e-6),
        (delta_disp or 1e-5),
        (delta_chrom or 5e-5),
        (zeta_disp or 1e-3),
        (zeta_shift or 0.0),
        (values_at_element_exit or False),
        (continue_on_closed_orbit_error or False),
        (freeze_longitudinal or False),
        (radiation_method or None),
        (spin or False),
        (polarization_analysis or False),
        (radiation_integrals or False),
        (radiation_analysis or False),
        (symplectify or False),
        (reverse or False),
        (strengths or False),
        (hide_thin_groups or False),
        (search_for_t_rev or False),
        (num_turns_search_t_rev or None),
        (only_twiss_init or False),
        (only_markers or False),
        (only_orbit or False),
        (compute_R_element_by_element or False),
        (compute_lattice_functions
            if compute_lattice_functions is not None else True),
        (chrom if chrom is not None else None),
        (num_turns or 1),
        (disable_apertures if disable_apertures is not None else True),
    )
