# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from warnings import warn

import numpy as np
import xobjects as xo

from ..general import DEPRECATION_INFO_PREP_1_0, _print


DEFAULT_STEPS_R_MATRIX = {
    'dx': 1e-6, 'dpx': 1e-7,
    'dy': 1e-6, 'dpy': 1e-7,
    'dzeta': 1e-5, 'ddelta': 1e-6,
}

DEFAULT_CO_SEARCH_TOL = [1e-11, 1e-11, 1e-11, 1e-11, 1e-5, 1e-9]

DEFAULT_MATRIX_RESPONSIVENESS_TOL = 1e-15
DEFAULT_MATRIX_STABILITY_TOL = 2e-3
DEFAULT_NUM_TURNS_SEARCH_T_REV = 10

VARS_FOR_TWISS_INIT_GENERATION = [
    'x', 'px', 'y', 'py', 'zeta', 'delta',
    'betx', 'alfx', 'bety', 'alfy', 'bets',
    'dx', 'dpx', 'dy', 'dpy', 'dzeta',
    'mux', 'muy', 'muzeta',
    'ax_chrom', 'bx_chrom', 'ay_chrom', 'by_chrom',
    'ddx', 'ddpx', 'ddy', 'ddpy',
]


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


def _normalize_twiss_inputs(twiss_kwargs):

    import xtrack as xt  # Local import avoids circular imports.

    twiss_config, input_kwargs = _normalize_public_twiss_arguments(
        twiss_kwargs)

    # A supplied open init without an explicit range covers the full line.
    if ((twiss_config['init'] is not None
            or twiss_config['betx'] is not None
            or twiss_config['bety'] is not None)
            and twiss_config['start'] is None):
        twiss_config['start'] = xt.START
        twiss_config['end'] = twiss_config['end'] or xt.END

    if twiss_config['num_turns'] != 1:
        assert twiss_config['num_turns'] > 0
        assert twiss_config['start'] is None
        assert twiss_config['end'] is None
        assert twiss_config['init'] is None
        assert twiss_config['reverse'] is False

    twiss_config['start'] = _resolve_twiss_range_endpoint(
        line=twiss_config['line'],
        endpoint=twiss_config['start'],
        reverse=twiss_config['reverse'])
    twiss_config['end'] = _resolve_twiss_range_endpoint(
        line=twiss_config['line'],
        endpoint=twiss_config['end'],
        reverse=twiss_config['reverse'])

    init = twiss_config['init']
    if (init is not None and init not in ['periodic', 'periodic_symmetric']
            or twiss_config['betx'] is not None
            or twiss_config['bety'] is not None):
        twiss_config['periodic'] = False
        twiss_config['periodic_mode'] = None
    else:
        twiss_config['periodic'] = True
        twiss_config['periodic_mode'] = init or 'periodic'
        for coordinate_name in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            assert twiss_config[coordinate_name] is None, (
                f'``{coordinate_name}`` not supported for periodic twiss')

    if twiss_config['method'] is None:
        twiss_config['method'] = '6d'
    assert twiss_config['method'] in ['6d', '4d'], (
        'Method must be ``6d`` or ``4d``')

    _prepare_twiss_at_s_markers(twiss_config)
    _normalize_radiation_method(twiss_config)
    _normalize_line_dependent_twiss_inputs(twiss_config)

    if twiss_config['line'].enable_time_dependent_vars:
        raise RuntimeError('Time dependent variables not supported in Twiss')

    track_flag_updates, line_config_updates = (
        _get_twiss_line_context_updates(twiss_config))
    return (
        twiss_config, input_kwargs, track_flag_updates, line_config_updates)


def _normalize_public_twiss_arguments(twiss_kwargs):

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

    return twiss_kwargs, input_kwargs


def _normalize_line_dependent_twiss_inputs(twiss_config):
    """Normalize inputs that depend on the selected line and range."""

    # Resolve symbolic init locations and TwissTable inputs after range names.
    import xtrack as xt  # Local import avoids circular imports.
    from .twiss_table import TwissTable

    init_at = twiss_config['init_at']
    if isinstance(init_at, xt.match._LOC):
        if init_at.name == 'START':
            twiss_config['init_at'] = twiss_config['start']
        elif init_at.name == 'END':
            twiss_config['init_at'] = twiss_config['end']

    if isinstance(twiss_config['init'], TwissTable):
        if twiss_config['init_at'] is None:
            twiss_config['init_at'] = twiss_config['start']
        twiss_config['init'] = twiss_config['init'].get_twiss_init(
            at_element=twiss_config['init_at'])
        twiss_config['init_at'] = None

    if twiss_config['matrix_responsiveness_tol'] is None:
        twiss_config['matrix_responsiveness_tol'] = (
            twiss_config['line'].matrix_responsiveness_tol)
    if twiss_config['matrix_stability_tol'] is None:
        twiss_config['matrix_stability_tol'] = twiss_config['line'].matrix_stability_tol
    if (twiss_config['line']._radiation_model is not None
            and twiss_config['radiation_method'] != 'kick_as_co'):
        twiss_config['matrix_stability_tol'] = None
        if twiss_config['use_full_inverse'] is None:
            twiss_config['use_full_inverse'] = True

    if twiss_config['particle_ref'] is None:
        if twiss_config['particle_on_co'] is not None:
            twiss_config['particle_ref'] = twiss_config['particle_on_co'].copy()
        elif twiss_config['co_guess'] is None and hasattr(twiss_config['line'], 'particle_ref'):
            twiss_config['particle_ref'] = twiss_config['line'].particle_ref

    if twiss_config['line'].iscollective and not twiss_config['include_collective']:
        _print(
            'The line has collective elements.\n'
            'In the twiss computation collective elements are'
            ' replaced by drifts')
        twiss_config['line'] = twiss_config['line']._get_non_collective_line()

    if twiss_config['particle_ref'] is None and twiss_config['co_guess'] is None:
        raise ValueError(
            "Either ``particle_ref`` or ``co_guess`` must be provided")

    if isinstance(twiss_config['init'], str):
        if twiss_config['init'] in ['preserve', 'preserve_start', 'preserve_end']:
            raise ValueError(f"init={twiss_config['init']} not anymore supported")
        assert twiss_config['init'] in ('periodic', 'full_periodic')
    if (not twiss_config['periodic']
            and (twiss_config['delta0'] is not None or twiss_config['zeta0'] is not None)):
        raise ValueError(
            'delta0 and zeta0 cannot be provided for open twiss')


def _resolve_twiss_range_endpoint(line, endpoint, reverse):

    import xtrack as xt  # Local import avoids circular imports.

    if endpoint is None:
        return None

    if isinstance(endpoint, xt.match._LOC):
        assert endpoint in [xt.START, xt.END]
        if reverse:
            endpoint = {xt.START: xt.END, xt.END: xt.START}[endpoint]
        endpoint = {
            xt.START: line._element_names_unique[0],
            xt.END: line._element_names_unique[-1],
        }[endpoint]

    assert isinstance(endpoint, str)  # index not supported anymore
    return endpoint


def _element_ref_to_index(line, element_ref, allow_end_point=True):
    if allow_end_point and element_ref == '_end_point':
        return len(line._element_names_unique)
    if isinstance(element_ref, str):
        if element_ref not in line._element_names_unique:
            raise ValueError(f'Element {element_ref} not found in line')
        return line._element_names_unique.index(element_ref)
    return element_ref


def _prepare_twiss_at_s_markers(twiss_config):

    if twiss_config['at_s'] is None:
        return

    if twiss_config['reverse']:
        raise NotImplementedError('``at_s`` not implemented for ``reverse``=True')
    if np.isscalar(twiss_config['at_s']):
        twiss_config['at_s'] = [twiss_config['at_s']]
    assert twiss_config['at_elements'] is None

    auxtracker, names_inserted_markers = (
        _build_auxiliary_tracker_with_extra_markers(
            tracker=twiss_config['line'].tracker,
            at_s=twiss_config['at_s'],
            marker_prefix='inserted_twiss_marker',
            algorithm='insert'))
    twiss_config['line'] = auxtracker.line
    twiss_config['at_elements'] = names_inserted_markers
    twiss_config['at_s'] = None
    twiss_config['strengths'] = True


def _build_auxiliary_tracker_with_extra_markers(
        tracker, at_s, marker_prefix, algorithm='auto'):

    import xtrack as xt  # Local import avoids circular imports.

    assert algorithm in ['auto', 'insert', 'regen_all_drift']
    if algorithm == 'auto':
        if len(at_s) < 10:
            algorithm = 'insert'
        else:
            algorithm = 'regen_all_drifts'

    auxline = xt.Line(
        elements=tracker.line._element_dict.copy(),
        element_names=list(tracker.line.element_names).copy())
    if tracker.line.particle_ref is not None:
        auxline.particle_ref = tracker.line.particle_ref.copy()

    insertions = []
    names_inserted_markers = []
    for ii, ss in enumerate(at_s):
        name = marker_prefix + f'{ii}'
        insertions.append(auxline.env.new(name, 'Marker', at=ss))
        names_inserted_markers.append(name)
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


def _normalize_radiation_method(twiss_config):

    line = twiss_config['line']
    radiation_method = twiss_config['radiation_method']

    if radiation_method is None and line._radiation_model is not None:
        if line._radiation_model in ('quantum', 'quantum-kick'):
            raise ValueError(
                'twiss cannot be called when the radiation model is stochastic')
        if twiss_config['method'] == '4d':
            raise RuntimeError(
                '4d twiss cannot be called when radiation is present')
        radiation_method = 'kick_as_co'
        twiss_config['radiation_method'] = radiation_method

    if radiation_method is not None and radiation_method != 'full':
        assert isinstance(line._context, xo.ContextCpu), (
            'Twiss with radiation computation is only supported on CPU')
        assert not line._context.openmp_enabled, (
            'Twiss with radiation computation is not supported with OpenMP '
            'parallelization')
        assert radiation_method in ['kick_as_co', 'scale_as_co']


def _get_twiss_line_context_updates(twiss_config):

    line = twiss_config['line']
    track_flag_updates = {}
    line_config_updates = {}

    if twiss_config['disable_apertures']:
        track_flag_updates.update(
            XS_FLAG_IGNORE_GLOBAL_APERTURE=True,
            XS_FLAG_IGNORE_LOCAL_APERTURE=True,
        )

    if twiss_config['method'] == '4d':
        track_flag_updates['XS_FLAG_KILL_CAVITY_KICK'] = True

    if twiss_config['radiation_method'] == 'kick_as_co':
        track_flag_updates['XS_FLAG_SR_KICK_SAME_AS_FIRST'] = True
    elif twiss_config['radiation_method'] == 'scale_as_co':
        line_config_updates['XTRACK_SYNRAD_SCALE_SAME_AS_FIRST'] = True

    # Avoid entering preservation contexts when the line already has the
    # requested state.
    track_flag_updates = {
        name: value for name, value in track_flag_updates.items()
        if getattr(line.tracker.track_flags, name) != value
    }
    line_config_updates = {
        name: value for name, value in line_config_updates.items()
        if getattr(line.config, name, None) != value
    }
    return track_flag_updates, line_config_updates


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
