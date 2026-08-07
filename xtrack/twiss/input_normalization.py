# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from warnings import warn

from ..general import DEPRECATION_INFO_PREP_1_0


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
