# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .constants import VARS_FOR_TWISS_INIT_GENERATION
from .periodic_init import _compute_periodic_twiss_init_and_data
from .twiss_init import _complete_twiss_init


_PERIODIC_INIT_ARGUMENTS_FROM_BASE_DATA = (
    'line',
    'particle_on_co',
    'particle_ref',
    'method',
    'co_search_settings',
    'continue_on_closed_orbit_error',
    'delta0',
    'zeta0',
    'zeta_shift',
    'steps_R_matrix',
    'W_matrix',
    'R_matrix',
    'co_guess',
    'delta_disp',
    'symplectify',
    'matrix_responsiveness_tol',
    'matrix_stability_tol',
    'num_turns',
    'co_search_at',
    'search_for_t_rev',
    'spin',
    'num_turns_search_t_rev',
    'nemitt_x',
    'nemitt_y',
    'step_W_sigma',
    'compute_R_element_by_element',
    'only_markers',
    'only_orbit',
    'periodic_mode',
    'include_collective',
)


def _acquire_base_twiss_init(data):

    acquisition_plan = data['twiss_computation_plan'].init_acquisition

    if acquisition_plan.source == 'open_input':
        assert not data['periodic']
        return {'skip_global_quantities': True}

    if acquisition_plan.source != 'periodic_solution':
        raise RuntimeError(
            f'Unexpected Twiss init source: {acquisition_plan.source}')

    assert data['periodic']
    periodic_init_data = _acquire_periodic_base_twiss_init(
        data=data, acquisition_plan=acquisition_plan)

    return {
        'periodic_init_data': periodic_init_data,
        'init': periodic_init_data.init,
        'R_matrix': periodic_init_data.R_matrix,
        'steps_R_matrix': periodic_init_data.steps_R_matrix,
        'eigenvalues': periodic_init_data.eigenvalues,
        'Rot': periodic_init_data.Rot,
        'RR_ebe': periodic_init_data.RR_ebe,
    }


def _acquire_periodic_base_twiss_init(data, acquisition_plan):

    assert acquisition_plan.computation_direction == 'forward'
    if acquisition_plan.scope == 'full_line':
        assert data['start'] is None and data['end'] is None
        periodic_start = periodic_end = None
    elif acquisition_plan.scope == 'requested_range':
        assert data['start'] is not None and data['end'] is not None
        periodic_start, periodic_end = data['start'], data['end']
    else:
        raise RuntimeError(
            f'Unexpected periodic Twiss scope: {acquisition_plan.scope}')
    periodic_init_kwargs = {
        name: data[name]
        for name in _PERIODIC_INIT_ARGUMENTS_FROM_BASE_DATA
    }
    periodic_init_kwargs.update(
        start=periodic_start,
        end=periodic_end,
        initial_particles=data['_initial_particles'],
    )

    return _compute_periodic_twiss_init_and_data(**periodic_init_kwargs)


def _clear_twiss_init_inputs(data):

    data['init_at'] = None
    for field_name in (
            *VARS_FOR_TWISS_INIT_GENERATION,
            'spin_x', 'spin_y', 'spin_z'):
        data[field_name] = None


def _complete_init_for_base_twiss(data):

    init = _complete_twiss_init(
        start=data['start'], end=data['end'], init_at=data['init_at'],
        init=data['init'], line=data['line'], reverse=data['reverse'],
        x=data['x'], px=data['px'], y=data['y'], py=data['py'],
        zeta=data['zeta'], delta=data['delta'],
        alfx=data['alfx'], alfy=data['alfy'],
        betx=data['betx'], bety=data['bety'], bets=data['bets'],
        dx=data['dx'], dpx=data['dpx'], dy=data['dy'], dpy=data['dpy'],
        dzeta=data['dzeta'], mux=data['mux'], muy=data['muy'],
        muzeta=data['muzeta'],
        ax_chrom=data['ax_chrom'], bx_chrom=data['bx_chrom'],
        ay_chrom=data['ay_chrom'], by_chrom=data['by_chrom'],
        ddx=data['ddx'], ddpx=data['ddpx'],
        ddy=data['ddy'], ddpy=data['ddpy'],
        spin_x=data['spin_x'], spin_y=data['spin_y'], spin_z=data['spin_z'],
    )
    completed_init = (init.copy() if hasattr(init, 'copy') else init)

    return init, completed_init
