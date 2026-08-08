# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from dataclasses import dataclass

from .periodic_solution import _find_periodic_solution
from .transfer_matrices import _complete_steps_r_matrix_with_default


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
