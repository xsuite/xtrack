# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #


def _select_twiss_backend(twiss_config):
    """The object that computes the twiss derivatives, chosen once per call."""
    if twiss_config['tpsa']:
        from ..tpsa.twiss import TpsaTwiss
        return TpsaTwiss.from_twiss_config(twiss_config)
    return FiniteDifferenceTwiss()


class FiniteDifferenceTwiss:
    """Closed orbit, R, W and chromatic functions from tracked particles."""

    def find_closed_orbit(self, line, method, **kwargs):
        return line.find_closed_orbit(**kwargs)

    def get_R_matrix(self, line, **kwargs):
        from .periodic_solution import _get_R_matrix_adapting_steps
        return _get_R_matrix_adapting_steps(line, **kwargs)

    def track_orbit_and_W(self, line, **kwargs):
        from .optics_propagation import _track_orbit_and_W_with_particles
        return _track_orbit_and_W_with_particles(line, **kwargs)

    def chromatic_functions(self, twiss_config, twiss_res):
        from .chromatic_functions import _get_chromatic_functions
        return _get_chromatic_functions(
            twiss_config, on_momentum_twiss_res=twiss_res)
