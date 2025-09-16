# copyright ############################### #
# This file is part of the Xtrack package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np


XTRACK_ELEMENTS_INIT_DEFAULTS = {
    'Bend': {
        'length': 1.,
    },
    'Quadrupole': {
        'length': 1.,
    },
    'Solenoid': {
        'length': 1.,
    },
    'LimitPolygon': {
        'x_vertices': np.array([0, 1, 1, 0]),
        'y_vertices': np.array([0, 0, 1, 1]),
    },
    'BeamProfileMonitor': {
        'range': 1,
    },
    'LastTurnsMonitor': {
        'n_last_turns': 1,
        'num_particles': 1,
    },
    'ParticlesMonitor': {
        'num_particles': 1,
        'start_at_turn': 0,
        'stop_at_turn': 1,
    },
    'Exciter': {
        'samples': [0],
    }
}
