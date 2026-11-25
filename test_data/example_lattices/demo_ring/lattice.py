# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

from scipy.constants import c
import xtrack as xt

################################################################################
# Load geometry and the nominal operating strengths
################################################################################
env         = xt.get_environment()
model_path  = Path(__file__).resolve().parent
env.call(model_path / 'geometry.py')
env.call(model_path / 'strengths.py')

################################################################################
# Assemble the ring from its named arcs and straight sections
################################################################################
line = env.new_line(
    name        = 'demo',
    components  = [
        'arc.s1', 'insertion',
        'arc.s2', 'utility.s2',
        'arc.s3', 'utility.s3',
        'arc.s4', 'utility.s4',
        'arc.s5', 'utility.s5',
        'arc.s6', 'utility.s6',
    ],
)
line.set_particle_ref('electron', p0c = env['p0c'])
env['frequency_rf'] = (
    env['harmonic_rf'] * line.particle_ref.beta0[0] * c / line.get_length())

################################################################################
# Select the named observation and steering devices
################################################################################
table = line.get_table()
line.steering_monitors_x = list(table.rows['bpm[.].*'].name)
line.steering_monitors_y = list(table.rows['bpm[.].*'].name)
line.steering_correctors_x = list(table.rows['cor[.].*'].name)
line.steering_correctors_y = list(table.rows['cor[.].*'].name)
