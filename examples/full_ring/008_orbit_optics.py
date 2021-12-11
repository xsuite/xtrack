import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

from make_short_line import make_short_line

short_test = False # Short line (5 elements)

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath(
                        './hllhc_14/line_and_particle.json')

####################
# Choose a context #
####################

context = xo.ContextCpu()


with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
if short_test:
    line = make_short_line(line)

print('Build tracker...')
freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
tracker = xt.Tracker(_context=context,
            line=line,
            local_particle_src=xp.gen_local_particle_api(
                                                freeze_vars=freeze_vars),
            )

part0 = xp.Particles(_context=context, **input_data['particle'])

part_on_co = tracker.find_closed_orbit(part0)

RR =  tracker.compute_one_turn_matrix_finite_differences(part_on_co)
