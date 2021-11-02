import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xline as xl
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
context = xo.ContextCupy()
context = xo.ContextPyopencl('0.0')

#############
# Load file #
#############

if str(fname_line_particles).endswith('.pkl'):
    with open(fname_line_particles, 'rb') as fid:
        input_data = pickle.load(fid)
elif str(fname_line_particles).endswith('.json'):
    with open(fname_line_particles, 'r') as fid:
        input_data = json.load(fid)

##################
# Get a sequence #
##################

sequence = xl.Line.from_dict(input_data['line'])
if short_test:
    sequence = make_short_line(sequence)

#################
# Build Tracker #
#################
print('Build tracker...')
freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
tracker = xt.Tracker(_context=context,
            sequence=sequence,
            local_particle_src=xp.gen_local_particle_api(
                                                freeze_vars=freeze_vars),
            )

######################
# Get some particles #
######################
input_data['particle']['x'] += np.linspace(-1e-4, 1e-4, 10)
particles = xp.Particles(_context=context, **input_data['particle'])

particles_before_tracking = particles.copy()

#########
# Track #
#########
print('Track a few turns...')
n_turns = 10
tracker.track(particles, num_turns=n_turns)

for vv in ['psigma', 'delta', 'rpp', 'rvv', 'zeta']:
    vv_before = context.nparray_from_context_array(
                        getattr(particles_before_tracking, vv))
    vv_after= context.nparray_from_context_array(
                        getattr(particles, vv))
    assert np.all(vv_before == vv_after)

print('Check passed')
