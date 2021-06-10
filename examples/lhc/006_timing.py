import pathlib
import json
import pickle
import numpy as np

import xtrack as xt
import xobjects as xo
import pysixtrack

from make_short_line import make_short_line
import time


test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath('lhc_no_bb/line_and_particle.json')
fname_line_particles = test_data_folder.joinpath(
                                './lhc_with_bb/line_and_particle.json')
# # Quick test (for debugging)
# short_test = True# Short line (5 elements)
# n_part = 20
# num_turns = int(100)

short_test = False
num_turns = int(100)

####################
# Choose a context #
####################

n_part = 200
context = xo.ContextCpu()

#n_part = 20000
#context = xo.ContextCupy()

#n_part = 20000
#context = xo.ContextPyopencl('0.0')

#############
# Load file #
#############

if str(fname_line_particles).endswith('.pkl'):
    with open(fname_line_particles, 'rb') as fid:
        input_data = pickle.load(fid)
elif str(fname_line_particles).endswith('.json'):
    with open(fname_line_particles, 'r') as fid:
        input_data = json.load(fid)

# # Force remove bb
# line_dict = input_data['line']
# for ii, ee in enumerate(line_dict['elements']):
#     if ee['__class__'] == 'BeamBeam6D' or ee['__class__'] == 'BeamBeam4D':
#         line_dict['elements'][ii] = {}
#         ee = line_dict['elements'][ii]
#         ee['__class__'] = 'Drift'
#         ee['length'] = 0.

##################
# Get a sequence #
##################

print('Import sequence')
sequence = pysixtrack.Line.from_dict(input_data['line'])
if short_test:
    sequence = make_short_line(sequence)

##################
# Build TrackJob #
##################

print('Build tracker')
tracker = xt.Tracker(context=context,
            sequence=sequence,
            particles_class=xt.Particles,
            local_particle_src=None,
            save_source_as='source.c')

######################
# Get some particles #
######################

print('Import particles')
part_pyst = pysixtrack.Particles.from_dict(input_data['particle'])

dx_array = np.linspace(-1e-4, 1e-4, n_part)
dy_array = np.linspace(-2e-4, 2e-4, n_part)
pysixtrack_particles = []
for ii in range(n_part):
    pp = part_pyst.copy()
    pp.x += dx_array[ii]
    pp.y += dy_array[ii]
    pysixtrack_particles.append(pp)

particles = xt.Particles(pysixtrack_particles=pysixtrack_particles,
                         _context=context)
#########
# Track #
#########

print('Track!')
print(f'context: {tracker.context}')
t1 = time.time()
tracker.track(particles, num_turns=num_turns)
context.synchronize()
t2 = time.time()
print(f'Time {(t2-t1)*1000:.2f} ms')
print(f'Time {(t2-t1)*1e6/num_turns/n_part:.2f} us/part/turn')


############################
# Check against pysixtrack #
############################

ip_check = n_part//3*2

print(f'\nTest against pysixtrack over {num_turns} turns on particle {ip_check}:')
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
pyst_part = pysixtrack_particles[ip_check].copy()
for iturn in range(num_turns):
    print(f'turn {iturn}/{num_turns}', end='\r', flush=True)
    sequence.track(pyst_part)

for vv in vars_to_check:
    pyst_value = getattr(pyst_part, vv)
    xt_value = context.nparray_from_context_array(
                        getattr(particles, vv)[ip_check])
    passed = np.isclose(xt_value, pyst_value, rtol=1e-9, atol=3e-11)
    if not passed:
        print(f'Not passed on var {vv}!\n'
              f'    pyst:   {pyst_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        raise ValueError
    else:
        print(f'Passed on var {vv}!\n'
              f'    pyst:   {pyst_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
