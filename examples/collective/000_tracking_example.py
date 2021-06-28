import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xtrack as xt
import xfields as xf


fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_with_spacecharge_and_particle.json')
turn_by_turn_monitor = True

num_turns = int(100)
n_part = 200

####################
# Choose a context #
####################

context = xo.ContextCpu()
context = xo.ContextCupy()
context = xo.ContextPyopencl('0.0')

##################
# Get a sequence #
##################

with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xl.Line.from_dict(input_data['line'])

# Replace all spacecharge with xobjects
newseq = sequence.copy()
_buffer = context.new_buffer()
spch_elements = []
for ii, ee in enumerate(newseq.elements):
    if ee.__class__.__name__ == 'SCQGaussProfile':
        newee = xf.SpaceChargeBiGaussian.from_xline(ee, _buffer=_buffer)
        #newee.update_sigma_x_on_track = True # Commented out for test
        newee.iscollective=True
        newseq.elements[ii] = newee
        spch_elements.append(newee)


#################
# Build Tracker #
#################
print('Build tracker...')
tracker= xt.Tracker(_buffer=_buffer,
             sequence=newseq,
             particles_class=xt.Particles,
             local_particle_src=None,
             save_source_as='source.c')

assert tracker.iscollective
assert tracker.track == tracker._track_with_collective

######################
# Get some particles #
######################
particles = xt.Particles(_context=context, **input_data['particle'])

#########
# Track #
#########

print('Track a few turns...')
n_turns = 10
tracker.track(particles, num_turns=n_turns,
              turn_by_turn_monitor=True)

assert tracker.record_last_track.x.shape == (1, 10)

#######################
# Check against xline #
#######################
print('Check against xline ...')
ip_check = 0
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
pyst_part = xl.Particles.from_dict(input_data['particle'])
for _ in range(n_turns):
    sequence.track(pyst_part)

for vv in vars_to_check:
    pyst_value = getattr(pyst_part, vv)
    xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
    passed = np.isclose(xt_value, pyst_value, rtol=2e-8, atol=7e-9)
    print(f'Check var {vv}:\n'
          f'    pyst:   {pyst_value: .7e}\n'
          f'    xtrack: {xt_value: .7e}\n')
    if not passed:
        raise ValueError

