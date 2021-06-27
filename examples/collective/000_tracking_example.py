import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xtrack as xt
import xfields as xf


fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_with_spacecharge_and_particle.json')


num_turns = int(100)
n_part = 200

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

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
        newee.iscollective=True
        newseq.elements[ii] = newee
        spch_elements.append(newee)

# # Test
# for ss in spch_elements:
#     ss.update_mean_x_on_track = True
#     ss.update_sigma_x_on_track = True

# Split the sequence
parts = []
this_part = xl.Line(elements=[], element_names=[])
for nn, ee in zip(newseq.element_names, newseq.elements):
    if not hasattr(ee, 'iscollective') or not ee.iscollective:
        this_part.append_element(ee, nn)
    else:
        if len(this_part.elements)>0:
            this_part.iscollective=False
            parts.append(this_part)
        parts.append(ee)
        this_part = xl.Line(elements=[], element_names=[])
if len(this_part.elements)>0:
    this_part.iscollective=False
    parts.append(this_part)

# Transform non collective elements into xtrack elements 
noncollective_xelements = []
for ii, pp in enumerate(parts):
    if hasattr(pp, 'iscollective') and not pp.iscollective:
        tempxtline = xt.Line(_buffer=_buffer,
                           sequence=pp)
        pp.elements = tempxtline.elements
        noncollective_xelements += pp.elements

# Build tracker for non collective elements
supertracker = xt.Tracker(_buffer=_buffer,
        sequence=xl.Line(elements=noncollective_xelements,
            element_names=[
                f'e{ii}' for ii in range(len(noncollective_xelements))]))

# Build trackers for non collective parts
for ii, pp in enumerate(parts):
    if hasattr(pp, 'iscollective') and not pp.iscollective:
        parts[ii] = xt.Tracker(_buffer=_buffer,
                            sequence=pp,
                            element_classes=supertracker.element_classes,
                            track_kernel=supertracker.track_kernel)


# #################
# # Build Tracker #
# #################
# print('Build tracker...')
# tracker= xt.Tracker(_buffer=_buffer,
#             sequence=newseq,
#             particles_class=xt.Particles,
#             local_particle_src=None,
#             save_source_as='source.c')

######################
# Get some particles #
######################
particles = xt.Particles(_context=context, **input_data['particle'])

#########
# Track #
#########
print('Track a few turns...')
n_turns = 10
#tracker.track(particles, num_turns=n_turns)
for tt in range(n_turns):
    for pp in parts:
        pp.track(particles)

############################
# Check against pysixtrack #
############################
print('Check against pysixtrack...')
import pysixtrack
ip_check = 0
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
pyst_part = pysixtrack.Particles.from_dict(input_data['particle'])
for _ in range(n_turns):
    sequence.track(pyst_part)

for vv in vars_to_check:
    pyst_value = getattr(pyst_part, vv)
    xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
    passed = np.isclose(xt_value, pyst_value, rtol=1e-9, atol=1e-11)
    print(f'Check var {vv}:\n'
          f'    pyst:   {pyst_value: .7e}\n'
          f'    xtrack: {xt_value: .7e}\n')
    if not passed:
        raise ValueError

