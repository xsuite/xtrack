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
        #newee.update_sigma_x_on_track = True # Commented out for test
        newee.iscollective=True
        newseq.elements[ii] = newee
        spch_elements.append(newee)

# # Test
# for ss in spch_elements:
#     ss.update_mean_x_on_track = True
#     ss.update_sigma_x_on_track = True

def _check_is_collective(ele):
    iscoll = not hasattr(ele, 'iscollective') or ele.iscollective
    return iscoll


# Split the sequence
parts = []
this_part = xl.Line(elements=[], element_names=[])
for nn, ee in zip(newseq.element_names, newseq.elements):
    if not _check_is_collective(ee):
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
    if not _check_is_collective(pp):
        tempxtline = xt.Line(_buffer=_buffer,
                           sequence=pp)
        pp.elements = tempxtline.elements
        noncollective_xelements += pp.elements

# Build tracker for all non collective elements
supertracker = xt.Tracker(_buffer=_buffer,
        sequence=xl.Line(elements=noncollective_xelements,
            element_names=[
                f'e{ii}' for ii in range(len(noncollective_xelements))]))

# Build trackers for non collective parts
for ii, pp in enumerate(parts):
    if not _check_is_collective(pp):
        parts[ii] = xt.Tracker(_buffer=_buffer,
                            sequence=pp,
                            element_classes=supertracker.element_classes,
                            track_kernel=supertracker.track_kernel,
                            skip_end_turn_actions=True)


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
self = supertracker
if turn_by_turn_monitor:

    monitor = self.particles_monitor_class(
        _context=_buffer.context,
        start_at_turn=0,
        stop_at_turn=n_turns,
        num_particles=particles.num_particles,
    )

#tracker.track(particles, num_turns=n_turns)
for tt in range(n_turns):
    if turn_by_turn_monitor:
        monitor.track(particles)
    for pp in parts:
        pp.track(particles)
    # Increment at_turn and reset at_element
    # (use the supertracker to perform only end-turn actions)
    supertracker.track(particles, ele_start=supertracker.num_elements,
                       num_elements=0)

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

