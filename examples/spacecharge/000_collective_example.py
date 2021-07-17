import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xtrack as xt
import xfields as xf

fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_with_spacecharge_and_particle.json')

##################
# Get a sequence #
##################
with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xl.Line.from_dict(input_data['line'])

###############################################
# Choose a context and create a memory buffer #
##############################################
#context = xo.ContextCpu()
context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

_buffer = context.new_buffer()

###################################
# Setup quasi-frozen space charge #
###################################
spch_elements = xf.replace_spaceharge_with_quasi_frozen(
                                    sequence, _buffer=_buffer)

#################
# Build Tracker #
#################
tracker= xt.Tracker(_buffer=_buffer,
             sequence=sequence)

######################
# Get some particles #
######################
particles = xt.Particles(_context=context, **input_data['particle'])

#########
# Track #
#########
tracker.track(particles, num_turns=10,
              turn_by_turn_monitor=True)

