import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp


fname_sequence = '../../test_data/hllhc_14/line_and_particle.json'

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
sequence = xt.Line.from_dict(input_data['line'])

##################
# Build TrackJob #
##################
tracker = xt.Tracker(_context=context, line=sequence)

######################
# Get some particles #
######################
particles = xp.Particles(_context=context,
                         p0c=6500e9,
                         x=np.linspace(-1e-3, 1e-3, n_part),
                         px=np.linspace(-1e-5, 1e-5, n_part),
                         y=np.linspace(-2e-3, 2e-3, n_part),
                         py=np.linspace(-3e-5, 3e-5, n_part),
                         zeta=np.linspace(-1e-2, 1e-2, n_part),
                         delta=np.linspace(-1e-4, 1e-4, n_part),
                         )
#########
# Track #
#########
tracker.track(particles, num_turns=1)


