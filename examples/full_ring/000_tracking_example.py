# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp


fname_line = '../../test_data/lhc_no_bb/line_and_particle.json'

num_turns = int(100)
n_part = 200

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

##############
# Get a line #
##############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])

##################
# Build TrackJob #
##################
line.build_tracker(_context=context)

######################
# Get some particles #
######################
particles = xp.Particles(_context=context,
                         p0c=6500e9,
                         x=np.random.uniform(-1e-3, 1e-3, n_part),
                         px=np.random.uniform(-1e-5, 1e-5, n_part),
                         y=np.random.uniform(-2e-3, 2e-3, n_part),
                         py=np.random.uniform(-3e-5, 3e-5, n_part),
                         zeta=np.random.uniform(-1e-2, 1e-2, n_part),
                         delta=np.random.uniform(-1e-4, 1e-4, n_part),
                         )
#########
# Track #
#########
line.track(particles, num_turns=num_turns)
