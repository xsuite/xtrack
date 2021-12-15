import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

import twiss_from_tracker as tft

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath(
                        './hllhc_14/line_and_particle.json')

fname_line_particles = './temp_precise_lattice/xtline.json'

####################
# Choose a context #
####################

context = xo.ContextCpu()


with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])


print('Build tracker...')
tracker = xt.Tracker(_context=context,
            line=line,
            )

part0 = xp.Particles(_context=context, **input_data['particle'])

twiss = tft.twiss_from_tracker(tracker, particle_ref=part0,
        r_sigma=0.01, nemitt_x=1e-6, nemitt_y=2.5e-6,
        n_theta=1000, delta_disp=1e-5, delta_chrom = 1e-4)
