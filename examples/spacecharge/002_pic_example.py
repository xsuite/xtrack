import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xtrack as xt
import xfields as xf


fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_with_spacecharge_and_particle.json')

####################
# Choose a context #
####################

context = xo.ContextCpu()
_buffer = context.new_buffer()


##################
# Get a sequence #
##################

with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xl.Line.from_dict(input_data['line'])

n_sigmas_range_pic_x = 10
n_sigmas_range_pic_y = 9
nx_grid = 256
ny_grid = 256
nz_grid = 50
n_lims_x = 7
n_lims_y = 5
z_range=(-30e-2, 30e-2)

xf.replace_spaceharge_with_PIC(_buffer, sequence,
        n_sigmas_range_pic_x, n_sigmas_range_pic_y,
        nx_grid, ny_grid, nz_grid, n_lims_x, n_lims_y, z_range)
