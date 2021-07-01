import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xtrack as xt
import xfields as xf

class PICCollection:

    def __init__(self, _buffer,
                 nx_grid,
                 ny_grid,
                 nz_grid,
                 x_lim_min,
                 x_lim_max,
                 y_lim_min,
                 y_lim_max,
                 z_range,
                 n_lims_x,
                 n_lims_y,
                 solver='FFTSolver2p5D',
                 apply_z_kick=False,
                     ):

        self._buffer = _buffer

        self.nx_grid = nx_grid
        self.ny_grid = ny_grid
        self.nz_grid = nz_grid

        self.z_range = z_range
        self.solver = solver
        self.apply_z_kick = apply_z_kick

        self.x_lims = np.linspace(x_lim_min, x_lim_max, n_lims_x)
        self.y_lims = np.linspace(y_lim_min, y_lim_max, n_lims_y)

        self._existing_pics = {}


    def get_pic(self, x_lim, y_lim):

        ix = np.argmin(np.abs(x_lim - self.x_lims))
        iy = np.argmin(np.abs(y_lim - self.y_lims))

        if (ix, iy) not in self._existing_pics.keys():
            print(f'Creating PIC ({ix}, {iy})')
            xlim_pic = self.x_lims[ix]
            ylim_pic = self.y_lims[iy]
            new_pic = xf.SpaceCharge3D(
                _buffer=self._buffer,
                length=0.,
                apply_z_kick=self.apply_z_kick,
                x_range=(-xlim_pic, xlim_pic),
                y_range=(-ylim_pic, ylim_pic),
                z_range=self.z_range,
                nx=self.nx_grid, ny=self.ny_grid, nz=self.nz_grid,
                solver=self.solver)
            self._existing_pics[ix, iy] = new_pic

        return self._existing_pics[ix, iy]


context = xo.ContextCpu()
_buffer = context.new_buffer()

fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_with_spacecharge_and_particle.json')

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

n_sigmas_range_pic = 10
nx_grid = 256
ny_grid = 256
nz_grid = 50
n_lims_x = 7
n_lims_y = 5
z_range=(-30e-2, 30e-2)

all_sc_elems = []
ind_sc_elems = []
all_sigma_x = []
all_sigma_y = []
for ii, ee in enumerate(sequence.elements):
    if ee.__class__.__name__ == 'SCQGaussProfile':
        all_sc_elems.append(ee)
        ind_sc_elems.append(ii)
        all_sigma_x.append(ee.sigma_x)
        all_sigma_y.append(ee.sigma_y)


x_lim_min = np.min(all_sigma_x) * (n_sigmas_range_pic + 0.5)
x_lim_max = np.max(all_sigma_x) * (n_sigmas_range_pic + 0.5)
y_lim_min = np.min(all_sigma_y) * (n_sigmas_range_pic + 0.5)
y_lim_max = np.max(all_sigma_y) * (n_sigmas_range_pic + 0.5)

pic_collection = PICCollection(_buffer,
    nx_grid=nx_grid, ny_grid=ny_grid, nz_grid=nz_grid,
    x_lim_min=x_lim_min, x_lim_max=x_lim_max, n_lims_x=n_lims_x,
    y_lim_min=y_lim_min, y_lim_max=y_lim_max, n_lims_y=n_lims_y,
    z_range=z_range)
