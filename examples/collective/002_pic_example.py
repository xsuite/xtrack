import numpy as np

import xobjects as xo
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
pic_collection = PICCollection(_buffer,
    nx_grid=256, ny_grid=256, nz_grid=50,
    x_lim_min=0.042, x_lim_max=0.055, n_lims_x=7,
    y_lim_min=0.027, y_lim_max=0.042, n_lims_y=3,
    z_range=(-30e-2, 30e-2))
