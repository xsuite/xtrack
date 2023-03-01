"""
Monitor to save particle state for last N turns before respective loss

Author: Philipp Niedermayer
Date: 2023-01-19
"""


import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..general import _pkg_root


class LastTurnsData(xo.Struct):

    lost_at_offset = xo.UInt32[:]

    # store with 32bit precission to save memory
    particle_id = xo.UInt32[:]  # TODO: can use xo.Int32[:] after fixing https://github.com/xsuite/xsuite/issues/283
    at_turn = xo.UInt32[:]      # TODO: can use xo.Int32[:] after fixing https://github.com/xsuite/xsuite/issues/283
    x = xo.Float32[:]
    px = xo.Float32[:]
    y = xo.Float32[:]
    py = xo.Float32[:]
    delta = xo.Float32[:]
    zeta = xo.Float32[:]
    


class LastTurnsMonitor(BeamElement):
    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'n_last_turns': xo.Int64,
        'every_n_turns': xo.Int64,
        'data': LastTurnsData,
    }
    
    # TODO: find a way to dynamically change what properties are being saved by this monitor
    properties = [field.name for field in LastTurnsData._fields if field.name != 'lost_at_offset']


    _extra_c_sources = [
        _pkg_root.joinpath('monitors/last_turns_monitor.h')
    ]


    def __init__(self, *, n_last_turns=None, num_particles=None, particle_id_range=None, every_n_turns=1, _xobject=None, **kwargs):
        """Monitor to save particle data in last turns before respective particle loss
        
        The monitor provides the following data as 2D array of shape (num_particles, n_last_turns),
        where the first index corresponds to the particle_id in particle_id_range
        and the second index corresponds to the turn (or every_n_turns) before the respective particle is lost:
        `particle_id`, `at_turn`, `x`, `px`, `y`, `py`, `delta`, `zeta`
        
        Args:
            n_last_turns (int): Amount of turns to store before particle loss.
            particle_id_range (tuple): Range of particle ids to monitor (start, stop).
            num_particles (int, optional): Number of particles. Equal to passing particle_id_range=(0, num_particles).
            every_n_turns (int, optional): Save only every n-th turn, i.e. turn numbers which are a multiples of this.
                Because `n_last_turns` defines the amount of turns to store (and not the range), the data will cover turn
                numbers up to `n_last_turns*every_n_turns` turns before particle loss.
        
        Example:
            monitor = LastTurnsMonitor(n_last_turns=5, particle_id_range=(1, 5))
            monitor.at_turn[:,-1]  # last turn before loss of each particle, respectively
            monitor.x[3,-2]  # x coordinate in one but last turn of particle with id 4
        
        """
        
        
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        
        else:

            if num_particles is not None and particle_id_range is None:
                particle_id_start = 0
            elif particle_id_range is not None and num_particles is None:
                particle_id_start = particle_id_range[0]
                num_particles = particle_id_range[1] - particle_id_range[0]
            else:
                raise ValueError("Exactly one of `num_particles` or `particle_id_range` parameters must be specified")
            
            # explicitely init with zeros (instead of size only) to have consistent default values for untouched arrays
            # see also https://github.com/xsuite/xsuite/issues/294
            data = {prop: [0]*(num_particles*n_last_turns) for prop in self.properties} # particle data
            data['lost_at_offset'] = [0]*(num_particles)  # meta data (rolling buffer offset per particle)

            super().__init__(n_last_turns=n_last_turns, particle_id_start=particle_id_start,
                             num_particles=num_particles, every_n_turns=every_n_turns, data=data, **kwargs)
    
    def __repr__(self):
        return (
            f"{type(self).__qualname__}(n_last_turns={self.n_last_turns}, every_n_turns={self.every_n_turns}, "
            f"particle_id_start={self.particle_id_start}, num_particles={self.num_particles}) at {hex(id(self))}"
        )


    def __getattr__(self, attr):
        if attr in self.properties:
            val = getattr(self.data, attr)
            val = val.to_nparray() # = self.data._buffer.context.nparray_from_context_array(val)
            val = np.reshape(val, (self.num_particles, self.n_last_turns))
            off = self.data.lost_at_offset.to_nparray() + 1
            # correct for rolling buffer offset
            r, c = np.ogrid[:val.shape[0], :val.shape[1]]
            c = (c + off[:, np.newaxis]) % val.shape[1]
            return val[r, c]
        return super().__getattr__(attr)



