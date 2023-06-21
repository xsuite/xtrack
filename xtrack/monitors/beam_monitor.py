"""
Monitor to save the centroid of the many slices within a turn (like an actual BPM)

Author: Rahul Singh
Date: 2023-06-10
"""

import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np
from pathlib import Path


################################################################
# Definition of a beam element with an internal data recording #
################################################################

# We define a data structure to allow all elements of a new BeamElement type
# to store data in one place. Such a data structure needs to contain a field called
# `_index` of type `xtrack.RecordIndex`, which will be used internally to
# keep count of the number of records stored in the data structure. Together with
# the index, the structure can contain an arbitrary number of other fields (which
# need to be arrays) where the data will be stored.

class BPMRecord(xo.Struct):

    at_turn = xo.Int64[:]
    x_cen = xo.Float64[:]
    y_cen = xo.Float64[:]
    summed_particles = xo.Int64[:]
    last_particle_id = xo.Int64[:]

class BPM(xt.BeamElement):
    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'start_at_turn':xo.Int64,
        'stop_at_turn' :xo.Int64,
        'samples_per_turn':xo.Int64,
        'sampling_frequency':xo.Int64,
        '_index':xt.RecordIndex,
        'data':BPMRecord,
    }
    
    behaves_like_drift = True
    allow_backtrack = True
    
    properties = [field.name for field in BPMRecord._fields]

    _extra_c_sources = [Path(__file__).parent.absolute().joinpath('BPM.h')]

    
    def __init__(self, *, particle_id_start=None,num_particles=None,start_at_turn=None, stop_at_turn=None,samples_per_turn=None,sampling_frequency=None, _xobject=None, **kwargs):
        """
        Monitor to save the transversal centroid data of the tracked particles
        """
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        if particle_id_start is None:
            particle_id_start = 0
        if num_particles is None:
            num_particles = 1
        if start_at_turn is None:
            start_at_turn = 0
        if stop_at_turn is None:
            stop_at_turn = 0
        if samples_per_turn is None:
            samples_per_turn = 1
        if sampling_frequency is None:
            sampling_frequency = 1
        data = {prop: [0]*int((stop_at_turn-start_at_turn)*samples_per_turn*1.05) for prop in self.properties} # 5% safety for particles much ahead
        super().__init__(particle_id_start=particle_id_start,num_particles=num_particles,start_at_turn=start_at_turn, stop_at_turn=stop_at_turn,samples_per_turn=samples_per_turn,sampling_frequency=sampling_frequency,data=data, **kwargs)
    

    def __getattr__(self, attr):
        if attr in self.properties:
            if (attr == 'x_cen' or attr == 'y_cen'):
                position_sum = getattr(self.data, attr)
                position_sum = position_sum.to_nparray()
                summed_particles = getattr(self.data,'summed_particles')
                summed_particles = summed_particles.to_nparray()
                val = np.zeros((len(position_sum)))
                for i in range(len(position_sum)):
                    if summed_particles[i] != 0:
                        val[i] = position_sum[i]/summed_particles[i]
                    else:
                        val[i] = 0
            else:
                val = getattr(self.data, attr)
                val = val.to_nparray()#self.data._buffer.context.nparray_from_context_array(val) # val.to_nparray()
            return val
        return super().__getattr__(attr)


    def __repr__(self):
        return (
            f"{type(self).__qualname__}(start_at_turn={self.start_at_turn}, stop_at_turn={self.stop_at_turn}, "
            f"particle_id_start={self.particle_id_start}, num_particles={self.num_particles}) at {hex(id(self))}"
        )

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return xt.Marker(_context=_context, _buffer=_buffer, _offset=_offset)
