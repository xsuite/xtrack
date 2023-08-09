"""
Monitor to save the centroid of the many slices or time slots within a turn (like an actual BPM)

Author: Rahul Singh, Cristopher Cortes, Philipp Niedermayer
Date: 2023-06-10
"""

import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..beam_elements import Marker
from ..internal_record import RecordIndex
from ..general import _pkg_root


################################################################
# Definition of a beam element with an internal data recording #
################################################################

# We define a data structure to allow all elements of a new BeamElement type
# to store data in one place. Such a data structure needs to contain a field called
# `_index` of type `xtrack.RecordIndex`, which will be used internally to
# keep count of the number of records stored in the data structure. Together with
# the index, the structure can contain an arbitrary number of other fields (which
# need to be arrays) where the data will be stored.

class BeamMonitorRecord(xo.Struct):

    at_turn = xo.Int64[:]
    x_cen = xo.Float64[:]
    y_cen = xo.Float64[:]
    summed_particles = xo.Int64[:]
    last_particle_id = xo.Int64[:]

class BeamMonitor(BeamElement):
    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'start_at_turn':xo.Int64,
        'stop_at_turn' :xo.Int64,
        'rev_frequency':xo.Float64,
        'sampling_frequency':xo.Float64,
        '_index':RecordIndex,
        'data':BeamMonitorRecord,
    }
    
    behaves_like_drift = True
    allow_backtrack = True

    # TODO: it currently only works on CPU!
    needs_cpu = True
    iscollective = True
    
    properties = [field.name for field in BeamMonitorRecord._fields]

    _extra_c_sources = [
        _pkg_root.joinpath('monitors/beam_monitor.h')
    ]

    
    def __init__(self, *, particle_id_start=None,num_particles=None,start_at_turn=None, stop_at_turn=None,rev_frequency=None,sampling_frequency=None, _xobject=None, **kwargs):
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
        if rev_frequency is None:
            rev_frequency = 1
        if sampling_frequency is None:
            sampling_frequency = 1
        data = {prop: [0]*int((stop_at_turn-start_at_turn)*(sampling_frequency/rev_frequency)) for prop in self.properties}
        super().__init__(particle_id_start=particle_id_start,num_particles=num_particles,start_at_turn=start_at_turn, stop_at_turn=stop_at_turn,rev_frequency=rev_frequency,sampling_frequency=sampling_frequency,data=data, **kwargs)
    

    def __getattr__(self, attr):
        if attr in self.properties:
            if (attr == 'x_cen' or attr == 'y_cen'):
                position_sum = getattr(self.data, attr)
                position_sum = position_sum.to_nparray()
# Convert from position sum to centroid for each time slot
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
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)
