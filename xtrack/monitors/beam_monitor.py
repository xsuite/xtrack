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




class BeamMonitorRecord(xo.Struct):
    count = xo.Int64[:]
    x_sum = xo.Float64[:]
    y_sum = xo.Float64[:]


class BeamMonitor(BeamElement):

    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'frev': xo.Float64,
        'sampling_frequency': xo.Float64,
        '_index': RecordIndex,
        'data': BeamMonitorRecord,
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

    
    def __init__(self, *, particle_id_range=None, num_particles=None,
                 start_at_turn=None, stop_at_turn=None, frev=None, 
                 sampling_frequency=None, _xobject=None, **kwargs):
        """
        Monitor to save the transversal centroid of the tracked particles

        
        The monitor allows for arbitrary sampling rate and can thus not only be used to monitor
        bunch positions, but also to record schottky spectra. Internally, the particle arrival time
        is used when determining the record index:

            i = sampling_frequency * ( ( at_turn - start_turn ) / f_rev - zeta / beta0 / c0 )

        where zeta=(s-beta0*c0*t) is the longitudinal coordinate of the particle, beta0 the
        relativistic beta factor of the particle, c0 is the speed of light, at_turn is the
        current turn number, f_rev is the revolution frequency, and sampling_frequency is the
        sampling frequency.

        Note that the index is rounded, i.e. the result array represents data of particles
        equally distributed around the reference particle. For example, if the sampling_frequency
        is twice the revolution frequency, the first item contains data from particles in the
        range zeta/circumference = -0.25 .. 0.25, the second item in the range 0.25 .. 0.75 and
        so on.


        The monitor provides the following data:
        `count`, `x_sum`, `x_mean`, `y_sum`, `y_mean`,
        each as an array of size:
            size = int(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)
        

        Args:
            particle_id_range (tuple): Range of particle ids to monitor (start, stop). Stop is exclusive.
            num_particles (int, optional): Number of particles. Equal to passing particle_id_range=(0, num_particles).
            start_at_turn (int): First turn of reference particle (inclusive) at which to monitor.
            stop_at_turn (int): Last turn of reference particle (exclusiv) at which to monitor.
            frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
            sampling_frequency (float): Sampling frequency in Hz.

        """
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        
        else:

            # dict parameters
            if num_particles is not None and particle_id_range is None:
                particle_id_start = 0
            elif particle_id_range is not None and num_particles is None:
                particle_id_start = particle_id_range[0]
                num_particles = particle_id_range[1] - particle_id_range[0]
            else:
                raise ValueError("Exactly one of `num_particles` or `particle_id_range` parameters must be specified")
            if start_at_turn is None:
                start_at_turn = 0
            if stop_at_turn is None:
                stop_at_turn = 0
            if frev is None:
                frev = 1
            if sampling_frequency is None:
                sampling_frequency = 1
            
            # explicitely init with zeros (instead of size only) to have consistent initial values
            size = int(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)
            data = {prop: np.zeros(size) for prop in self.properties}
            super().__init__(particle_id_start=particle_id_start, num_particles=num_particles,
                             start_at_turn=start_at_turn, stop_at_turn=stop_at_turn, frev=frev,
                             sampling_frequency=sampling_frequency, data=data, **kwargs)


    def __repr__(self):
        return (
            f"{type(self).__qualname__}(start_at_turn={self.start_at_turn}, stop_at_turn={self.stop_at_turn}, "
            f"particle_id_start={self.particle_id_start}, num_particles={self.num_particles}, frev={self.frev}, "
            f"sampling_frequency={self.sampling_frequency}) at {hex(id(self))}"
        )
    

    def __getattr__(self, attr):
        if attr in self.properties:
            return getattr(self.data, attr).to_nparray()
        
        if attr in ('x_mean', 'y_mean', 'x_cen', 'y_cen', 'x_centroid', 'y_centroid'):
            with np.errstate(invalid='ignore'):  # NaN for zero particles is expected behaviour
                return getattr(self, attr[0]+"_sum") / self.count
        
        return getattr(super(), attr)


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)
