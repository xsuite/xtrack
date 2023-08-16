"""
Beam Profile Monitor

Author: Philipp Niedermayer
Date: 2023-08-15
"""

import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..beam_elements import Marker
from ..internal_record import RecordIndex
from ..general import _pkg_root




class BeamProfileMonitorRecord(xo.Struct):
    counts_x = xo.Int64[:]
    counts_y = xo.Int64[:]


class BeamProfileMonitor(BeamElement):

    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'frev': xo.Float64,
        'sampling_frequency': xo.Float64,
        'raster_size_x': xo.Int64,
        'raster_range_x0': xo.Float64,
        'raster_range_dx': xo.Float64,
        'raster_size_y': xo.Int64,
        'raster_range_y0': xo.Float64,
        'raster_range_dy': xo.Float64,
        'sample_size': xo.Int64,
        '_index': RecordIndex,
        'data': BeamProfileMonitorRecord,
    }
    
    behaves_like_drift = True
    allow_backtrack = True
    
    properties = [field.name for field in BeamProfileMonitorRecord._fields]

    _extra_c_sources = [
        _pkg_root.joinpath('monitors/beam_profile_monitor.h')
    ]

    
    def __init__(self, *, particle_id_range=None, particle_id_start=None, num_particles=None,
                 start_at_turn=None, stop_at_turn=None, frev=None, sampling_frequency=None,
                 raster_size_x=None, raster_range_x=None, raster_size_y=None, raster_range_y=None,
                 raster_size=None, raster_range=None, _xobject=None, **kwargs):
        """
        Monitor to save the transverse profile of the tracked particles

        
        The monitor allows for arbitrary sampling rate and can thus not only be used to monitor
        bunch profiles, but also for coasting beams. Internally, the particle arrival time
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
        - `intensity_x`, `intensity_y`: the profile intensity (particle count per bin) as 2D array of shape (size, raster_size)
                                        where size = round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)
        - `raster_edges_x`, `raster_edges_y`: the profile edges (position) as 1D array of shape (raster_size+1)
        - `raster_midpoints_x`, `raster_midpoints_y`: the profile position (midpoints) as 1D array of shape (raster_size)
        
        

        Args:
            num_particles (int, optional): Number of particles to monitor. Defaults to -1 which means ALL.
            particle_id_start (int, optional): First particle id to monitor. Defaults to 0.
            particle_id_range (tuple, optional): Range of particle ids to monitor (start, stop). Stop is exclusive.
                                                 Defaults to (particle_id_start, particle_id_start+num_particles).
            start_at_turn (int): First turn of reference particle (inclusive) at which to monitor.
            stop_at_turn (int): Last turn of reference particle (exclusiv) at which to monitor.
            frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
            sampling_frequency (float): Sampling frequency in Hz.
            raster_size_x (int, optional): Number of raster points of the horizontal profile. Defaults to 128.
            raster_range_x (float or tuple): Extend of raster points of the profile. Either a tuple of (min_x, max_x)
                                             or a scalar `width` in which case a range of (-width/2, width/2) is used.
            raster_size_y (int, optional): Number of raster points of the vertical profile. Defaults to 128.
            raster_range_y (float or tuple): Extend of raster points of the profile. Either a tuple of (min_y, max_y)
                                             or a scalar `width` in which case a range of (-width/2, width/2) is used.
            raster_size: Default value for `raster_size_x` and `raster_size_y` if these are not set.
            raster_range: Default value for `raster_range_x` and `raster_range_y` if these are not set.

        """
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        
        else:

            # dict parameters
            if particle_id_range is None:
                if particle_id_start is None:
                    particle_id_start = 0
                if num_particles is None:
                    num_particles = -1
            elif particle_id_start is None and num_particles is None:
                particle_id_start = particle_id_range[0]
                num_particles = particle_id_range[1] - particle_id_range[0]
            else:
                raise ValueError("Parameter `particle_id_range` must not be used together with `num_particles` and/or `particle_id_start`")
            if start_at_turn is None:
                start_at_turn = 0
            if stop_at_turn is None:
                stop_at_turn = 0
            if frev is None:
                frev = 1
            if sampling_frequency is None:
                sampling_frequency = 1

            if raster_size_x is None:
                raster_size_x = raster_size or 128
            if raster_range_x is None:
                if raster_range is None:
                    raise ValueError("Either `raster_range_x` or `raster_range` must be provided")
                raster_range_x = raster_range
            if np.isscalar(raster_range_x):
                raster_range_x = (-raster_range_x/2, raster_range_x/2)
            raster_range_x0 = raster_range_x[0]
            raster_range_dx = (raster_range_x[1]-raster_range_x[0])/raster_size_x

            if raster_size_y is None:
                raster_size_y = raster_size or 128
            if raster_range_y is None:
                if raster_range is None:
                    raise ValueError("Either `raster_range_y` or `raster_range` must be provided")
                raster_range_y = raster_range
            if np.isscalar(raster_range_y):
                raster_range_y = (-raster_range_y/2, raster_range_y/2)
            raster_range_y0 = raster_range_y[0]
            raster_range_dy = (raster_range_y[1]-raster_range_y[0])/raster_size_y
                        
            sample_size = int(round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev))

            if "data" not in kwargs:
                # explicitely init with zeros (instead of size only) to have consistent initial values
                kwargs["data"] = dict(
                    counts_x = np.zeros(sample_size*raster_size_x),
                    counts_y = np.zeros(sample_size*raster_size_y),
                )

            super().__init__(particle_id_start=particle_id_start, num_particles=num_particles,
                             start_at_turn=start_at_turn, stop_at_turn=stop_at_turn, frev=frev,
                             sampling_frequency=sampling_frequency, raster_size_x=raster_size_x,
                             raster_range_x0=raster_range_x0, raster_range_dx=raster_range_dx,
                             raster_size_y=raster_size_y, raster_range_y0=raster_range_y0, 
                             raster_range_dy=raster_range_dy, sample_size=sample_size, **kwargs)


    def __repr__(self):
        return (
            f"{type(self).__qualname__}(start_at_turn={self.start_at_turn}, stop_at_turn={self.stop_at_turn}, "
            f"particle_id_start={self.particle_id_start}, num_particles={self.num_particles}, frev={self.frev}, "
            f"sampling_frequency={self.sampling_frequency}) at {hex(id(self))}"
        )

    @property
    def raster_edges_x(self):
        """Edge positions of horizontal profile as array of shape (raster_size+1)"""
        return self.raster_range_x0 + self.raster_range_dx * np.arange(self.raster_size_x + 1)
    
    @property
    def raster_midpoints_x(self):
        """Profile positions (midpoints) of horizontal profile as array of shape (raster_size)"""
        return self.raster_edges_x[1:] - self.raster_range_dx/2

    @property
    def intensity_x(self):
        """Horizontal profile intensity (particle count per bin) as 2D array of shape (size, raster_size)
           where size = round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)"""
        x = self.data.counts_x.to_nparray()
        return x.reshape((-1, self.raster_size_x))

    @property
    def raster_edges_y(self):
        """Edge positions of vertical profile as array of shape (raster_size+1)"""
        return self.raster_range_y0 + self.raster_range_dy * np.arange(self.raster_size_y + 1)
    
    @property
    def raster_midpoints_y(self):
        """Profile positions (midpoints) of vertical profile as array of shape (raster_size)"""
        return self.raster_edges_y[1:] - self.raster_range_dy/2

    @property
    def intensity_y(self):
        """Vertical profile intensity (particle count per bin) as 2D array of shape (size, raster_size)
           where size = round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)"""
        y = self.data.counts_y.to_nparray()
        return y.reshape((-1, self.raster_size_y))


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)
