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
    counts_x = xo.Float64[:]
    counts_y = xo.Float64[:]


class BeamProfileMonitor(BeamElement):

    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'frev': xo.Float64,
        'sampling_frequency': xo.Float64,
        'nx': xo.Int64,
        'x_min': xo.Float64,
        'dx': xo.Float64,
        'ny': xo.Int64,
        'y_min': xo.Float64,
        'dy': xo.Float64,
        'sample_size': xo.Int64,
        '_index': RecordIndex,
        'data': BeamProfileMonitorRecord,
    }

    behaves_like_drift = True
    allow_loss_refinement = True

    properties = [field.name for field in BeamProfileMonitorRecord._fields]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/atomicadd.h'),
        _pkg_root.joinpath('monitors/beam_profile_monitor.h')
    ]


    def __init__(self, *, particle_id_range=None, particle_id_start=None, num_particles=None,
                 start_at_turn=None, stop_at_turn=None, frev=None, sampling_frequency=None,
                 nx=None, x_range=None, ny=None, y_range=None,
                 n=None, range=None, _xobject=None, **kwargs):
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
        - `x_intensity`, `y_intensity`: the profile intensity (particle count per bin) as 2D array of shape (size, n)
                                        where size = round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)
        - `x_edges`, `y_edges`: the profile edges (position) in m as 1D array of shape (n+1)
        - `x_grid`, `y_grid`: the profile position (midpoints) in m as 1D array of shape (n)

        Args:
            num_particles (int, optional): Number of particles to monitor. Defaults to -1 which means ALL.
            particle_id_start (int, optional): First particle id to monitor. Defaults to 0.
            particle_id_range (tuple, optional): Range of particle ids to monitor (start, stop). Stop is exclusive.
                                                 Defaults to (particle_id_start, particle_id_start+num_particles).
            start_at_turn (int): First turn of reference particle (inclusive) at which to monitor.
            stop_at_turn (int): Last turn of reference particle (exclusiv) at which to monitor.
            frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
            sampling_frequency (float): Sampling frequency in Hz.
            nx (int, optional): Number of raster points of the horizontal profile. Defaults to 128.
            x_range (float or tuple): Extend of raster points of the profile in m. Either a tuple of (min_x, max_x)
                                             or a scalar `width` in which case a range of (-width/2, width/2) is used.
            ny (int, optional): Number of raster points of the vertical profile. Defaults to 128.
            y_range (float or tuple): Extend of raster points of the profile in m. Either a tuple of (min_y, max_y)
                                             or a scalar `width` in which case a range of (-width/2, width/2) is used.
            n: Default value for `nx` and `ny` if these are not set.
            range: Default value for `x_range` and `y_range` if these are not set.

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

            if nx is None:
                nx = n or 128
            if x_range is None:
                if range is None:
                    raise ValueError("Either `x_range` or `range` must be provided")
                x_range = range
            if np.isscalar(x_range):
                x_range = (-x_range/2, x_range/2)
            x_min = x_range[0]
            dx = (x_range[1]-x_range[0])/nx

            if ny is None:
                ny = n or 128
            if y_range is None:
                if range is None:
                    raise ValueError("Either `y_range` or `range` must be provided")
                y_range = range
            if np.isscalar(y_range):
                y_range = (-y_range/2, y_range/2)
            y_min = y_range[0]
            dy = (y_range[1]-y_range[0])/ny

            sample_size = int(round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev))

            if "data" not in kwargs:
                # explicitely init with zeros (instead of size only) to have consistent initial values
                kwargs["data"] = dict(
                    counts_x = np.zeros(sample_size*nx),
                    counts_y = np.zeros(sample_size*ny),
                )

            super().__init__(particle_id_start=particle_id_start, num_particles=num_particles,
                             start_at_turn=start_at_turn, stop_at_turn=stop_at_turn, frev=frev,
                             sampling_frequency=sampling_frequency, nx=nx,
                             x_min=x_min, dx=dx,
                             ny=ny, y_min=y_min,
                             dy=dy, sample_size=sample_size, **kwargs)


    def __repr__(self):
        return (
            f"{type(self).__qualname__}(start_at_turn={self.start_at_turn}, stop_at_turn={self.stop_at_turn}, "
            f"particle_id_start={self.particle_id_start}, num_particles={self.num_particles}, frev={self.frev}, "
            f"sampling_frequency={self.sampling_frequency}) at {hex(id(self))}"
        )

    @property
    def x_edges(self):
        """Edge positions of horizontal profile as array of shape (n+1)"""
        return self.x_min + self.dx * np.arange(self.nx + 1)

    @property
    def x_grid(self):
        """Profile positions (midpoints) of horizontal profile as array of shape (n)"""
        return self.x_edges[1:] - self.dx/2

    @property
    def x_intensity(self):
        """Horizontal profile intensity (particle count per bin) as 2D array of shape (size, n)
           where size = round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)"""
        x = self.data.counts_x.to_nparray()
        return x.reshape((-1, self.nx))

    @property
    def y_edges(self):
        """Edge positions of vertical profile as array of shape (n+1)"""
        return self.y_min + self.dy * np.arange(self.ny + 1)

    @property
    def y_grid(self):
        """Profile positions (midpoints) of vertical profile as array of shape (n)"""
        return self.y_edges[1:] - self.dy/2

    @property
    def y_intensity(self):
        """Vertical profile intensity (particle count per bin) as 2D array of shape (size, n)
           where size = round(( stop_at_turn - start_at_turn ) * sampling_frequency / frev)"""
        y = self.data.counts_y.to_nparray()
        return y.reshape((-1, self.ny))

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)
