"""
Beam Size Monitor

Author: Philipp Niedermayer, Cristopher Cortes
Date: 2023-08-14
Edit: 2024-06-12
"""

import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..beam_elements import Marker
from ..internal_record import RecordIndex
from ..general import _pkg_root

class BunchMonitorRecord(xo.Struct):
    count = xo.Float64[:]
    zeta_sum = xo.Float64[:]
    zeta2_sum = xo.Float64[:]
    delta_sum = xo.Float64[:]
    delta2_sum = xo.Float64[:]

class BunchMonitor(BeamElement):

    _xofields={
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'start_at_turn': xo.Int64,
        'stop_at_turn': xo.Int64,
        'frev': xo.Float64,
        'harmonic': xo.Int64,
        '_index': RecordIndex,
        'data': BunchMonitorRecord,
    }

    behaves_like_drift = True
    allow_loss_refinement = True

    properties = [field.name for field in BunchMonitorRecord._fields]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/atomicadd.h'),
        _pkg_root.joinpath('monitors/bunch_monitor.h')
    ]

    def __init__(self, *, particle_id_range=None, particle_id_start=None, num_particles=None,
                 start_at_turn=None, stop_at_turn=None, frev=None,
                 harmonic=None, _xobject=None, **kwargs):
        """
        Monitor to save the longitudinal bunch position and size (mean and std of zeta) as well as mean and std of momentum spread (delta)


        The monitor allows for arbitrary sampling rate and can thus not only be used to monitor
        bunch emittance, but also to record coasting beams. Internally, the particle arrival time
        is used when determining the record index:

            i = harmonic * ( ( at_turn - start_turn ) -  f_rev * zeta / beta0 / c0 )

        where zeta=(s-beta0*c0*t) is the longitudinal coordinate of the particle, beta0 the
        relativistic beta factor of the particle, c0 is the speed of light, at_turn is the
        current turn number, f_rev is the revolution frequency, and sampling_frequency is the
        sampling frequency.

        Note that the index is rounded, i.e. the result array represents data of particles
        equally distributed around the reference particle. For example, if the sampling_frequency
        is twice the revolution frequency, the first item contains data from particles in the
        range zeta/circumference = -0.25 .. 0.25, the second item in the range 0.25 .. 0.75 and
        so on.

        The monitor is a carbon copy of the beam size monitor but dedicated to the longitudinal coordinates.

        The monitor provides the following data:
        - `count` Number of particles
        - `zeta_mean`, `delta_mean` Beam position in m and unitless (centroid, i.e. mean of particle zeta, delta)
        - `zeta_std`, `delta_std` Beam size in m (standard deviation of particle zeta, delta)
        - `zeta_var`, `delta_var` Variance of particle zeta [m²], delta  (= std**2)
        - `zeta_sum`, `delta_sum` Sum of particle zeta [m], delta (= mean * count)
        - `zeta2_sum`, `delta2_sum` Sum of particle zeta [m²], delta squared  (= (std**2 + mean**2) * count)
        each as an array of size:
            size = int(( stop_at_turn - start_at_turn ) * harmonic)

        Args:
            num_particles (int, optional): Number of particles to monitor. Defaults to -1 which means ALL.
            particle_id_start (int, optional): First particle id to monitor. Defaults to 0.
            particle_id_range (tuple, optional): Range of particle ids to monitor (start, stop). Stop is exclusive.
                                                 Defaults to (particle_id_start, particle_id_start+num_particles).
            start_at_turn (int): First turn of reference particle (inclusive) at which to monitor.
            stop_at_turn (int): Last turn of reference particle (exclusiv) at which to monitor.
            frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
            harmonic (int): Harmonic of the revolution frequency.

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
            if harmonic is None:
                harmonic = 1

            if "data" not in kwargs:
                # explicitely init with zeros (instead of size only) to have consistent initial values
                size = int(round(( stop_at_turn - start_at_turn ) * harmonic))
                kwargs["data"] = {prop: np.zeros(size) for prop in self.properties}

            super().__init__(particle_id_start=particle_id_start, num_particles=num_particles,
                             start_at_turn=start_at_turn, stop_at_turn=stop_at_turn, frev=frev,
                             harmonic=harmonic, **kwargs)

    def __repr__(self):
        return (
            f"{type(self).__qualname__}(start_at_turn={self.start_at_turn}, stop_at_turn={self.stop_at_turn}, "
            f"particle_id_start={self.particle_id_start}, num_particles={self.num_particles}, frev={self.frev}, "
            f"harmonic={self.harmonic}) at {hex(id(self))}"
        )

    def __getattr__(self, attr):
        if attr in self.properties:
            return getattr(self.data, attr).to_nparray()

        if attr in ('zeta_mean', 'delta_mean', 'zeta_cen', 'delta_cen', 'zeta_centroid', 'delta_centroid'):
            with np.errstate(invalid='ignore'):  # NaN for zero particles is expected behaviour
                attri = attr.split('_')[0]
                return getattr(self, attri+"_sum") / self.count

        if attr in ('zeta_var', 'delta_var'):
            with np.errstate(invalid='ignore'):  # NaN for zero particles is expected behaviour
                # var = mean(x^2) - mean(x)^2
                attri = attr.split('_')[0]
                return getattr(self, attri+"2_sum") / self.count - getattr(self, attri+"_mean")**2

        if attr in ('zeta_std', 'delta_std'):
            attri = attr.split('_')[0]
            return getattr(self, attr[0]+"_var")**0.5

        return getattr(super(), attr)


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return Marker(_context=_context, _buffer=_buffer, _offset=_offset)
