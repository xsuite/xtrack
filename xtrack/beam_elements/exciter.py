"""
Exciter element

Author: Philipp Niedermayer
Date: 11.11.2022

"""


import xobjects as xo
import numpy as np

from ..base_element import BeamElement
from ..general import _pkg_root


class Exciter(BeamElement):
    """Beam element modeling a transverse exciter as a time-dependent thin multipole.

    The given multipole components (knl and ksl) are scaled according to a custom waveform,
    allowing for arbitrary time dependence. The waveform is specified by an array of samples:

        knl(t) = knl * samples[i]

    It is *not* assumed that the variations are slow compared to the revolution frequency
    and the particle arrival time is taken into account when determining the sample index:

        i = floor( sampling_frequency * ( ( at_turn - start_turn ) / f_rev - zeta / beta0 / c0 ) )

    With absolute_time, the index follows the shared clock instead:

        i = floor( sampling_frequency * ( time_s - start_turn / f_rev + ( s - zeta ) / beta0 / c0 ) )

    where zeta=(s-beta0*c0*t) is the longitudinal coordinate of the particle, beta0 the
    relativistic beta factor of the particle, c0 is the speed of light, at_turn is the
    current turn number, f_rev is the revolution frequency, and sampling_frequency is the sampling
    frequency. The excitation starts with the first sample when the reference particle 
    arrives at the element in start_turn.

    For example, to compute samples for a sinusoidal excitation with frequency f_ex one
    would calculate the waveform as: samples[i] = np.sin(2*np.pi*f_ex*i/sampling_frequency)

    Notes:
        - This is not to be confused with an RFMultipole, which inherits the characteristics
          of an RFCavity and whose oscillation is therefore with respect to the reference
          particle. While the frequency of the RFMultipole is therefore restricted to
          harmonics of the revolution frequency, the exciter allows for arbitrary frequencies.
        - This is also not to be confused with an ACDipole, for which the oscillation is 
          assumed to be slow compared to the revolution frequency and the kick is the same 
          for all particles independent of their longitudinal coordinate.


    Parameters:
        - knl (float array): Normalized integrated strength of the normal components. Unit: m^-n (n=0,1,2,...).
        - ksl (float array): Normalized integrated strength of the skew components. Unit: m^-n (n=0,1,2,...).
        - order (int): Multipole order (readonly), i.e. largest n with non-zero knl or ksl.
        - samples (float array): Samples of excitation strength to scale knl and ksl as function of time.
        - nsamples (int): Number of samples. Pass this instead of samples to reserve memory for later initialisation.
        - sampling_frequency (float): Sampling frequency in Hz.
        - frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index). Required unless absolute_time is set: without it the turn clock has no time.
        - start_turn (int): Turn of the reference particle when to start excitation.
        - absolute_time (bool): If True, the sample index follows the absolute arrival time, time_s + (s - zeta) / beta0 / c0, from the time start_turn / frev, so the excitation keeps its own clock while the reference energy changes. Default is False.
        - duration (float): Duration of excitation in s (defaults to nsamples/sampling_frequency). Repeats the waveform to fill the duration.

    Example:
        >>> fs = 10e6 # sampling frequency in Hz
        >>> 
        >>> # load waveform into memory
        >>> signal = np.copy(np.memmap("signal.10MHz.float32", np.float32))
        >>> 
        >>> # alternatively compute samples on the fly, for example a simple sine at 500 kHz ...
        >>> t = np.arange(1000)/fs
        >>> f_ex = 5e5 # excitation frequency in Hz
        >>> signal = np.sin(2*np.pi*f_ex*t)
        >>> 
        >>> # ... or a sweep from 500 to 800 kHz
        >>> f_ex_1 = 8e5
        >>> signal = scipy.signal.chirp(t, f_ex, t[-1], f_ex_1)
        >>> 
        >>> # create the exciter
        >>> frev = 1e6 # revolution frequency in Hz
        >>> k0l = 0.1 # this is scaled by the waveform
        >>> exciter = Exciter(samples=signal, sampling_frequency=fs, frev=frev, start_turn=0, knl=[k0l])
        >>> 
        >>> # add it to the line
        >>> line.insert_element(index=..., name=..., element=exciter)

    """

    _xofields={
        'order': xo.Int64,
        'nsamples': xo.Int64,
        'sampling_frequency': xo.Float64,
        'frev': xo.Float64,
        'start_turn': xo.Int64,
        'absolute_time': xo.Int64,
        'nduration': xo.Int64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'samples': xo.Float32[:],
        }

    _rename = {'frev': '_frev', 'start_turn': '_start_turn', 'absolute_time': '_absolute_time'}

    has_backtrack = True

    _extra_c_sources = ['#include "xtrack/beam_elements/elements_src/exciter.h"']


    def __init__(self, *, samples=None, nsamples=None, sampling_frequency=0, frev=0, knl=None, ksl=None, start_turn=0, duration=None, _xobject=None, **kwargs):

        if _xobject is not None:
            super().__init__(_xobject=_xobject)

        else:
            # The default is needed, as we wish to be able to instantiate
            # elements without properties (empty elements can be templates).
            if knl is None and ksl is None:
                knl = [0]
                ksl = []
            elif knl is None:
                knl = []
            elif ksl is None:
                ksl = []

            # sanitize knl and ksl array length
            n = max(len(knl), len(ksl))
            nknl = np.zeros(n, dtype=np.float64)
            nksl = np.zeros(n, dtype=np.float64)
            if knl is not None:
                nknl[:len(knl)] = np.array(knl)
            if ksl is not None:
                nksl[:len(ksl)] = np.array(ksl)
            kwargs["order"] = n - 1

            # The turn clock divides by frev in the kernel, which is undefined at zero.
            if ((samples is not None or nsamples is not None) and not frev
                    and (not kwargs.get('absolute_time') or start_turn)):
                raise ValueError('`frev` relates the turn number to the samples: '
                                 'set it, or use `absolute_time` with no `start_turn`.')

            # determine sample length or create empty samples for later initialisation
            if samples is not None:
                if nsamples is not None and nsamples != len(samples):
                    raise ValueError("Only one of samples or nsamples may be specified")
                nsamples = len(samples)
            if samples is None:
                samples = np.zeros(nsamples)
            kwargs["nduration"] = nsamples if duration is None else (duration * sampling_frequency)

            super().__init__(
                knl=nknl, ksl=nksl, samples=samples, nsamples=nsamples,
                sampling_frequency=sampling_frequency,
                frev=frev, start_turn=start_turn, **kwargs)

    @property
    def duration(self):
        return self.nduration / self.sampling_frequency

    @duration.setter
    def duration(self, duration):
        self.nduration = duration * self.sampling_frequency

    @property
    def frev(self):
        return self._frev

    @frev.setter
    def frev(self, value):
        # The turn clock divides by frev in the kernel, which is undefined at zero.
        if not value and (not self.absolute_time or self.start_turn):
            raise ValueError('`frev` relates the turn number to the samples: '
                             'set it, or use `absolute_time` with no `start_turn`.')
        self._frev = value

    @property
    def start_turn(self):
        return self._start_turn

    @start_turn.setter
    def start_turn(self, value):
        # A start turn relates to time through frev, even on absolute time.
        if value and not self.frev:
            raise ValueError('`start_turn` needs `frev` to be set.')
        self._start_turn = value

    @property
    def absolute_time(self):
        return self._absolute_time

    @absolute_time.setter
    def absolute_time(self, value):
        # The turn clock divides by frev in the kernel, which is undefined at zero.
        if not value and not self.frev:
            raise ValueError('The turn clock needs `frev` to be set.')
        self._absolute_time = value

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(knl=-ctx2np(self.knl),
                              ksl=-ctx2np(self.ksl),
                              samples=self.samples,
                              nsamples=self.nsamples,
                              sampling_frequency=self.sampling_frequency,
                              frev=self.frev,
                              start_turn=self.start_turn,
                              absolute_time=self.absolute_time,
                              _context=_context, _buffer=_buffer, _offset=_offset)
