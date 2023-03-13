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

        i = sampling_frequency * ( ( at_turn - start_turn ) / f_rev - zeta / beta0 / c0 )

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
        - frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
        - start_turn (int): Turn of the reference particle when to start excitation.
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
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'samples': xo.Float64[:],
        'nsamples': xo.Int64,
        'sampling_frequency': xo.Float64,
        'frev': xo.Float64,
        'start_turn': xo.Int64,
        'nduration': xo.Int64,
        }

    _extra_c_sources = [_pkg_root.joinpath('beam_elements/elements_src/exciter.h')]


    def __init__(self, *, samples=None, nsamples=None, sampling_frequency=0, frev=0, knl=[1], ksl=[], start_turn=0, duration=None, _xobject=None, **kwargs):

        if _xobject is not None:
            super().__init__(_xobject=_xobject)

        else:

            # sanitize knl and ksl array length
            n = max(len(knl), len(ksl))
            nknl = np.zeros(n, dtype=np.float64)
            nksl = np.zeros(n, dtype=np.float64)
            if knl is not None:
                nknl[:len(knl)] = np.array(knl)
            if ksl is not None:
                nksl[:len(ksl)] = np.array(ksl)
            kwargs["order"] = n - 1

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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(knl=-ctx2np(self.knl),
                              ksl=-ctx2np(self.ksl),
                              samples=self.samples,
                              nsamples=self.nsamples,
                              sampling_frequency=self.sampling_frequency,
                              frev=self.frev,
                              start_turn=self.start_turn,
                              _context=_context, _buffer=_buffer, _offset=_offset)
