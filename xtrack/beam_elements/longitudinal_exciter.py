"""
LongitudinalExciter element

Author: Pablo Arrutia
Date: 15.07.2025

"""

import xobjects as xo
import numpy as np

from ..base_element import BeamElement
from ..general import _pkg_root


class LongitudinalExciter(BeamElement):
    """Beam element modeling a longitudinal exciter as a time-dependent voltage source. 

    The given voltage is scaled according to a custom waveform,
    allowing for arbitrary time dependence. The waveform is specified by an array of samples:

        voltage(t) = voltage * samples[i]

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
        - This is similar to an Exciter but applies longitudinal kicks instead of transverse kicks.
        - The voltage is applied as an energy change to the particles, similar to a Cavity.

    Parameters:
        - voltage (float): Base voltage in Volts that will be scaled by the waveform.
        - samples (float array): Samples of excitation strength to scale voltage as function of time.
        - nsamples (int): Number of samples. Pass this instead of samples to reserve memory for later initialisation.
        - sampling_frequency (float): Sampling frequency in Hz.
        - frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
        - start_turn (int): Turn of the reference particle when to start excitation.
        - duration (float): Duration of excitation in s (defaults to nsamples/sampling_frequency). Repeats the waveform to fill the duration.

    Example:
        >>> fs = 10e6 # sampling frequency in Hz
        >>> 
        >>> # A simple sine at 500 kHz ...
        >>> t = np.arange(1000)/fs
        >>> f_ex = 5e5 # excitation frequency in Hz
        >>> signal = np.sin(2*np.pi*f_ex*t)
        >>> 
        >>> # create the longitudinal exciter
        >>> frev = 1e6 # revolution frequency in Hz
        >>> voltage = 1000 # this is scaled by the waveform
        >>> long_exciter = LongitudinalExciter(samples=signal, sampling_frequency=fs, frev=frev, start_turn=0, voltage=voltage)
        >>> 
        >>> # add it to the line
        >>> line.insert_element(index=..., name=..., element=long_exciter)

    """

    _xofields={
        'voltage': xo.Float64,
        'nsamples': xo.Int64,
        'sampling_frequency': xo.Float64,
        'frev': xo.Float64,
        'start_turn': xo.Int64,
        'nduration': xo.Int64,
        'samples': xo.Float32[:],
        }

    has_backtrack = True

    _extra_c_sources = ['#include <beam_elements/elements_src/longitudinal_exciter.h>']


    def __init__(self, *, samples=None, nsamples=None, sampling_frequency=0., frev=0., voltage=0., start_turn=0, duration=None, _xobject=None, **kwargs):

        if _xobject is not None:
            super().__init__(_xobject=_xobject)

        else:
            # determine sample length or create empty samples for later initialisation
            if samples is not None:
                if nsamples is not None and nsamples != len(samples):
                    raise ValueError("Only one of samples or nsamples may be specified")
                nsamples = len(samples)
            if samples is None:
                samples = np.zeros(nsamples)
            kwargs["nduration"] = nsamples if duration is None else (duration * sampling_frequency)

            super().__init__(
                voltage=voltage, samples=samples, nsamples=nsamples,
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
        return self.__class__(voltage=-ctx2np(self.voltage),
                              samples=self.samples,
                              nsamples=self.nsamples,
                              sampling_frequency=self.sampling_frequency,
                              frev=self.frev,
                              start_turn=self.start_turn,
                              _context=_context, _buffer=_buffer, _offset=_offset)