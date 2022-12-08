"""
Exciter element

Author: Philipp Niedermayer
Date: 11.11.2022

"""


from pathlib import Path

import xobjects as xo
import xtrack as xt
import numpy as np


class Exciter(xt.BeamElement):
    """Beam element modeling a transverse exciter as a time-dependent thin multipole.
    
    The given multipole components (knl and ksl) are scaled according to an array of
    samples, allowing for arbitrary time dependence:
        
        knl(t) = knl * samples[i]

    It is *not* assumed that the variations are slow compared to the revolution frequency
    and the particle arrival time is taken into account when determining the sample index:
    
        i = sampling * ( ( at_turn - start_turn ) / f_rev - zeta / beta0 / c0 )
    
    where zeta=(s-beta0*c0*t) is the longitudinal coordinate of the particle, beta0 the
    relativistic beta factor of the particle, c0 is the speed of light, at_turn is the
    current turn number, f_rev is the revolution frequency, and sampling is the sampling
    frequency. The excitation starts with the first sample when the reference particle 
    arrives at the element in start_turn.

    For example, to compute samples for a sinusoidal excitation with frequency f_ex one
    would do: samples[i] = np.sin(2*np.pi*f_ex*i/sampling).    
    
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
        - samples (float array): Samples of excitation strength to scale knl and ksl as function of time.
        - sampling (float): Sampling frequency in Hz.
        - frev (float): Revolution frequency in Hz of circulating beam (used to relate turn number to sample index).
        - start_turn (int): Start turn of excitation.
        - nsamples (int): Number of samples. Pass this instead of samples to reserve memory for later initialisation.
        - order (int): Multipole order (readonly), i.e. largest n with non-zero knl or ksl.
        
    """

    _xofields={
        'order': xo.Int64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'samples': xo.Float64[:],
        'nsamples': xo.Int64,
        'sampling': xo.Float64,
        'frev': xo.Float64,
        'start_turn': xo.Int64,
    }

    _extra_c_sources = [_pkg_root.joinpath('beam_elements/elements_src/exciter.h')]


    def __init__(self, *, samples=None, nsamples=None, sampling=0, frev=0, knl=[1], ksl=[], start_turn=0, **kwargs):        
        # sanitize knl and ksl array length
        n = max(len(knl), len(ksl))
        nknl = np.zeros(n, dtype=np.float64)
        nksl = np.zeros(n, dtype=np.float64)
        if knl is not None:
            nknl[:len(knl)] = np.array(knl)
        if ksl is not None:
            nksl[:len(ksl)] = np.array(ksl)
        order = n - 1

        # determine sample length or create empty samples for later initialisation
        if samples is not None:
            if nsamples is not None and nsamples != len(samples):
                raise ValueError("Only one of samples or nsamples may be specified")
            nsamples = len(samples)
        if samples is None:
            samples = np.zeros(nsamples)

        super().__init__(order=order, knl=nknl, ksl=nksl, samples=samples, 
                         nsamples=nsamples, sampling=sampling, frev=frev,
                         start_turn=start_turn, **kwargs)

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(knl=-ctx2np(self.knl),
                              ksl=-ctx2np(self.ksl),
                              samples=self.samples,
                              nsamples=self.nsamples,
                              sampling=self.sampling,
                              frev=self.frev,
                              start_turn=self.start_turn,
                              _context=_context, _buffer=_buffer, _offset=_offset)
