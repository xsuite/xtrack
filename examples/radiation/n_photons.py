'''
Adapted from P. Dijkstal, 2016, CERN
https://github.com/PyCOMPLETE/CellStudyInputPyECLOUD/

Based on A. Hofmann, The physics of synchrotron radiation
'''

from __future__ import division
import numpy as np
import scipy.special as special
import scipy.integrate as integrate
from scipy.constants import e, c, m_e, epsilon_0, h, hbar


def accept_arrays(func):
    def new_func(first_arg, *args, **kwargs):
        if hasattr(first_arg,'__len__'):
            return np.array([func(arg, *args, **kwargs) for arg in first_arg])
        else:
            return func(first_arg, *args, **kwargs)
    return new_func

def bessel_5_3(z):
    return special.kv(5./3., z)

@accept_arrays
def s_s(x):
    # Implements A. Hofmann, Eq. 5.17c
    factor = 9.*np.sqrt(3.) / (8*np.pi)
    upper_limit = np.inf
    integral, abs_error = integrate.quad(bessel_5_3, x, upper_limit)
    return factor * x * integral

def compute_power_single(gamma, rho):
    return e**2 / (4*np.pi*epsilon_0) * 2*c*gamma**4 / (3*rho**2)

def compute_critical_energy_eV(gamma, rho):
    return 3*c*h*gamma**3 / (4*np.pi*rho) / e

@accept_arrays
def integral(x_min, verbose=False):
    upper_limit = np.inf
    part_1, err_1 = integrate.quad(lambda x: bessel_5_3(x)*(x-x_min), x_min, upper_limit)
    if verbose:
        print('Absolute error for n_photons integral %.2e is %.2e' % (part_1, err_1))
    return part_1, err_1

@accept_arrays
def photon_spectrum(energy, gamma, rho):
    # Implements Hofmann, Eq. 5.48b
    ps = compute_power_single(gamma, rho)
    ec = compute_critical_energy_eV(gamma, rho)
    return ps / e /ec  / energy * s_s(energy/ec)

