from __future__ import division
import numpy as np
import scipy.special as special
import scipy.integrate as integrate
from scipy.constants import e, c, m_e, epsilon_0, h, hbar

copper_work_function_eV = 4.6 # R. Cimino paper
lhc_bending_radius = 2803.95 # design report

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
    factor = 9.*np.sqrt(3.) / (8*np.pi)
    upper_limit = np.inf
    integral, abs_error = integrate.quad(bessel_5_3, x, upper_limit)
    return factor * x * integral

def compute_gamma(energy_eV):
    return energy_eV*e/(m_e*c**2)

def compute_power_single(gamma, rho):
    return e**2 / (4*np.pi*epsilon_0) * 2*c*gamma**4 / (3*rho**2)

def compute_critical_energy_eV(gamma, rho):
    return 3*c*h*gamma**3 / (4*np.pi*rho) / e

def compute_critical_angle(beam_energy, rho, energy_eV):
    gamma = compute_gamma(beam_energy)
    critical_energy_eV = compute_critical_energy_eV(gamma, rho)
    return 1./gamma * (2*critical_energy_eV/energy_eV)**(1./3.)

@accept_arrays
def n_photons_meter(energy_eV, rho=lhc_bending_radius, e_min_eV=copper_work_function_eV):
    gamma = compute_gamma(energy_eV)
    critical_energy_eV = compute_critical_energy_eV(gamma, rho)
    factor2 = np.sqrt(3)/(8*np.pi**2*hbar) * (e**2*gamma)/(epsilon_0*rho*c)
    x_min = e_min_eV / critical_energy_eV
    output, abs_error = integral(x_min)
    return factor2 * output

@accept_arrays
def integral(x_min, verbose=False):
    upper_limit = np.inf
    part_1, err_1 = integrate.quad(lambda x: bessel_5_3(x)*(x-x_min), x_min, upper_limit)
    if verbose:
        print('Absolute error for n_photons integral %.2e is %.2e' % (part_1, err_1))
    return part_1, err_1

@accept_arrays
def spectral_at_energy(energy, energy_eV, rho):
    gamma = compute_gamma(energy_eV)
    ps = compute_power_single(gamma, rho)
    ec = compute_critical_energy_eV(gamma, rho)
    return ps /ec  / energy * s_s(energy/ec)

