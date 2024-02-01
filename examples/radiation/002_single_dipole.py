# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import epsilon_0, hbar

import xtrack as xt
import xobjects as xo

context = xo.ContextCpu()

L_bend = 1.
B_T = 2

delta = 0
particles_ave = xt.Particles(
        _context=context,
        p0c=5e9 / (1 + delta), # 5 GeV
        x=np.zeros(1000000),
        px=1e-4,
        py=-1e-4,
        delta=delta,
        mass0=xt.ELECTRON_MASS_EV)
particles_ave_0 = particles_ave.copy()
gamma = (particles_ave.energy/particles_ave.mass0)[0]
gamma0 = (particles_ave.gamma0[0])
particles_rnd = particles_ave.copy()

P0_J = particles_ave.p0c[0] / clight * qe
h_bend = B_T * qe / P0_J
theta_bend = h_bend * L_bend

dipole_ave = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                          radiation_flag=1, _context=context)
dipole_rnd = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                          radiation_flag=2, _context=context)


dct_ave_before = particles_ave.to_dict()
dct_rng_before = particles_rnd.to_dict()

particles_ave._init_random_number_generator()
particles_rnd._init_random_number_generator()

dipole_ave.track(particles_ave)
dipole_rnd.track(particles_rnd)

dct_ave = particles_ave.to_dict()
dct_rng = particles_rnd.to_dict()

assert np.allclose(dct_ave['delta'], np.mean(dct_rng['delta']),
                  atol=0, rtol=5e-3)

rho_0 = L_bend/theta_bend
mass0_kg = (dct_ave['mass0']*qe/clight**2)
r0 = qe**2/(4*np.pi*epsilon_0*mass0_kg*clight**2)
Ps = (2 * r0 * clight * mass0_kg * clight**2 * gamma0**2 * gamma**2)/(3*rho_0**2) # W

Delta_E_eV = -Ps*(L_bend/clight) / qe
Delta_E_trk = (dct_ave['ptau']-dct_ave_before['ptau'])*dct_ave['p0c']

assert np.allclose(Delta_E_eV, Delta_E_trk, atol=0, rtol=4e-5)

# Check photons
line=xt.Line(elements=[
             xt.Drift(length=1.0),
             xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend),
             xt.Drift(length=1.0),
             xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend)
            ])
line.build_tracker(_context=context)
line.configure_radiation(model='quantum')

record_capacity = int(100e6)
record = line.start_internal_logging_for_elements_of_type(xt.Multipole,
                                                            capacity=record_capacity)
particles_test = particles_ave_0.copy()
particles_test_before = particles_test.copy()
line.track(particles_test)

Delta_E_test = (particles_test.ptau - particles_test_before.ptau
                                                      )*particles_test.p0c
n_recorded = record._index.num_recorded
assert n_recorded < record_capacity
assert np.allclose(-np.sum(Delta_E_test), np.sum(record.photon_energy[:n_recorded]),
                  atol=0, rtol=1e-6)

p0_J = particles_ave.p0c[0] / clight * qe
B_T = p0_J / qe / rho_0
mass_0_kg = particles_ave.mass0 * qe / clight**2
E_crit_J = 3 * qe * hbar * gamma**2 * B_T / (2 * mass_0_kg)

E_ave_J = 8 * np.sqrt(3) / 45 * E_crit_J
E_ave_eV = E_ave_J / qe

E_sq_ave_J = 11 / 27 * E_crit_J**2
E_sq_ave_eV = E_sq_ave_J / qe**2

assert np.isclose(np.mean(record.photon_energy[:n_recorded]), E_ave_eV, rtol=1e-2, atol=0)
assert np.isclose(np.std(record.photon_energy[:n_recorded]), np.sqrt(E_sq_ave_eV - E_ave_eV**2), rtol=1e-3, atol=0)