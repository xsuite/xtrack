import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import epsilon_0

import xpart as xp
import xtrack as xt
import xobjects as xo

context = xo.ContextCpu()

theta_bend = 0.05
L_bend = 5.

dipole_ave = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                          radiation_flag=1, _context=context)
dipole_rnd = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                          radiation_flag=2, _context=context)

particles_ave = xp.Particles(
        _context=context,
        p0c=5e9, # 5 GeV
        x=np.zeros(1000000),
        px=1e-4,
        py=-1e-4,
        mass0=xp.ELECTRON_MASS_EV)
particles_rnd = particles_ave.copy()

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

rho = L_bend/theta_bend
mass0_kg = (dct_ave['mass0']*qe/clight**2)
r0 = qe**2/(4*np.pi*epsilon_0*mass0_kg*clight**2)
Ps = (2*r0*clight*mass0_kg*clight**2*
      dct_ave['beta0'][0]**4*dct_ave['gamma0'][0]**4)/(3*rho**2) # W

Delta_E_eV = -Ps*(L_bend/clight) / qe
Delta_E_trk = (dct_ave['ptau']-dct_ave_before['ptau'])*dct_ave['p0c']

assert np.allclose(Delta_E_eV, Delta_E_trk, atol=0, rtol=1e-6)

