import pandas as pd
import numpy as np

from scipy.constants import c as clight

import xtrack as xt

fname = 'RF_DoubleHarm.dat'

df = pd.read_csv(fname, sep='\t', skiprows=2,
    names=['t_s', 'E_kin_GeV', 'V1_MV', 'phi1_rad', 'V2_MV', 'phi2_rad'])
E_kin_GeV = df.E_kin_GeV.values
t_s = df.t_s.values

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
mass0_eV = line.particle_ref.mass0

e_tot_ev = df.E_kin_GeV.values*1e9 + mass0_eV
gamma = e_tot_ev/mass0_eV
beta = np.sqrt(1 - 1/gamma**2)

beta_mid = 0.5*(beta[1:] + beta[:-1])
L = line.get_length()

dt_s = np.diff(t_s)

i_turn = np.zeros_like(e_tot_ev)
i_turn[1:] = np.cumsum(beta_mid * clight / L * dt_s)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
sp_ekin = plt.subplot(3,1,1)
plt.plot(t_s, E_kin_GeV)
plt.ylabel(r'$E_{kin}$ [GeV]')

sp_dekin = plt.subplot(3,1,2, sharex=sp_ekin)
# GeV/sec
dekin = (E_kin_GeV[1:] - E_kin_GeV[:-1])/(t_s[1:] - t_s[:-1])*1e3
plt.plot(t_s[:-1], dekin)
plt.ylabel(r'd$E_{kin}$/dt [GeV/s]')

sp_beta = plt.subplot(3,1,3, sharex=sp_ekin)
plt.plot(t_s, beta)
plt.ylabel(r'$\beta$')
plt.xlabel('t [s]')


plt.show()
