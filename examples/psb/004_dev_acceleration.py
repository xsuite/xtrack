import pandas as pd
import numpy as np

import xtrack as xt

fname = 'RF_DoubleHarm.dat'

df = pd.read_csv(fname, sep='\t', skiprows=2,
    names=['t_ms', 'E_kin_GeV', 'V1_MV', 'phi1_rad', 'V2_MV', 'phi2_rad'])

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
mass0_eV = line.particle_ref.mass0

e_tot_ev = df.E_kin_GeV.values*1e9 + mass0_eV
gamma = e_tot_ev/mass0_eV
beta = np.sqrt(1 - 1/gamma**2)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
sp_ekin = plt.subplot(3,1,1)
plt.plot(df.t_ms, df.E_kin_GeV)
plt.ylabel(r'$E_{kin}$ [GeV]')

sp_dekin = plt.subplot(3,1,2, sharex=sp_ekin)
# GeV/sec
dekin = (df.E_kin_GeV - df.E_kin_GeV.shift(1))/(df.t_ms - df.t_ms.shift(1))*1e3
plt.plot(df.t_ms, dekin)
plt.ylabel(r'd$E_{kin}$/dt [GeV/s]')

sp_beta = plt.subplot(3,1,3, sharex=sp_ekin)
plt.plot(df.t_ms, beta)
plt.ylabel(r'$\beta$')
plt.xlabel('t [ms]')


plt.show()
