import pandas as pd
import numpy as np

from scipy.constants import c as clight

import xtrack as xt
import xdeps as xd

# REMEMBER:
# - Handle zero ramp rate

fname = 'RF_DoubleHarm.dat'

df = pd.read_csv(fname, sep='\t', skiprows=2,
    names=['t_s', 'E_kin_GeV', 'V1_MV', 'phi1_rad', 'V2_MV', 'phi2_rad'])
E_kin_GeV = df.E_kin_GeV.values
t_s = df.t_s.values

# Shift to enhance change in revolution frequency
E_min = np.min(E_kin_GeV)
E_max = np.max(E_kin_GeV)
E_kin_GeV = E_min/100 + (E_kin_GeV - E_min)
# Shift the time scale for testing purposes
t_s = t_s + 5e-3

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line.build_tracker()

# Attach energy program
ep = xt.EnergyProgram(t_s=t_s, kinetic_energy0=E_kin_GeV*1e9)
line.energy_program = ep

line['br1.acwf7l1.1'].voltage = 2e6
line['br1.acwf7l1.1'].frequency = 1e3


p_test = line.build_particles()
line.enable_time_dependent_vars = True
n_turn_test = 5000
monitor = xt.ParticlesMonitor(num_particles=len(p_test.zeta), start_at_turn=0,
                              stop_at_turn=n_turn_test)
for ii in range(n_turn_test):
    if ii % 10 == 0:
        print(f'Tracking turn {ii}/{n_turn_test}     ', end='\r', flush=True)
    line.track(p_test, turn_by_turn_monitor=monitor)

beta_at_turn = monitor.beta0[0, :]

t_turn_ref = np.cumsum(line.get_length()/clight/beta_at_turn)
E_kin_turn = line.particle_ref.mass0 * (monitor.gamma0[0, :] - 1)

t_check = np.linspace(0, 20e-3, 1000)
E_check = np.interp(t_check, t_turn_ref, E_kin_turn)
E_check_ref = np.interp(t_check, t_s, E_kin_GeV*1e9)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(beta_at_turn, '.')

plt.figure()
plt.plot(t_turn_ref, E_kin_turn)
plt.plot(t_s, E_kin_GeV*1e9)

plt.show()
