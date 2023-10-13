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
t_s = t_s

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
gamma_at_turn = 1 / np.sqrt(1 - beta_at_turn**2)

t_turn_ref = np.cumsum(line.get_length()/clight/beta_at_turn)
t_turn_ref = t_turn_ref - t_turn_ref[0]
E_kin_turn = line.particle_ref.mass0 * (monitor.gamma0[0, :] - 1)

t_check = np.linspace(0, 20e-3, 1000)
E_check = np.interp(t_check, t_turn_ref, E_kin_turn)
E_check_ref = np.interp(t_check, t_s, E_kin_GeV*1e9)
assert np.allclose(E_check, E_check_ref, atol=0, rtol=2e-3)

t_turn_check = line.energy_program.get_t_s_at_turn(np.arange(n_turn_test))
assert np.allclose(t_turn_check, t_turn_ref, atol=0, rtol=6e-4)

p0c_check = line.energy_program.get_p0c_at_t_s(t_check)
p0c_ref = np.interp(t_check,
    t_turn_check,
    line.particle_ref.mass0 * gamma_at_turn * beta_at_turn)
assert np.allclose(p0c_check, p0c_ref, atol=0, rtol=1e-3)

beta0_check = line.energy_program.get_beta0_at_t_s(t_check)
beta0_ref = np.interp(t_check, t_turn_check, beta_at_turn)
assert np.allclose(beta0_check, beta0_ref, atol=0, rtol=1e-3)

frev_check = line.energy_program.get_frev_at_t_s(t_check)
frev_ref = np.interp(t_check, t_turn_check[:-1], 1/np.diff(t_turn_ref))
assert np.allclose(frev_check, frev_ref, atol=0, rtol=4e-5)

p0c_increse_per_turn_check = line.energy_program.get_p0c_increse_per_turn_at_t_s(t_check)
p0c_increse_per_turn_ref = np.interp(t_check, t_turn_check[:-1], np.diff(monitor.p0c[0, :]))
assert np.allclose(p0c_increse_per_turn_check - p0c_increse_per_turn_ref, 0,
                   atol= 5e-5 * p0c_ref[0], rtol=0)

line.enable_time_dependent_vars = False
line.vars['t_turn_s'] = 20e-3

E_kin_expected = np.interp(line.vv['t_turn_s'], t_s, E_kin_GeV*1e9)
E_tot_expected = E_kin_expected + line.particle_ref.mass0
assert np.isclose(
    E_tot_expected, line.particle_ref.energy0[0], rtol=1e-4, atol=0)

tw = line.twiss(method='6d')
assert np.isclose(tw.zeta[0], -13.48, rtol=0, atol=1e-4) # To check that it does not change
assert np.isclose(line.particle_ref.mass0 * tw.gamma0, E_tot_expected,
                  atol=0, rtol=1e-12)

line.vars['t_turn_s'] = 0
line.vars['on_chicane_k0'] = 0
tw = line.twiss(method='6d')
assert np.allclose(tw.zeta[0], 0, rtol=0, atol=1e-12)
assert np.allclose(line.particle_ref.mass0 * tw.gamma0, line.particle_ref.mass0 + E_kin_turn[0],
                   rtol=1e-10, atol=0)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(beta_at_turn, '.')

plt.figure()
plt.plot(t_turn_ref, E_kin_turn)
plt.plot(t_s, E_kin_GeV*1e9)

plt.figure()
plt.plot(t_turn_ref)

plt.show()
