import pandas as pd
import numpy as np

from scipy.constants import c as clight

import xtrack as xt
import xdeps as xd


fname = '../../test_data/psb_chicane/Ramp_and_RF_functions.dat'

df = pd.read_csv(fname, sep='\t', skiprows=2,
    names=['t_s', 'E_kin_GeV', 'V1_MV', 'phi1_rad', 'V2_MV', 'phi2_rad'])
E_kin_GeV = df.E_kin_GeV.values
t_s = df.t_s.values

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line.build_tracker()

# Attach energy program
ep = xt.EnergyProgram(t_s=t_s, kinetic_energy0=E_kin_GeV*1e9)
line.energy_program = ep

# Frequency program
freq_rev = line.energy_program.get_frev_at_t_s(t_s)
line.functions['fun_freq_rev'] = xd.FunctionPieceWiseLinear(x=t_s, y=freq_rev)
line.vars['freq_rev'] = line.functions['fun_freq_rev'](line.vars['t_turn_s'])
line.vars['freq_h1'] = line.vars['freq_rev']
line.vars['freq_h2'] = 2 * line.vars['freq_rev']

# voltage programs

# Shift phases to have the beam centered around zero

V1_MV = df.V1_MV.values
V2_MV = df.V2_MV.values # to have opposite slopes at z=0
phi1_rad = df.phi1_rad.values - np.pi
phi2_rad = df.phi2_rad.values - np.pi

line.functions['fun_volt_mv_h1'] = xd.FunctionPieceWiseLinear(x=t_s, y=V1_MV)
line.functions['fun_volt_mv_h2'] = xd.FunctionPieceWiseLinear(x=t_s, y=V2_MV)
line.vars['volt_mv_h1'] = line.functions['fun_volt_mv_h1'](line.vars['t_turn_s'])
line.vars['volt_mv_h2'] = line.functions['fun_volt_mv_h2'](line.vars['t_turn_s'])

# phase programs
line.functions['fun_phi_rad_h1'] = xd.FunctionPieceWiseLinear(x=t_s, y=phi1_rad)
line.functions['fun_phi_rad_h2'] = xd.FunctionPieceWiseLinear(x=t_s, y=phi2_rad)
line.vars['phi_rad_h1'] = line.functions['fun_phi_rad_h1'](line.vars['t_turn_s'])
line.vars['phi_rad_h2'] = line.functions['fun_phi_rad_h2'](line.vars['t_turn_s'])

# Setup cavities
line.element_refs['br1.acwf5l1.1'].voltage = line.vars['volt_mv_h1'] * 1e6
line.element_refs['br1.acwf5l1.2'].voltage = line.vars['volt_mv_h2'] * 1e6

line.element_refs['br1.acwf5l1.1'].lag = line.vars['phi_rad_h1'] * 360 / 2 / np.pi
line.element_refs['br1.acwf5l1.2'].lag = line.vars['phi_rad_h2'] * 360 / 2 / np.pi

line.element_refs['br1.acwf5l1.1'].frequency = line.vars['freq_h1']
line.element_refs['br1.acwf5l1.2'].frequency = line.vars['freq_h2']


# tw6d = line.twiss(method='6d')

t_rev = []
beta0 = []
gamma0 = []
f_h1 = []
f_h2 = []
lag_h1 = []
lag_h2 = []
volt_h1 = []
volt_h2 = []
for ii in range(len(t_s)):
    print(f'Computing twiss at t_s = {t_s[ii]:.4} s    ', end='\r', flush=True)
    line.vars['t_turn_s'] = t_s[ii]
    tt = line.twiss(method='4d')
    t_rev.append(tt.T_rev0)
    beta0.append(tt.beta0)
    gamma0.append(tt.gamma0)
    f_h1.append(line['br1.acwf5l1.1'].frequency)
    f_h2.append(line['br1.acwf5l1.2'].frequency)
    lag_h1.append(line['br1.acwf5l1.1'].lag)
    lag_h2.append(line['br1.acwf5l1.2'].lag)
    volt_h1.append(line['br1.acwf5l1.1'].voltage)
    volt_h2.append(line['br1.acwf5l1.2'].voltage)


line.vars['t_turn_s'] = 0


# Test tracking
p_test = line.build_particles(x_norm=0, zeta=np.linspace(0, 100., 20))
assert np.isclose(p_test.energy0[0] - p_test.mass0,  E_kin_GeV[0] * 1e9,
                  atol=0, rtol=1e-10)

line.enable_time_dependent_vars = True
n_turn_test = 10000
monitor = xt.ParticlesMonitor(num_particles=len(p_test.zeta), start_at_turn=0,
                              stop_at_turn=n_turn_test)
for ii in range(n_turn_test):
    if ii % 10 == 0:
        print(f'Tracking turn {ii}/{n_turn_test}     ', end='\r', flush=True)
    line.track(p_test, turn_by_turn_monitor=monitor)


t_test = 40e-3
p0c_test = ep.get_p0c_at_t_s(t_test)
p_test.update_p0c_and_energy_deviations(p0c_test)
ekin_test = p_test.energy0[0] - p_test.mass0

t_turn = line.energy_program.get_t_s_at_turn(np.arange(n_turn_test))


import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
sp_ekin = plt.subplot(3,1,1)
plt.plot(t_s, E_kin_GeV)
plt.plot(t_test, ekin_test*1e-9, 'o')
plt.ylabel(r'$E_{kin}$ [GeV]')

sp_dekin = plt.subplot(3,1,2, sharex=sp_ekin)
# GeV/sec
dekin = (E_kin_GeV[1:] - E_kin_GeV[:-1])/(t_s[1:] - t_s[:-1])
plt.plot(t_s[:-1], dekin)
plt.ylabel(r'd$E_{kin}$/dt [GeV/s]')

sp_beta = plt.subplot(3,1,3, sharex=sp_ekin)
plt.plot(t_turn, monitor.beta0.T)
plt.ylabel(r'$\beta$')
plt.xlabel('t [s]')

# plot rf parameters
plt.figure(2)
plt.subplot(3,1,1)
plt.plot(t_s, f_h1, label='h1')
plt.plot(t_s, f_h2, label='h2')
plt.ylabel('frequency [Hz]')
plt.legend()
plt.subplot(3,1,2)
plt.plot(t_s, lag_h1, label='h1')
plt.plot(t_s, lag_h2, label='h2')
plt.ylabel('lag [deg]')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t_s, volt_h1, label='h1')
plt.plot(t_s, volt_h2, label='h2')
plt.ylabel('voltage [V]')
plt.legend()
plt.xlabel('t [s]')

plt.figure(3)
colors = plt.cm.jet(np.linspace(0,1, len(p_test.zeta)))
for ii in range(len(colors)):
    plt.plot(monitor.zeta[ii, :], monitor.delta[ii, :], color=colors[ii])
plt.xlim(-300, 300)
plt.ylim(-5e-3, 5e-3)

plt.show()
