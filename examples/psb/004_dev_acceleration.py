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

# Stretch it to enhance change in revolution frequency
E_min = np.min(E_kin_GeV)
E_max = np.max(E_kin_GeV)
E_kin_GeV = E_min/100 + (E_kin_GeV - E_min)

t_s = df.t_s.values

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line.build_tracker()

mass0_eV = line.particle_ref.mass0

e_tot_ev = E_kin_GeV*1e9 + mass0_eV
gamma = e_tot_ev/mass0_eV
beta = np.sqrt(1 - 1/gamma**2)

beta_mid = 0.5*(beta[1:] + beta[:-1])
L = line.get_length()

dt_s = np.diff(t_s)

i_turn = np.zeros_like(e_tot_ev)
i_turn[1:] = np.cumsum(beta_mid * clight / L * dt_s)

class EnergyProgram:

    def __init__(self, t_s, circumference, mass0, kinetic_energy0=None, p0c=None):

        assert hasattr (t_s, '__len__'), 't_s must be a list or an array'

        assert p0c is not None or kinetic_energy0 is not None, (
            'Either p0c or kinetic_energy0 needs to be provided')

        enevars = {}

        if p0c is not None:
            assert hasattr (p0c, '__len__'), 'p0c must be a list or an array'
            assert len(t_s) == len(p0c), 't_s and p0c must have same length'
            enevars['p0c'] = p0c

        if kinetic_energy0 is not None:
            assert hasattr (kinetic_energy0, '__len__'), (
                'kinetic_energy0 must be a list or an array')
            assert len(t_s) == len(kinetic_energy0), (
                't_s and kinetic_energy0 must have same length')

            energy0 = kinetic_energy0 + mass0
            enevars['energy0'] = energy0

        # I use a particle to make the conversions
        p = xt.Particles(**enevars, mass0=mass0)
        beta0_program = p.beta0
        bet0_mid = 0.5*(beta0_program[1:] + beta0_program[:-1])

        dt_s = np.diff(t_s)

        i_turn_at_t_samples = np.zeros_like(t_s)
        i_turn_at_t_samples[1:] = np.cumsum(
                                    bet0_mid * clight / circumference * dt_s)

        self.t_s = t_s
        self.t_at_turn_interpolator = xd.FunctionPieceWiseLinear(
                                x=i_turn_at_t_samples, y=t_s)
        self.p0c_interpolator = xd.FunctionPieceWiseLinear(
                                x=t_s, y=np.array(p.p0c))
    def get_t_s_at_turn(self, i_turn):
        return self.t_at_turn_interpolator(i_turn)

    def get_p0c_at_t_s(self, t_s):
        return self.p0c_interpolator(t_s)

ep = EnergyProgram(t_s=t_s, kinetic_energy0=E_kin_GeV*1e9, mass0=mass0_eV,
                   circumference=line.get_length())


p_test = line.build_particles(x=0)

t_test = 40e-3
p0c_test = ep.get_p0c_at_t_s(t_test)
p_test.update_p0c_and_energy_deviations(p0c_test)
ekin_test = p_test.energy0[0] - p_test.mass0

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
sp_ekin = plt.subplot(3,1,1)
plt.plot(t_s, E_kin_GeV)
plt.plot(t_test, ekin_test*1e-9, 'o')
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

plt.figure(2)
plt.plot(t_s, i_turn)

i_turn_test = 1000
t_test = ep.get_t_s_at_turn(i_turn_test)
plt.plot(t_test, i_turn_test, 'o')

plt.show()
