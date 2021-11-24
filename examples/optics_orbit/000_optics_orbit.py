import pathlib
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp
from scipy.constants import c as clight


fname_line = '../../test_data/hllhc_14/line_and_particle.json'


##############
# Get a line #
##############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])

tracker = xt.Tracker(line=line)

particle_ref = xp.Particles.from_dict(input_data['particle'])
particle_co = tracker.find_closed_orbit(particle_ref)

R_matrix = tracker.compute_one_turn_matrix_finite_differences(
                   particle_on_co=particle_co)

eta = -R_matrix[4, 5]/line.get_length() # minus sign comes from z = s-ct
alpha_mom_compaction = eta + 1/particle_ref.gamma0[0]**2

T_rev = line.get_length()/(particle_ref.beta0[0]*clight)

freq_list = []
lag_list_deg = []
voltage_list = []
h_list = []
for ee in line.elements:
    if ee.__class__.__name__ == 'Cavity':
        freq_list.append(ee.frequency)
        lag_list_deg.append(ee.lag)
        voltage_list.append(ee.voltage)
        h_list.append(ee.frequency*T_rev)

zeta, delta = xp.longitudinal.generate_longitudinal_coordinates(
        distribution='gaussian',
        mass0=particle_ref.mass0,
        q0=particle_ref.q0,
        gamma0=particle_ref.gamma0,
        num_particles=10000,
        circumference=line.get_length(),
        alpha_momentum_compaction=alpha_mom_compaction,
        rf_harmonic=h_list,
        rf_voltage=voltage_list,
        rf_phase=(np.array(lag_list_deg) - 180)/180*np.pi,
        sigma_z=10e-2)
