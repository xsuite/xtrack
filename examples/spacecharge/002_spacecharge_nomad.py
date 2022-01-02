import pathlib
import json
import numpy as np
import pandas as pd

import xobjects as xo
import xpart as xp
import xtrack as xt
import xfields as xf

from xtrack.line import _is_thick, _is_drift

fname_line = ('../../test_data/sps_w_spacecharge/'
                  'line_no_spacecharge_and_particle.json')

# Realistic settings (feasible only on GPU)
bunch_intensity = 1e11/3 # Need short bunch to avoid bucket non-linearity
sigma_z = 22.5e-2/3
neps_x=2.5e-6
neps_y=2.5e-6
n_part=int(1e6)
num_turns=32
nz_grid = 100
z_range = (-3*sigma_z, 3*sigma_z)

num_spacecharge_interactions = 540
tol_spacecharge_position = 1e-2

##############
# Get a line #
##############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line_no_sc = xt.Line.from_dict(input_data['line'])
particle_ref = xp.Particles.from_dict(input_data['particle'])

# Make a matched bunch just to get the matched momentum spread
bunch = xp.generate_matched_gaussian_bunch(
         num_particles=int(2e6), total_intensity_particles=bunch_intensity,
         nemitt_x=neps_x, nemitt_y=neps_y, sigma_z=sigma_z,
         particle_ref=particle_ref, tracker=xt.Tracker(line=line_no_sc))
delta_rms = np.std(bunch.delta)

# Define longitudinal profile
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

# Remove all drifts
s_no_drifts = []
e_no_drifts = []
n_no_drifts = []
for ss, ee, nn in zip(line_no_sc.get_s_elements(), line_no_sc.elements,
                      line_no_sc.element_names):
    if not _is_drift(ee):
        assert not _is_thick(ee)
        s_no_drifts.append(ss)
        e_no_drifts.append(ee)
        n_no_drifts.append(nn)

s_no_drifts = np.array(s_no_drifts)

# Generate spacecharge positions
s_spacecharge = np.linspace(0, line_no_sc.get_length(),
                            num_spacecharge_interactions+1)[:-1]

# Adjust spacecharge positions where possible
for ii, ss in enumerate(s_spacecharge):
    s_closest = np.argmin(np.abs(ss-s_no_drifts))
    if np.abs(ss - s_closest) < tol_spacecharge_position:
        s_spacecharge[ii] = s_closest

# Create spacecharge elements (dummy for now)
sc_elements = []
sc_names = []
for ii, _ in enumerate(s_spacecharge):
    sc_elements.append(xf.SpaceChargeBiGaussian(
        length=0,
        apply_z_kick=False,
        longitudinal_profile=lprofile,
        mean_x=0.,
        mean_y=0.,
        sigma_x=1.,
        sigma_y=1.))
    sc_names.append(f'spacecharge_{ii}')

df_lattice = pd.DataFrame({'s': s_no_drifts, 'elements': e_no_drifts,
                           'element_names': n_no_drifts})
df_spacecharge = pd.DataFrame({'s': s_spacecharge, 'elements': sc_elements,
                               'element_names': sc_names})
df_elements = pd.concat([df_lattice, df_spacecharge]).sort_values('s')

new_elements = []
new_names = []

s_curr = 0
i_drift = 0
for ss, ee, nn, in zip(df_elements['s'].values,
                       df_elements['elements'].values,
                       df_elements['element_names'].values):

    if ss > s_curr + 1e-10:
        new_elements.append(xt.Drift(length=(ss-s_curr)))
        new_names.append(f'drift_{i_drift}')
        s_curr = ss
        i_drift += 1
    new_elements.append(ee)
    new_names.append(nn)

if s_curr < line_no_sc.get_length():
    new_elements.append(xt.Drift(length=line_no_sc.get_length() - s_curr))
    new_names.append(f'drift_{i_drift}')

line = xt.Line(elements=new_elements, element_names=new_names)

assert np.isclose(line.get_length(), line_no_sc.get_length(), rtol=0, atol=1e-10)
