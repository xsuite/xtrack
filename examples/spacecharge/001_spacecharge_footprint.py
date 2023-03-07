# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import json
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt
import xfields as xf

############
# Settings #
############

fname_line = ('../../test_data/sps_w_spacecharge/'
                  'line_no_spacecharge_and_particle.json')

# Realistic settings (feasible only on GPU)
bunch_intensity = 1e11/3 # Need short bunch to avoid bucket non-linearity
sigma_z = 22.5e-2/3
nemitt_x=2.5e-6
nemitt_y=2.5e-6
n_part=int(1e6)
num_turns=32
nz_grid = 100
z_range = (-3*sigma_z, 3*sigma_z)

num_spacecharge_interactions = 540
tol_spacecharge_position = 1e-2

mode = 'frozen'
#mode = 'quasi-frozen'
mode = 'pic'

#context = xo.ContextCpu()
context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

print(context)

############
# Get line #
############

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line= xt.Line.from_dict(input_data['line'])
particle_ref = xp.Particles.from_dict(input_data['particle'])

#############################################
# Install spacecharge interactions (frozen) #
#############################################

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

xf.install_spacecharge_frozen(line=line,
                   particle_ref=particle_ref,
                   longitudinal_profile=lprofile,
                   nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                   sigma_z=sigma_z,
                   num_spacecharge_interactions=num_spacecharge_interactions,
                   tol_spacecharge_position=tol_spacecharge_position)


##########################
# Configure space-charge #
##########################

if mode == 'frozen':
    pass # Already configured in line
elif mode == 'quasi-frozen':
    xf.replace_spacecharge_with_quasi_frozen(
                                    line, _buffer=context.new_buffer(),
                                    update_mean_x_on_track=True,
                                    update_mean_y_on_track=True)
elif mode == 'pic':
    pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
        _context=context, line=line,
        n_sigmas_range_pic_x=8,
        n_sigmas_range_pic_y=8,
        nx_grid=256, ny_grid=256, nz_grid=nz_grid,
        n_lims_x=7, n_lims_y=3,
        z_range=z_range)
else:
    raise ValueError(f'Invalid mode: {mode}')

##################
# Build trackers #
##################

line.build_tracker(_context=context)
line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')

######################
# Generate particles #
######################

import footprint
r_max_sigma = 5
N_r_footprint = 10
N_theta_footprint = 8
theta_min = 0.05 * np.pi / 2
theta_max = np.pi / 2 - 0.05 * np.pi / 2

x_norm_fp, y_norm_fp, r_footprint, theta_footprint = xp.generate_2D_polar_grid(
        r_range=(0.3, r_max_sigma),
        nr=N_r_footprint+1,
        theta_range=(theta_min, theta_max), ntheta=N_theta_footprint)
N_footprint = len(x_norm_fp)

particles_fp = line.build_particles(
            particle_ref=particle_ref,
            weight=0, # pure probe particles
            zeta=0, delta=0,
            x_norm=x_norm_fp, px_norm=0,
            y_norm=y_norm_fp, py_norm=0,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

# I add explicitly a probe particle at1.5 sigma
particle_probe = line.build_particles(
            particle_ref=particle_ref,
            weight=0, # pure probe particles
            zeta=0, delta=0,
            x_norm=1.5, px_norm=0,
            y_norm=1.5, py_norm=0,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

particles_gaussian = xp.generate_matched_gaussian_bunch(_context=context,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_ref=particle_ref, line=line_sc_off)

particles = xp.Particles.merge(
                          [particles_fp, particle_probe, particles_gaussian])

particles_0 = particles.copy()

#########
# Track #
#########

ctx2arr = context.nparray_from_context_array
x_tbt = np.zeros((N_footprint, num_turns), dtype=np.float64)
y_tbt = np.zeros((N_footprint, num_turns), dtype=np.float64)
for ii in range(num_turns):
    print(f'Turn: {ii}', end='\r', flush=True)
    x_tbt[:, ii] = ctx2arr(particles.x[:N_footprint]).copy()
    y_tbt[:, ii] = ctx2arr(particles.y[:N_footprint]).copy()
    line.track(particles)

tw = line_sc_off.twiss(particle_ref=particle_ref, at_elements=[0])

######################
# Frequency analysis #
######################

import NAFFlib

xy_norm = np.zeros((N_r_footprint + 1, N_theta_footprint, 2), dtype=np.float64)
xy_norm[:, :, 0] = x_norm_fp.reshape((N_r_footprint + 1, N_theta_footprint))
xy_norm[:, :, 1] = y_norm_fp.reshape((N_r_footprint + 1, N_theta_footprint))

Qx = np.zeros(N_footprint)
Qy = np.zeros(N_footprint)

for i_part in range(N_footprint):
    Qx[i_part] = NAFFlib.get_tune(x_tbt[i_part, :])
    Qy[i_part] = NAFFlib.get_tune(y_tbt[i_part, :])

Qxy_fp = np.zeros_like(xy_norm)

Qxy_fp[:, :, 0] = np.reshape(Qx, Qxy_fp[:, :, 0].shape)
Qxy_fp[:, :, 1] = np.reshape(Qy, Qxy_fp[:, :, 1].shape)


###############################
# Tune shift from single turn #
###############################

p_probe_before = particles_0.filter(
        particles_0.particle_id == N_footprint).to_dict()

line.track(particles_0)

p_probe_after = particles_0.filter(
        particles_0.particle_id == N_footprint).to_dict()

betx = tw['betx'][0]
alfx = tw['alfx'][0]
phasex_0 = np.angle(p_probe_before['x'] / np.sqrt(betx) -
                   1j*(p_probe_before['x'] * alfx / np.sqrt(betx) +
                       p_probe_before['px'] * np.sqrt(betx)))[0]
phasex_1 = np.angle(p_probe_after['x'] / np.sqrt(betx) -
                   1j*(p_probe_after['x'] * alfx / np.sqrt(betx) +
                       p_probe_after['px'] * np.sqrt(betx)))[0]
bety = tw['bety'][0]
alfy = tw['alfy'][0]
phasey_0 = np.angle(p_probe_before['y'] / np.sqrt(bety) -
                   1j*(p_probe_before['y'] * alfy / np.sqrt(bety) +
                       p_probe_before['py'] * np.sqrt(bety)))[0]
phasey_1 = np.angle(p_probe_after['y'] / np.sqrt(bety) -
                   1j*(p_probe_after['y'] * alfy / np.sqrt(bety) +
                       p_probe_after['py'] * np.sqrt(bety)))[0]
qx_probe = (phasex_1 - phasex_0)/(2*np.pi)
qy_probe = (phasey_1 - phasey_0)/(2*np.pi)

#########
# Plots #
#########

import matplotlib.pyplot as plt
plt.close('all')

fig3 = plt.figure(3)
axcoord = fig3.add_subplot(1, 1, 1)
footprint.draw_footprint(xy_norm, axis_object=axcoord, linewidth = 1)
axcoord.set_xlim(right=np.max(xy_norm[:, :, 0]))
axcoord.set_ylim(top=np.max(xy_norm[:, :, 1]))

fig4 = plt.figure(4)
axFP = fig4.add_subplot(1, 1, 1)
footprint.draw_footprint(Qxy_fp, axis_object=axFP, linewidth = 1)
axFP.plot(qx_probe, qy_probe, 'x')
axFP.set_xlim(.15-.07, .15)
axFP.set_ylim(.25-.07, .25)
axFP.set_aspect('equal')
fig4.suptitle(mode)
plt.show()
