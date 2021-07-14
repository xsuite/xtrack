import pathlib
import json
import numpy as np

import xobjects as xo
import xline as xl
import xtrack as xt
import xfields as xf

fname_sequence = ('../../test_data/sps_w_spacecharge/'
                  'line_with_spacecharge_and_particle.json')

number_of_particles=1e11
delta_rms=2e-3
neps_x=2.5e-6
neps_y=2.5e-6
bunchlength_rms=10e-2

####################
# Choose a context #
####################

context = xo.ContextCpu()
context = xo.ContextCupy()
context = xo.ContextPyopencl('0.0')

_buffer = context.new_buffer()

##################
# Get a sequence #
##################

with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xl.Line.from_dict(input_data['line'])

# # TEST
# for ee in sequence.elements:
#     if hasattr(ee, 'number_of_particles'):
#         ee.number_of_particles = 5e10

#################################
# Get beam sigmas at start ring #
# from first space-charge lens  #
#################################
first_sc = sequence.elements[1]
sigma_x = first_sc.sigma_x
sigma_y = first_sc.sigma_y

#################
# Build Tracker #
#################
tracker = xt.Tracker(_buffer=_buffer,
                    sequence=sequence)

####################################
# Generate particles for footprint #
####################################

import footprint
r_max_sigma = 5
N_r_footprint = 10
N_theta_footprint = 8
xy_norm = footprint.initial_xy_polar(
        r_min=1e-2, r_max=r_max_sigma,
        r_N=N_r_footprint + 1, theta_min=np.pi / 100,
        theta_max=np.pi / 2 - np.pi / 100,
        theta_N=N_theta_footprint)

particles = xt.Particles(_context=context,
        p0c=26e9,
        x=sigma_x*xy_norm[:, :, 0].flatten(),
        y=sigma_y*xy_norm[:, :, 1].flatten())

#########
# Track #
#########

tracker.track(particles, num_turns=256, turn_by_turn_monitor=True)

######################
# Frequency analysis #
######################
import NAFFlib

Qx = np.zeros(particles.num_particles)
Qy = np.zeros(particles.num_particles)

x_tbt = tracker.record_last_track.x
y_tbt = tracker.record_last_track.y

for i_part in range(particles.num_particles):
    Qx[i_part] = NAFFlib.get_tune(x_tbt[i_part, :])
    Qy[i_part] = NAFFlib.get_tune(y_tbt[i_part, :])

Qxy_fp = np.zeros_like(xy_norm)

Qxy_fp[:, :, 0] = np.reshape(Qx, Qxy_fp[:, :, 0].shape)
Qxy_fp[:, :, 1] = np.reshape(Qy, Qxy_fp[:, :, 1].shape)

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
plt.show()
