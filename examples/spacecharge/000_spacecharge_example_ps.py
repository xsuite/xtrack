# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt
import xfields as xf

############
# Settings #
############

# Load PS line from test data
fname_line = ('../../test_data/ps_ion/PS_2022_Pb_nominal.json')

# Nominal Pb ion parameters
bunch_intensity = 6.0e8
sigma_z = 6.0
nemitt_x=0.75e-6
nemitt_y=0.5e-6
n_part=int(1e6)
num_turns=32

num_spacecharge_interactions = 540

# Available modes: frozen/quasi-frozen/pic
# mode = 'pic'
#mode = 'quasi-frozen'
mode = 'frozen'

# Choose solver between `FFTSolver2p5DAveraged` and `FFTSolver2p5D`
pic_solver = 'FFTSolver2p5DAveraged'

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

print(context)

#############
# Load line #
#############

line = xt.Line.from_json(fname_line)
print(line.particle_ref)
#line.particle_ref = xp.Particles.from_dict(input_data['particle'])


line.build_tracker(_context=context, compile=False)

#############################################
# Install spacecharge interactions (frozen) #
#############################################

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

xf.install_spacecharge_frozen(line=line,
                   longitudinal_profile=lprofile,
                   nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                   sigma_z=sigma_z,
                   num_spacecharge_interactions=num_spacecharge_interactions,
                   )

#################################
# Switch to PIC or quasi-frozen #
#################################

if mode == 'frozen':
    pass # Already configured in line
elif mode == 'quasi-frozen':
    xf.replace_spacecharge_with_quasi_frozen(
                                    line,
                                    update_mean_x_on_track=True,
                                    update_mean_y_on_track=True)
elif mode == 'pic':
    pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
        line=line,
        n_sigmas_range_pic_x=8,
        n_sigmas_range_pic_y=8,
        nx_grid=256, ny_grid=256, nz_grid=100,
        n_lims_x=7, n_lims_y=3,
        z_range=(-3*sigma_z, 3*sigma_z),
        solver=pic_solver)
else:
    raise ValueError(f'Invalid mode: {mode}')

#################
# Build Tracker #
#################

line.build_tracker(_context=context)

######################
# Generate particles #
######################

# Build a line without spacecharge (recycling the track kernel)
line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')
line_sc_off.build_tracker(track_kernel=line.tracker.track_kernel)

# (we choose to match the distribution without accounting for spacecharge)
particles = xp.generate_matched_gaussian_bunch(line=line_sc_off,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z)

####################################################
# Check that everything is on the selected context #
####################################################

assert particles._context == context
assert len(set([ee._buffer for ee in line.elements])) == 1 # All on same context
assert line._context == context

#########
# Track #
#########
line.track(particles, num_turns=num_turns, with_progress=1)

