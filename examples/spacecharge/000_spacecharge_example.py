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
                  'line_with_spacecharge_and_particle.json')

fname_optics = ('../../test_data/sps_w_spacecharge/'
                'optics_and_co_at_start_ring.json')

bunch_intensity = 1e11/3
sigma_z = 22.5e-2/3
neps_x=2.5e-6
neps_y=2.5e-6
n_part=int(1e6)
rf_voltage=3e6
num_turns=32

# Available modes: frozen/quasi-frozen/pic
mode = 'pic'

####################
# Choose a context #
####################

#context = xo.ContextCpu()
context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

print(context)

########################
# Get optics and orbit #
########################

with open(fname_optics, 'r') as fid:
    ddd = json.load(fid)
part_on_co = xp.Particles.from_dict(ddd['particle_on_madx_co'])
RR = np.array(ddd['RR_madx'])

##################################################
#                   Load line                    #
# (assume frozen SC lenses are alredy installed) #
##################################################

with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])

##########################
# Configure space-charge #
##########################

if mode == 'frozen':
    pass # Already configured in line
elif mode == 'quasi-frozen':
    xf.replace_spacecharge_with_quasi_frozen(
                                    line,
                                    update_mean_x_on_track=True,
                                    update_mean_y_on_track=True)
elif mode == 'pic':
    pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
        _context=context, line=line,
        n_sigmas_range_pic_x=8,
        n_sigmas_range_pic_y=8,
        nx_grid=256, ny_grid=256, nz_grid=100,
        n_lims_x=7, n_lims_y=3,
        z_range=(-3*sigma_z, 3*sigma_z))
else:
    raise ValueError(f'Invalid mode: {mode}')


#################
# Build Tracker #
#################
tracker = xt.Tracker(_context=context,
                    line=line)

######################
# Generate particles #
######################

particles = xp.generate_matched_gaussian_bunch(_context=context,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=neps_x, nemitt_y=neps_y, sigma_z=sigma_z,
         particle_ref=part_on_co, tracker=tracker)

#########
# Track #
#########
tracker.track(particles, num_turns=3)

