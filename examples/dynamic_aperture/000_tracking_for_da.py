import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.build_tracker()

# Switch on LHC octupole circuits to have a smaller dynamic aperture
for arc in ['12', '23', '34', '45', '56', '67', '78', '81']:
    line.vars[f'kod.a{arc}b1'] = 2.0
    line.vars[f'kof.a{arc}b1'] = 2.0

# Generate normalized particle coordinates on a polar grid
n_r = 50
n_theta = 60
x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
    r_range=(0, 40.), # beam sigmas
    theta_range=(0, np.pi/2),
    nr=n_r, ntheta=n_theta)

# Set initial momentum deviation
delta_init = 0 # In case off-momentum DA is needed

# Match particles to the machine optics and orbit
particles = line.build_particles(
    x_norm=x_normalized, px_norm=0,
    y_norm=y_normalized, py_norm=0,
    nemitt_x=3e-6, nemitt_y=3e-6, # normalized emittances
    delta=delta_init)

# # Optional: activate multi-core CPU parallelization
# line.discard_tracker()
# line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

# Track
line.track(particles, num_turns=200, time=True, with_progress=5)

print(f'Tracked in {line.time_last_track} seconds')

# Sort particles to get the initial order
# (during tracking lost particles are moved to the end)
particles.sort(interleave_lost_particles=True)

# Plot result using scatter or pcolormesh
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.scatter(x_normalized, y_normalized, c=particles.at_turn)
plt.xlabel(r'$A_x [\sigma]$')
plt.ylabel(r'$A_y [\sigma]$')
cb = plt.colorbar()
cb.set_label('Lost at turn')

plt.figure(2)
plt.pcolormesh(
    x_normalized.reshape(n_r, n_theta), y_normalized.reshape(n_r, n_theta),
    particles.at_turn.reshape(n_r, n_theta), shading='gouraud')
plt.xlabel(r'$A_x [\sigma]$')
plt.ylabel(r'$A_y [\sigma]$')
ax = plt.colorbar()
ax.set_label('Lost at turn')

plt.show()