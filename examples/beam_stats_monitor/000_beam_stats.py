import numpy as np


import xtrack as xt

# Load the PIMMS lattice and set particle_ref
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Add an RF cavity
line.insert("pimms_cavity", xt.Cavity(harmonic=7, voltage=10e3), at=0.001)

# Create a BeamStatsMonitor to record beam statistics at each turn.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=300,
    stats=["num_particles", "mean_x", "mean_zeta", "sigma_x"],
)

# Insert the monitor at the beginning of the line
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a bunch
np.random.seed(12345)
particles = line.xpart.generate_matched_gaussian_bunch(
    num_particles=50,
    total_intensity_particles=1e10,
    nemitt_x=1e-6,
    nemitt_y=1e-6,
    sigma_z=0.01,
)

# Track
line.track(particles, num_turns=300,
           with_progress=10 # progress bar updated every 10 turns
)

# Inspect what has been recorded by the monitor
monitor.stats  # is ('num_particles', 'mean_x', 'mean_zeta', 'sigma_x')
monitor.available_levels  # is ('beam',)
monitor.default_level  # is 'beam'
monitor.turns  # is array([0, 1, 2, ..., 299])

# The shape is (logged turns,).
monitor.mean_x.shape  # is (300,)

# num_particles is the sum of particle weights in the whole beam.
monitor.num_particles.shape  # is (300,)

# A recorded statistic can be accessed as an attribute or with monitor.get().
monitor.mean_x[monitor.record_index(10)]
monitor.get("mean_x", turn=10)  # same as above

# Plot the beam centroids as a function of turn.
import matplotlib.pyplot as plt
plt.close('all')
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(monitor.turns, monitor.mean_x * 1000)
axes[0].set_ylabel("mean_x [mm]")
axes[1].plot(monitor.turns, monitor.mean_zeta * 1000)
axes[1].set_ylabel("mean_zeta [mm]")
axes[1].set_xlabel("turn")
plt.show()
