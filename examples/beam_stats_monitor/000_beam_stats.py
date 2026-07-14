import numpy as np


import xtrack as xt

NUM_TURNS = 300
NUM_PARTICLES = 50

# Load the PIMMS lattice from the test data.
env = xt.load("../../test_data/pimms/PIMMS.seq")
line = env.pimms
line.particle_ref = xt.Particles("proton", kinetic_energy0=100e6)

line["kqfa"] = 0.30247
line["kqfb"] = 0.523281
line["kqd"] = -0.518932

# Add an RF cavity to provide longitudinal focusing.
line.insert("pimms_cavity", xt.Cavity(harmonic=7, voltage=10e3), at=0.001)

# No zeta_range or num_slices are provided: this records whole-beam stats.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    stats=["num_particles", "mean_x", "mean_zeta", "sigma_x"],
)

line.insert("beam_stats_monitor", monitor, at=0)

rng = np.random.default_rng(12345)
particles = line.build_particles(
    x=1e-3 + 0.2e-3 * rng.normal(size=NUM_PARTICLES),
    px=20e-6 * rng.normal(size=NUM_PARTICLES),
    zeta=0.05 + 0.01 * rng.normal(size=NUM_PARTICLES),
    delta=1e-4 * rng.normal(size=NUM_PARTICLES),
)

line.track(particles, num_turns=NUM_TURNS,
           with_progress=10 # progress bar updated every 10 turns
)

# Inspect what has been recorded.
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
