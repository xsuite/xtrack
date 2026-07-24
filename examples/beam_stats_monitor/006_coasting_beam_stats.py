import numpy as np

import xtrack as xt


NUM_TURNS = 8
NUM_SLICES = 80
NUM_PARTICLES = 4000
TOTAL_INTENSITY = 1e10
AMPLITUDE_X = 1.0e-3
MODULATION_PHASE = 0.3

# Load the PIMMS lattice and set particle_ref.
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Use the periodic Twiss parameters at the monitor location to initialize a
# matched transverse wave.
tw = line.twiss(method="4d")
qx = tw.qx
betx = tw.betx[0]
alfx = tw.alfx[0]

# Create a coasting BeamStatsMonitor to record full-turn slice statistics.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    coasting=True,
    num_slices=NUM_SLICES,
    stats=["num_particles", "mean_x", "mean_zeta"],
)

# Insert the monitor at the beginning of the line.
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a coasting distribution spanning one full turn.
line_length = line.get_length()
mass0 = np.asarray(line.particle_ref.mass0).ravel()[0]
p0c = np.asarray(line.particle_ref.p0c).ravel()[0]
beta0 = np.asarray(line.particle_ref.beta0).ravel()[0]
zeta = np.linspace(-0.5 * line_length, 0.5 * line_length,
                   NUM_PARTICLES, endpoint=False)
zeta += 0.5 * line_length / NUM_PARTICLES
particles = xt.Particles(
    mass0=mass0,
    p0c=p0c,
    zeta=zeta,
    weight=np.full(NUM_PARTICLES, TOTAL_INTENSITY / NUM_PARTICLES),
)

# Impress a sinusoidal horizontal modulation over the coasting beam.
theta = -2 * np.pi * qx * particles.zeta / line_length + MODULATION_PHASE
particles.x = AMPLITUDE_X * np.sin(theta)
particles.px = (
    AMPLITUDE_X / betx * (np.cos(theta) - alfx * np.sin(theta)))

# Track for multiple turns.
line.track(particles, num_turns=NUM_TURNS,
           with_progress=1)  # progress bar updated every turn

# Inspect what has been recorded.
monitor.stats  # is ('num_particles', 'mean_x', 'mean_zeta')
monitor.available_levels  # is ('beam', 'slice')
monitor.default_level  # is 'slice'
monitor.turns  # is array([0, 1, 2, ..., 7])

# In coasting mode the default slice-level shape is (logged turns, slices).
monitor.mean_x.shape  # is (8, 80)
monitor.zeta_centers  # is None

# Use line-length-aware helpers to build unwrapped coordinates.
tt = monitor.time_centers(line_length=line_length, beta0=beta0)
tt.shape  # is (8, 80)

zz = monitor.zeta_centers_unwrapped(line_length=line_length)
zz.shape  # is (8, 80)

# Plot the coasting-beam centroid as a function of unwrapped time.
import matplotlib.pyplot as plt
plt.close("all")
fig, ax = plt.subplots(figsize=(8, 4.5))

num_particles = np.asarray(monitor.num_particles)
mean_x = np.asarray(monitor.mean_x)

valid = num_particles > 0
tt_plot = tt[valid]
mean_x_plot = mean_x[valid]

ax.plot(
    tt_plot * 1e6,
    mean_x_plot * 1e3,
    "-",
    linewidth=1.5,
)

ax.set_xlabel("time [us]")
ax.set_ylabel("mean_x [mm]")
ax.grid(True)
fig.tight_layout()

plt.show()
