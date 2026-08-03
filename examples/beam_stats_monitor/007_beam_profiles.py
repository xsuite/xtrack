import numpy as np

import xtrack as xt


NUM_TURNS = 32
NUM_PARTICLES = 2000
TOTAL_INTENSITY = 1e10
NUM_X_BINS = 80
NUM_DELTA_BINS = 60

# Load the PIMMS lattice and set particle_ref.
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Add an RF cavity.
line.insert("pimms_cavity", xt.Cavity(harmonic=7, voltage=10e3), at=0.001)

# Create a BeamStatsMonitor that records scalar statistics and profiles.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    stats=["num_particles", "mean_x", "sigma_x"],
    profiles={
        "x": {"range": (-20e-3, 20e-3), "num_bins": NUM_X_BINS},
        "delta": {"range": (-3e-3, 3e-3), "num_bins": NUM_DELTA_BINS},
    },
)

# Insert the monitor at the beginning of the line.
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a bunch and apply a horizontal offset so the profile motion is
# visible over consecutive turns.
rng = np.random.default_rng(12345)
mass0 = np.asarray(line.particle_ref.mass0).ravel()[0]
p0c = np.asarray(line.particle_ref.p0c).ravel()[0]
particles = xt.Particles(
    mass0=mass0,
    p0c=p0c,
    x=rng.normal(loc=4e-3, scale=1.5e-3, size=NUM_PARTICLES),
    px=rng.normal(loc=0.0, scale=0.2e-3, size=NUM_PARTICLES),
    y=rng.normal(loc=0.0, scale=1.0e-3, size=NUM_PARTICLES),
    py=rng.normal(loc=0.0, scale=0.2e-3, size=NUM_PARTICLES),
    zeta=rng.normal(loc=0.0, scale=0.01, size=NUM_PARTICLES),
    delta=rng.normal(loc=0.0, scale=0.8e-3, size=NUM_PARTICLES),
    weight=np.full(NUM_PARTICLES, TOTAL_INTENSITY / NUM_PARTICLES),
)

# Track.
line.track(particles, num_turns=NUM_TURNS,
           with_progress=4)  # progress bar updated every 4 turns

# Inspect what has been recorded.
monitor.profile_coordinates  # is ('x', 'delta')
monitor.profile_num_bins  # is {'x': 80, 'delta': 60}
monitor.profile_bin_edges["x"].shape  # is (81,)
monitor.profile_bin_centers["x"].shape  # is (80,)

# With no bunch or slice inputs, profile arrays have axes
# (logged turns, profile bin).
monitor.profiles["x"].shape  # is (32, 80)
monitor.profiles["delta"].shape  # is (32, 60)

# Scalar statistics are recorded in the same monitor.
monitor.mean_x.shape  # is (32,)
monitor.sigma_x.shape  # is (32,)

# Plot the horizontal profile at selected turns.
import matplotlib.pyplot as plt
plt.close("all")
fig, ax = plt.subplots(figsize=(8, 4.5))

x_centers = monitor.profile_bin_centers["x"] * 1e3
x_profiles = monitor.profiles["x"]

for turn in [0, 8, 16, 24]:
    profile = x_profiles[monitor.record_index(turn)]
    normalized_profile = profile / np.max(profile)
    ax.step(
        x_centers,
        normalized_profile,
        where="mid",
        label=f"turn {turn}",
    )

ax.set_xlabel("x [mm]")
ax.set_ylabel("normalized profile")
ax.grid(True)
ax.legend()
fig.tight_layout()

plt.show()
