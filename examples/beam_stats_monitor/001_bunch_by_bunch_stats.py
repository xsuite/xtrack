import numpy as np

import xtrack as xt

NUM_TURNS = 100
NUM_SLOTS = 7
NUM_PARTICLES_PER_BUNCH = 30
BUNCH_INTENSITY = 1e10

# Load the PIMMS lattice and set particle_ref.
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Add an RF cavity and use its bucket spacing to define the bunch train.
harmonic = 7
bunch_spacing_zeta = line.get_length() / harmonic
line.insert("pimms_cavity", xt.Cavity(harmonic=harmonic, voltage=10e3),
            at=0.001)

filled_slots = np.arange(NUM_SLOTS)
filling_scheme = np.ones(NUM_SLOTS, dtype=int)

# Define a monitor to record bunch-by-bunch statistics at each turn
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    filled_slots=filled_slots,
    bunch_spacing_zeta=bunch_spacing_zeta,
    stats=["num_particles", "mean_x", "mean_zeta", "sigma_x"],
)

# Insert the monitor at the beginning of the line
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a multi-bunch beam with a sinusoidal offset in x.
np.random.seed(12345)
particles = line.xpart.generate_matched_gaussian_multibunch_beam(
    filling_scheme=filling_scheme,
    bunch_num_particles=NUM_PARTICLES_PER_BUNCH,
    bunch_intensity_particles=BUNCH_INTENSITY,
    nemitt_x=1e-8,
    nemitt_y=1e-8,
    sigma_z=0.01,
    bucket_length=bunch_spacing_zeta,
)
particles.x = -1e-3 * np.sin(
    2 * np.pi * particles.zeta / (NUM_SLOTS * bunch_spacing_zeta))

# Track
line.track(particles, num_turns=NUM_TURNS,
           with_progress=10)  # progress bar updated every 10 turns

# Inspect what has been recorded.
monitor.stats  # is ('num_particles', 'mean_x', 'mean_zeta', 'sigma_x')
monitor.available_levels  # is ('beam', 'bunch')
monitor.default_level  # is 'bunch'
monitor.selected_slots  # is array([0, 1, 2, ..., 6])
monitor.turns  # is array([0, 1, 2, ..., 99])

# The shape is (logged turns, selected slots).
monitor.mean_x.shape  # is (100, 7)

# Bunch-level stats are the default when bunches are configured.
mean_x_by_bunch = monitor.get("mean_x")
mean_x_by_bunch.shape  # is (100, 7)

# The selected-slot axis follows monitor.selected_slots.
monitor.get("mean_x", slot=3).shape  # is (100,)
monitor.get("mean_x", turn=20, slot=[0, 3, 6]).shape  # is (3,)

# Beam-level stats are also available as weighted sums over bunches.
mean_x_beam = monitor.get("mean_x", level="beam")
mean_x_beam.shape  # is (100,)

# Extract and plot bunch centroids for 10 consecutive turns.
turns_to_plot = monitor.turns[:10]
mean_x_to_plot = monitor.get("mean_x", turn=turns_to_plot) * 1000

import matplotlib.pyplot as plt
plt.close('all')
fig, ax = plt.subplots()

for ii, turn in enumerate(turns_to_plot):
    ax.plot(monitor.selected_slots, mean_x_to_plot[ii],
            "o-", label=f"turn {turn}")
ax.set_xlabel("slot")
ax.set_ylabel("mean_x [mm]")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

plt.show()
