import numpy as np

import xtrack as xt


NUM_TURNS = 10
NUM_SLOTS = 3
NUM_SLICES = 48
NUM_PARTICLES_PER_BUNCH = 600
BUNCH_INTENSITY = 1e10
SIGMA_Z = 2.0
ZETA_RANGE = (-10.0, 10.0)

# Load the PIMMS lattice and set particle_ref.
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Add an RF cavity and use its bucket spacing to define the bunch train.
harmonic = NUM_SLOTS
bunch_spacing_zeta = line.get_length() / harmonic
line.insert("pimms_cavity", xt.Cavity(harmonic=harmonic, voltage=10e3),
            at=0.001)

filling_scheme = np.ones(NUM_SLOTS, dtype=int)
selected_slots = [0, 1, 2]

# Define a monitor to record slice-by-slice statistics at each turn.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    filling_scheme=filling_scheme,
    selected_slots=selected_slots,
    bunch_spacing_zeta=bunch_spacing_zeta,
    zeta_range=ZETA_RANGE,
    num_slices=NUM_SLICES,
    stats=["num_particles", "mean_x", "mean_zeta", "sigma_x"],
)

# Insert the monitor at the beginning of the line.
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a multi-bunch beam.
np.random.seed(12345)
particles = line.xpart.generate_matched_gaussian_multibunch_beam(
    filling_scheme=filling_scheme,
    bunch_num_particles=NUM_PARTICLES_PER_BUNCH,
    bunch_intensity_particles=BUNCH_INTENSITY,
    nemitt_x=1e-12,
    nemitt_y=1e-12,
    sigma_z=SIGMA_Z,
    bucket_length=bunch_spacing_zeta,
)

# Impress a horizontal sinusoidal pattern with a wavelength comparable to the
# bunch spacing, so the phase changes visibly along the bunch train.
train_wavelength = 0.3 * bunch_spacing_zeta
particles.x += 1.0e-3 * np.sin(2 * np.pi * particles.zeta
                                / train_wavelength)

# Track.
line.track(particles, num_turns=NUM_TURNS,
           with_progress=1)  # progress bar updated every turn

# Inspect what has been recorded.
monitor.stats  # is ('num_particles', 'mean_x', 'mean_zeta', 'sigma_x')
monitor.available_levels  # is ('beam', 'bunch', 'slice')
monitor.default_level  # is 'slice'
monitor.selected_slots  # is array([0, 1, 2])
monitor.turns  # is array([0, 1, 2, ..., 9])

# The default slice-level shape is (logged turns, selected slots, slices).
monitor.mean_x.shape  # is (10, 3, 48)
monitor.zeta_centers.shape  # is (3, 48)

# Slice coordinates are reported for each selected physical slot.
for slot, zeta_centers in zip(monitor.selected_slots, monitor.zeta_centers):
    print("slot", slot, "zeta centers:", zeta_centers)

# A longitudinal coordinate can be converted to a recorded slice index.
slice_index = monitor.slice_index(
    zeta=-selected_slots[0] * bunch_spacing_zeta + 0.01,
    slot=selected_slots[0])
print("example slice index:", slice_index)

# Slice-level stats can be selected by turn, slot, and slice index.
monitor.get(
    "mean_x", level="slice", turn=5, slot=selected_slots[0],
    slice_index=slice_index).shape  # is ()

# Bunch-level and beam-level statistics are also available as reductions of
# the recorded slice data.
mean_x_bunch = monitor.get("mean_x", level="bunch")
mean_x_bunch.shape  # is (10, 3)

mean_x_beam = monitor.get("mean_x", level="beam")
mean_x_beam.shape  # is (10,)

# Plot the dipole moment (mean_x * num_particles) versus absolute zeta for all turns.
import matplotlib.pyplot as plt
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 4.5))

for turn in monitor.turns:
    for slot_index, slot in enumerate(monitor.selected_slots):
        label = f"turn {turn}" if slot_index == 0 else None
        ax.plot(
            monitor.zeta_centers[slot_index],
            (monitor.get("mean_x", turn=turn, slot=slot)
             * monitor.get("num_particles", turn=turn, slot=slot)),
            "-", label=label)

ax.set_xlabel("zeta [m]")
ax.set_ylabel("mean_x * num_particles [m]")
ax.grid(True)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

plt.show()
