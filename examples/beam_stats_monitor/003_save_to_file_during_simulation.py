import h5py
import numpy as np

import xtrack as xt


NUM_TURNS = 60
SAVE_EVERY = 15
NUM_SLOTS = 3
NUM_PARTICLES_PER_BUNCH = 30
BUNCH_INTENSITY = 1e10

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

output_file = "beam_stats_monitor_progress.h5"

# The monitor is configured for the full simulation, so all records remain
# available in memory. The output file is initialized at construction.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    filling_scheme=filling_scheme,
    bunch_spacing_zeta=bunch_spacing_zeta,
    stats=["num_particles", "mean_x", "sigma_x"],
    output_file=output_file,
)
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a multi-bunch beam.
np.random.seed(12345)
particles = line.xpart.generate_matched_gaussian_multibunch_beam(
    filling_scheme=filling_scheme,
    bunch_num_particles=NUM_PARTICLES_PER_BUNCH,
    bunch_intensity_particles=BUNCH_INTENSITY,
    nemitt_x=1e-6,
    nemitt_y=1e-6,
    sigma_z=0.01,
    bucket_length=bunch_spacing_zeta,
)

# Track in chunks and save the newly recorded suffix after each chunk.
for _ in range(NUM_TURNS // SAVE_EVERY):
    line.track(particles, num_turns=SAVE_EVERY)
    monitor.save_to_file()

# The monitor still keeps the full configured frame in memory.
monitor.turns  # is array([0, 1, 2, ..., 59])
monitor.mean_x.shape  # is (60, 3)

# Open the HDF5 file and access the saved statistics.
with h5py.File(output_file, "r") as h5file:
    turns_from_file = h5file["turns"][...]
    mean_x_beam_from_file = h5file["stats/beam/mean_x"][...]
    sigma_x_beam_from_file = h5file["stats/beam/sigma_x"][...]
    mean_x_bunch_from_file = h5file["stats/bunch/mean_x"][...]
    sigma_x_bunch_from_file = h5file["stats/bunch/sigma_x"][...]

turns_from_file.shape  # is (60,)
mean_x_beam_from_file.shape  # is (60,)
sigma_x_beam_from_file.shape  # is (60,)
mean_x_bunch_from_file.shape  # is (60, 3)
sigma_x_bunch_from_file.shape  # is (60, 3)
