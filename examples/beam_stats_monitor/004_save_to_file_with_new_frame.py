import h5py
import numpy as np

import xtrack as xt


TOTAL_TURNS = 60
FRAME_TURNS = 20

# Load the PIMMS lattice and set particle_ref.
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Add an RF cavity.
line.insert("pimms_cavity", xt.Cavity(harmonic=7, voltage=10e3), at=0.001)

output_file = "beam_stats_monitor_frames.h5"

# The monitor stores only one frame in memory. The HDF5 file receives all
# frames as one flat time series.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=FRAME_TURNS,
    stats=["num_particles", "mean_x", "sigma_x"],
    output_file=output_file,
)
line.insert("beam_stats_monitor", monitor, at=0)

# Generate a bunch.
np.random.seed(12345)
particles = line.xpart.generate_matched_gaussian_bunch(
    num_particles=50,
    total_intensity_particles=1e10,
    nemitt_x=1e-6,
    nemitt_y=1e-6,
    sigma_z=0.01,
)

# Track, save, clear the in-memory frame, and retarget the same monitor to
# the next turn interval.
for start_turn in range(0, TOTAL_TURNS, FRAME_TURNS):
    if start_turn != 0:
        monitor.start_new_frame(start_at_turn=start_turn)

    line.track(particles, num_turns=FRAME_TURNS)
    monitor.save_to_file()

# The in-memory arrays contain only the last frame.
monitor.turns  # is array([40, 41, ..., 59])
monitor.mean_x.shape  # is (20,)

# The HDF5 file contains all saved frames as one flat sequence.
with h5py.File(output_file, "r") as h5file:
    turns_from_file = h5file["turns"][...]
    mean_x_from_file = h5file["stats/beam/mean_x"][...]
    sigma_x_from_file = h5file["stats/beam/sigma_x"][...]

turns_from_file  # is array([0, 1, 2, ..., 59])
mean_x_from_file.shape  # is (60,)
sigma_x_from_file.shape  # is (60,)
