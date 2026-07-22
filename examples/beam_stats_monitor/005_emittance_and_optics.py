import numpy as np

import xtrack as xt


NUM_TURNS = 20
NUM_PARTICLES = 10_000
NEMITT_X = 1.0e-6
NEMITT_Y = 2.0e-6
SIGMA_Z = 0.01

# Load the PIMMS lattice and set particle_ref.
env = xt.load(["../../test_data/pimms/PIMMS.seq",
               "../../test_data/pimms/pimms_optics.str"])
line = env.pimms
line.set_particle_ref("proton", kinetic_energy0=100e6)

# Add an RF cavity so that a matched bunch can be generated.
line.insert("pimms_cavity", xt.Cavity(harmonic=7, voltage=10e3), at=0.001)

# The monitor is inserted at the start of the line. The first row of the Twiss
# table is therefore the model reference for the measured beta functions.
tw = line.twiss()
betx_model = tw.betx[0]
bety_model = tw.bety[0]

# Requesting normal-mode emittances or covariance optics makes the monitor store
# the full 6D covariance moment set. These stats can be mixed with ordinary beam
# statistics in the same monitor. The covariance and optics calculations are
# done as Python-side postprocessing from the stored primitive moments.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=NUM_TURNS,
    stats=[
        "num_particles",
        "mean_x", "mean_y", "sigma_x", "sigma_y",
        "gemitt_x", "gemitt_y", "gemitt_zeta",
        "nemitt_x", "nemitt_y", "nemitt_zeta",
        "betx", "bety",
        "alfx", "alfy",
        "dx", "dpx",
    ],
)

line.insert("beam_stats_monitor", monitor, at=0)

# Generate a matched Gaussian bunch for the line. The particle weights sum to
# the total bunch intensity, and the monitor uses these weights for all
# statistics.
np.random.seed(12345)
particles = line.xpart.generate_matched_gaussian_bunch(
    num_particles=NUM_PARTICLES,
    total_intensity_particles=1e10,
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y,
    sigma_z=SIGMA_Z,
)

# Track and record the covariance-derived quantities turn by turn.
line.track(particles, num_turns=NUM_TURNS,
           with_progress=1)  # progress bar updated every turn

# Inspect the recorded quantities.
monitor.stats  # includes ordinary stats, emittances, optics, and dispersion
monitor.available_levels  # is ('beam',)
monitor.default_level  # is 'beam'
monitor.turns  # is array([0, 1, 2, ..., 19])

# Ordinary beam statistics are recorded together with covariance-derived stats.
monitor.mean_x.shape  # is (20,)
monitor.sigma_x.shape  # is (20,)

# Normal-mode emittances are available as arrays indexed by logged turn.
monitor.nemitt_x.shape  # is (20,)
monitor.nemitt_y.shape  # is (20,)
monitor.nemitt_zeta.shape  # is (20,)

# Covariance optics are also available turn by turn.
monitor.betx.shape  # is (20,)
monitor.bety.shape  # is (20,)

# Plot measured emittances and beta functions versus turn.
import matplotlib.pyplot as plt
plt.close("all")
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

axes[0].plot(monitor.turns, monitor.nemitt_x * 1e6, "o-", label="x mode")
axes[0].plot(monitor.turns, monitor.nemitt_y * 1e6, "o-", label="y mode")
axes[0].axhline(NEMITT_X * 1e6, color="C0", linestyle="--", linewidth=1)
axes[0].axhline(NEMITT_Y * 1e6, color="C1", linestyle="--", linewidth=1)
axes[0].set_ylabel("nemitt [um]")
axes[0].set_ylim(0, None)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(monitor.turns, monitor.betx, "o-", label="betx from covariance")
axes[1].plot(monitor.turns, monitor.bety, "o-", label="bety from covariance")
axes[1].axhline(betx_model, color="C0", linestyle="--", linewidth=1)
axes[1].axhline(bety_model, color="C1", linestyle="--", linewidth=1)
axes[1].set_ylabel("beta [m]")
axes[1].set_xlabel("turn")
axes[1].set_ylim(0, None)
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
plt.show()
