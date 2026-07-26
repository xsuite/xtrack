# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo

################################################################################
# User parameters
################################################################################

########################################
# Time Limit
########################################
TIME_LIMIT          = 20
N_TURNS             = int(5E1)
N_PARTICLES_INIT    = int(1)
OPTIMIZE_FOR_TRACKING = False
FIGURE_NAME         = "compare_default_cupy_all_radiations.png"

########################################
# Line
########################################
REPO_ROOT           = Path(__file__).resolve().parents[2]
ENV_PATH            = REPO_ROOT / "examples" / "fcc_ee_solenoid" / "fccee_z_lcc.json"

########################################
# Radiation modes
########################################
RADIATION_MODES = [
    {
        "key":          "none",
        "model":        None,
        "label":        "None",
        "needs_taper":  False,
        "needs_rng":    False},
    {
        "key":          "mean",
        "model":        "mean",
        "label":        "Mean",
        "needs_taper":  True,
        "needs_rng":    False},
    {
        "key":          "quantum",
        "model":        "quantum",
        "label":        "Quantum",
        "needs_taper":  True,
        "needs_rng":    True},
    {
        "key":          "quantum-kick",
        "model":        "quantum-kick",
        "label":        "Quantum Kick",
        "needs_taper":  True,
        "needs_rng":    True}]

################################################################################
# Contexts
################################################################################
CONTEXTS = [
    {
        "key":          "default",
        "label":        "Default CPU",
        "context":      xo.context_default},
    {
        "key":          "cupy",
        "label":        "GPU CuPy",
        "context":      xo.ContextCupy()}]

################################################################################
# Helpers
################################################################################
def build_particles(line, context, n_particles):
    return line.build_particles(
        _context    = context,
        x           = np.zeros(n_particles),
        px          = np.zeros(n_particles),
        y           = np.zeros(n_particles),
        py          = np.zeros(n_particles),
        zeta        = np.zeros(n_particles),
        delta       = np.zeros(n_particles))

def track_timing_scan(line, context, needs_rng):
    tracking_times  = []
    n_particles     = []
    time_last_track = 0
    iteration       = 0

    while time_last_track < TIME_LIMIT:

        n_particles_track = int(N_PARTICLES_INIT * 2**iteration)
        print(f"Tracking with {n_particles_track} particles...")

        particles = build_particles(line, context, n_particles_track)

        if needs_rng:
            particles._init_random_number_generator()

        line.track(
            particles       = particles,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles.state == 1)

        time_last_track = line.time_last_track
        n_particles.append(n_particles_track)
        tracking_times.append(line.time_last_track)
        iteration += 1

    tracking_times = np.array(tracking_times)
    n_particles = np.array(n_particles)
    particle_turn_time = tracking_times / N_TURNS / n_particles

    return n_particles, particle_turn_time


################################################################################
# Lattice setup
################################################################################
print("\n" + "#"*80 + "\n" + "Loading Line" + "\n" + "#"*80 + "\n")
print(f"optimize_for_tracking = {OPTIMIZE_FOR_TRACKING}")

########################################
# Load line from JSON
########################################
env         = xt.load(ENV_PATH)
line_base   = env.lines["fccee_p_ring"]
line_base.build_tracker(_context = xo.context_default)

if OPTIMIZE_FOR_TRACKING:
    line_base.optimize_for_tracking(compile=False, verbose=False)

########################################
# Taper Line
########################################
line_taper  = line_base.copy()
line_taper.configure_radiation(model = "mean")
line_taper.compensate_radiation_energy_loss()
line_taper.discard_tracker()

################################################################################
# Per context/radiation line setup
################################################################################
print("\n" + "#"*80 + "\n" + "Building Lines" + "\n" + "#"*80 + "\n")

lines = {}

for mode in RADIATION_MODES:
    source_line = line_taper if mode["needs_taper"] else line_base

    for context_info in CONTEXTS:
        print(
            "Creating line for "
            f"{context_info['label']}: {mode['label']}")

        line_mode = source_line.copy()
        line_mode.configure_radiation(model = mode["model"])
        line_mode.build_tracker(_context = context_info["context"])

        lines[(context_info["key"], mode["key"])] = line_mode

################################################################################
# Track
################################################################################
print("\n" + "#"*80 + "\n" + "Tracking" + "\n" + "#"*80 + "\n")

results = {}

for mode in RADIATION_MODES:
    for context_info in CONTEXTS:
        print(
            "#"*40 + "\n"
            f"Tracking with {context_info['label']}: {mode['label']}" + "\n"
            + "#"*40)

        line_mode = lines[(context_info["key"], mode["key"])]

        results[(context_info["key"], mode["key"])] = track_timing_scan(
            line        = line_mode,
            context     = context_info["context"],
            needs_rng   = mode["needs_rng"])

################################################################################
# Plot overlayed
################################################################################
fig, ax = plt.subplots(figsize = (10, 6))

for mode in RADIATION_MODES:
    for context_info in CONTEXTS:
        n_particles, particle_turn_time = results[
            (context_info["key"], mode["key"])]

        ax.plot(
            n_particles,
            particle_turn_time * 1E6,
            label   = f"{context_info['label']}: {mode['label']}",
            marker  = "o")

ax.set_xlabel("Number of Particles")
ax.set_ylabel("Mean Tracking Time per Particle per Turn [us]")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend()

fig.suptitle(
    f"Default CPU vs GPU CuPy by Radiation Mode - {N_TURNS} Turns")

fig.savefig(FIGURE_NAME)

plt.show()
