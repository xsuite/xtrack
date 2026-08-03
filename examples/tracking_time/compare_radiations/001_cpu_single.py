# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xobjects as xo
import xtrack as xt


################################################################################
# User parameters
################################################################################

########################################
# Radiation modes to test
########################################
TRACK_NONE              = True
TRACK_MEAN              = True
TRACK_QUANTUM           = True
TRACK_QUANTUM_KICK      = True

########################################
# Test settings
########################################
TIME_LIMIT              = 20
N_TURNS                 = int(5E1)
N_PARTICLES_INIT        = int(1)
OPTIMIZE_FOR_TRACKING   = False

########################################
# Lattice Path
########################################
REPO_ROOT               = Path(__file__).resolve().parents[3]
ENV_PATH                = REPO_ROOT / "examples" / "fcc_ee_solenoid" / "fccee_z_lcc.json"

########################################
# Context Settings
########################################
CONTEXT                 = xo.context_default
CONTEXT_LABEL           = "CPU Single Thread"
FIGURE_NAME             = "compare_radiations_cpu_single.png"

################################################################################
# Helpers
################################################################################

########################################
# Radiation Modes
########################################
RADIATION_MODES = [
    {
        "key":          "none",
        "model":        None,
        "label":        "None",
        "enabled":      TRACK_NONE,
        "needs_taper":  False,
        "needs_rng":    False},
    {
        "key":          "mean",
        "model":        "mean",
        "label":        "Mean",
        "enabled":      TRACK_MEAN,
        "needs_taper":  True,
        "needs_rng":    False},
    {
        "key":          "quantum",
        "model":        "quantum",
        "label":        "Quantum",
        "enabled":      TRACK_QUANTUM,
        "needs_taper":  True,
        "needs_rng":    True},
    {
        "key":          "quantum_kick",
        "model":        "quantum-kick",
        "label":        "Quantum Kick",
        "enabled":      TRACK_QUANTUM_KICK,
        "needs_taper":  True,
        "needs_rng":    True}]


########################################
# Particle Construction
########################################
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

        line.track(particles = particles, num_turns = N_TURNS, time = True)

        assert np.all(particles.state == 1)

        time_last_track = line.time_last_track
        n_particles.append(n_particles_track)
        tracking_times.append(line.time_last_track)
        iteration += 1

    tracking_times      = np.array(tracking_times)
    n_particles         = np.array(n_particles)
    particle_turn_time  = tracking_times / N_TURNS / n_particles
    return n_particles, particle_turn_time


################################################################################
# Lattice setup
################################################################################
print("\n" + "#" * 80 + "\n" + "Loading Line" + "\n" + "#" * 80 + "\n")
print(f"optimize_for_tracking = {OPTIMIZE_FOR_TRACKING}")

env         = xt.load(ENV_PATH)
line_base   = env.lines["fccee_p_ring"]
line_base.build_tracker(_context=xo.context_default)

if OPTIMIZE_FOR_TRACKING:
    line_base.optimize_for_tracking(compile = False, verbose = False)

line_taper = line_base.copy()
line_taper.configure_radiation(model="mean")
line_taper.compensate_radiation_energy_loss()
line_taper.discard_tracker()

################################################################################
# Build lines
################################################################################
print("\n" + "#" * 80 + "\n" + "Building Lines" + "\n" + "#" * 80 + "\n")

lines = {}

for mode in RADIATION_MODES:
    if not mode["enabled"]:
        continue

    print(f"Creating line for radiation mode: {mode['label']}")
    source_line = line_taper if mode["needs_taper"] else line_base
    line_mode   = source_line.copy()
    line_mode.configure_radiation(model=mode["model"])
    line_mode.build_tracker(_context=CONTEXT)
    lines[mode["key"]] = line_mode

################################################################################
# Track
################################################################################
print("\n" + "#" * 80 + "\n" + "Tracking" + "\n" + "#" * 80 + "\n")

results = {}
for mode in RADIATION_MODES:
    if not mode["enabled"]:
        continue

    print(
        "#" * 40
        + "\n"
        + f"Tracking with radiation mode: {mode['label']}"
        + "\n"
        + "#" * 40)

    results[mode["key"]] = track_timing_scan(
        line        = lines[mode["key"]],
        context     = CONTEXT,
        needs_rng   = mode["needs_rng"])

################################################################################
# Output results
################################################################################
print("\n" + "#" * 80)
print("Raw timing results")
print("#" * 80)

for mode in RADIATION_MODES:
    if not mode["enabled"]:
        continue

    n_particles, particle_turn_time = results[mode["key"]]
    print(f"\n{CONTEXT_LABEL} | {mode['label']}")
    print("n_particles =", n_particles.tolist())
    print("us_per_particle_turn =", (particle_turn_time * 1E6).tolist())

################################################################################
# Plot
################################################################################
fig, ax = plt.subplots(figsize = (10, 6))

for mode in RADIATION_MODES:
    if not mode["enabled"]:
        continue

    n_particles, particle_turn_time = results[mode["key"]]
    ax.plot(
        n_particles,
        particle_turn_time * 1E6,
        label   = mode["label"],
        marker  = "o")

ax.set_xlabel("Number of Particles")
ax.set_ylabel("Mean Tracking Time per Particle per Turn [us]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

fig.suptitle(
    f"Tracking Time by Radiation Mode ({CONTEXT_LABEL}) - {N_TURNS} Turns")
fig.savefig(FIGURE_NAME)

plt.show()
