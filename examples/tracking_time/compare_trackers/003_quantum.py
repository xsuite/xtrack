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
# Trackers to test
########################################
TRACK_CPU_SINGLE        = True
TRACK_CPU_OPENMP        = False
TRACK_GPU_CUPY          = False

########################################
# OpenMP settings
########################################
OPEN_MP_THREADS         = 4

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
# Radiation Settings
########################################
RADIATION_MODEL         = "quantum"
NEEDS_TAPER             = True
NEEDS_RNG               = True
RADIATION_LABEL         = "Quantum Radiation"
FIGURE_NAME             = "compare_trackers_quantum.png"

################################################################################
# Helpers
################################################################################

########################################
# Context Generation
########################################
def make_tracker_contexts():
    contexts    = []
    if TRACK_CPU_SINGLE:
        contexts.append({
            "key":      "cpu_single",
            "label":    "CPU Single Thread",
            "context":  xo.context_default})
    if TRACK_CPU_OPENMP:
        contexts.append({
            "key":      "cpu_openmp",
            "label":    f"CPU OpenMP ({OPEN_MP_THREADS} Threads)",
            "context":  xo.ContextCpu(omp_num_threads = OPEN_MP_THREADS)})
    if TRACK_GPU_CUPY:
        contexts.append({
            "key":      "gpu_cupy",
            "label":    "GPU CuPy",
            "context":  xo.ContextCupy()})
    return contexts

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


########################################
# Scan tracking time
########################################
def track_timing_scan(line, context):
    tracking_times  = []
    n_particles     = []
    time_last_track = 0
    iteration       = 0

    while time_last_track < TIME_LIMIT:
        n_particles_track = int(N_PARTICLES_INIT * 2**iteration)
        print(f"Tracking with {n_particles_track} particles...")

        particles = build_particles(line, context, n_particles_track)
        if NEEDS_RNG:
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

tracker_contexts = make_tracker_contexts()

env         = xt.load(ENV_PATH)
line_base   = env.lines["fccee_p_ring"]
line_base.build_tracker(_context = xo.context_default)

if OPTIMIZE_FOR_TRACKING:
    line_base.optimize_for_tracking(compile = False, verbose = False)

line_source = line_base
if NEEDS_TAPER:
    line_source = line_base.copy()
    line_source.configure_radiation(model="mean")
    line_source.compensate_radiation_energy_loss()
    line_source.discard_tracker()

################################################################################
# Build lines
################################################################################
print("\n" + "#" * 80 + "\n" + "Building Lines" + "\n" + "#" * 80 + "\n")

lines = {}

for context_info in tracker_contexts:
    print(f"Creating line for {context_info['label']}")
    line_mode = line_source.copy()
    line_mode.configure_radiation(model=RADIATION_MODEL)
    line_mode.build_tracker(_context=context_info["context"])
    lines[context_info["key"]] = line_mode


################################################################################
# Track
################################################################################
print("\n" + "#" * 80 + "\n" + "Tracking" + "\n" + "#" * 80 + "\n")

results = {}
for context_info in tracker_contexts:
    print(
        "#" * 40
        + "\n"
        + f"Tracking with {context_info['label']}"
        + "\n"
        + "#" * 40)
    results[context_info["key"]] = track_timing_scan(
        line    = lines[context_info["key"]],
        context = context_info["context"])

################################################################################
# Plot
################################################################################
fig, ax = plt.subplots(figsize=(10, 6))

for context_info in tracker_contexts:
    n_particles, particle_turn_time = results[context_info["key"]]
    ax.plot(
        n_particles,
        particle_turn_time * 1e6,
        label   = context_info["label"],
        marker  = "o")

ax.set_xlabel("Number of Particles")
ax.set_ylabel("Mean Tracking Time per Particle per Turn [us]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

fig.suptitle(
    f"Tracking Time by Tracker Context ({RADIATION_LABEL}) - {N_TURNS} Turns")
fig.savefig(FIGURE_NAME)

plt.show()
