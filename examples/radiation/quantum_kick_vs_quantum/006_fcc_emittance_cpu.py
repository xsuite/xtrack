# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xcoll as xc
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Tracking settings
########################################
N_PARTICLES             = int(2**10)
N_TURNS                 = int(1.5E2)
SMOOTHING_WINDOW        = int(1E1)
EQ_WINDOW               = int(3E1)

########################################
# FCC-ee tt lattice
########################################
REPO_ROOT               = Path(__file__).resolve().parents[3]
ENV_PATH                = REPO_ROOT / "test_data" / "fcc_ee" / "fccee_t.seq"
LINE_NAME               = "fccee_p_ring"
MONITOR_POINT           = "ip.1"
P0C                     = 182.5E9

########################################
# Acceptance limits
########################################
MODEL_LATE_REL_LIMIT        = 0.35
MODEL_CURVE_NRMSE_LIMIT     = 0.35
REFERENCE_LATE_REL_LIMIT    = 0.50
THEORY_CURVE_NRMSE_LIMIT    = 0.50
TRAJECTORY_CORRELATION_MIN  = 0.70
TRAJECTORY_START_FRACTION   = 0.10
MIN_TRAJECTORY_POINTS       = 10
RAISE_ON_FAIL               = False

########################################
# Plot settings
########################################
PLOT_RESULTS            = True

########################################
# Context
########################################
CONTEXT                 = xo.context_default

################################################################################
# Smoothing helpers
################################################################################

########################################
# Trailing moving average
########################################
def trailing_moving_average(turns, values, window):
    turns = np.asarray(turns)
    values = np.asarray(values)
    window = min(int(window), len(values))

    if window <= 1:
        return turns.copy(), values.copy(), window

    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return turns[window - 1:], smoothed, window


########################################
# Relative difference
########################################
def symmetric_relative_difference(reference, candidate):
    scale = 0.5 * (np.abs(reference) + np.abs(candidate))
    out = np.full(np.shape(scale), np.nan, dtype=float)
    mask = scale > np.finfo(float).tiny
    out[mask] = (candidate[mask] - reference[mask]) / scale[mask]
    return out

################################################################################
# Lattice setup
################################################################################

########################################
# Load and taper lattice
########################################
def load_tapered_line():
    print("\n" + "#" * 80)
    print("Loading and Tapering FCC-ee tt Lattice")
    print("#" * 80 + "\n")

    env = xt.load(ENV_PATH)
    line = env.lines[LINE_NAME]
    line.set_particle_ref("positron", p0c=P0C)
    line.build_tracker(_context=CONTEXT)

    line_tapered = line.copy()
    line_tapered.configure_radiation(model="mean")
    line_tapered.compensate_radiation_energy_loss()

    twiss = line_tapered.twiss(radiation_analysis=True)
    line_tapered.discard_tracker()

    return line_tapered, twiss


########################################
# Radiation reference
########################################
def radiation_reference(twiss):
    return {
        "mode_names":            ["I", "III"],
        "equilibrium_emittance": np.array([
            twiss.eq_gemitt_x,
            twiss.eq_gemitt_zeta]),
        "damping_constants":     np.asarray([
            twiss.damping_constants_turns[0],
            twiss.damping_constants_turns[2]], dtype=float),
        "energy_loss":           float(twiss.energy_loss)}


########################################
# Build one model line
########################################
def build_model_line(line_tapered, radiation_model):
    line = line_tapered.copy()

    monitor = xc.EmittanceMonitor.install(
        line                = line,
        name                = "emittance_monitor",
        at                  = MONITOR_POINT,
        num_particles       = N_PARTICLES,
        start_at_turn       = 0,
        stop_at_turn        = N_TURNS,
        suppress_warnings   = True)

    line.configure_radiation(model=radiation_model)

    return line, monitor


########################################
# Build zero-amplitude particles
########################################
def build_zero_particles(line):
    particles = line.build_particles(
        _context    = CONTEXT,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))

    particles._init_random_number_generator()
    return particles

################################################################################
# Tracking
################################################################################

########################################
# Extract monitor results
########################################
def extract_monitor_results(monitor):
    return {
        "turns":        np.asarray(monitor.turns),
        "count":        np.asarray(monitor.count),
        "gemitt_I":     np.asarray(monitor.gemitt_x),
        "gemitt_III":   np.asarray(monitor.gemitt_zeta)}


########################################
# Track one radiation model
########################################
def track_model(line_tapered, radiation_model):
    print("\n" + "#" * 80)
    print(f"Tracking Radiation Model: {radiation_model}")
    print("#" * 80 + "\n")

    line, monitor = build_model_line(line_tapered, radiation_model)
    particles = build_zero_particles(line)

    time_start = time.perf_counter()
    line.track(
        particles       = particles,
        num_turns       = N_TURNS,
        with_progress   = 10)
    tracking_time = time.perf_counter() - time_start

    particle_states = CONTEXT.nparray_from_context_array(particles.state)
    n_surviving = np.sum(particle_states > 0)

    results = extract_monitor_results(monitor)
    results.update({
        "radiation_model":   radiation_model,
        "tracking_time":     tracking_time,
        "n_surviving":       int(n_surviving)})

    print(f"Tracking time       = {tracking_time:.6f} s")
    print(f"Surviving particles = {n_surviving}/{N_PARTICLES}")

    return results

################################################################################
# Expected emittance trajectory
################################################################################

########################################
# Growth from zero amplitude
########################################
def expected_emittance_growth(turns, equilibrium_emittance, damping_constant):
    turns = np.asarray(turns, dtype=float)
    return equilibrium_emittance * (
        1.0 - np.exp(-2.0 * damping_constant * turns))


########################################
# Prepare one mode
########################################
def prepare_mode_results(
        mode_name, mode_index, reference, quantum, quantum_kick):

    raw_quantum = quantum[f"gemitt_{mode_name}"]
    raw_quantum_kick = quantum_kick[f"gemitt_{mode_name}"]
    turns = quantum["turns"]

    smooth_turns, smooth_quantum, effective_window = (
        trailing_moving_average(turns, raw_quantum, SMOOTHING_WINDOW))
    smooth_turns_kick, smooth_quantum_kick, _ = trailing_moving_average(
        quantum_kick["turns"], raw_quantum_kick, SMOOTHING_WINDOW)

    if not np.array_equal(smooth_turns, smooth_turns_kick):
        raise RuntimeError("Monitor turn arrays do not agree")

    equilibrium = reference["equilibrium_emittance"][mode_index]
    damping = reference["damping_constants"][mode_index]
    theory_raw = expected_emittance_growth(turns, equilibrium, damping)
    _theory_turns, theory_smooth, _ = trailing_moving_average(
        turns, theory_raw, SMOOTHING_WINDOW)

    return {
        "name":                 mode_name,
        "index":                mode_index,
        "turns":                turns,
        "raw_quantum":          raw_quantum,
        "raw_quantum_kick":     raw_quantum_kick,
        "smooth_turns":         smooth_turns,
        "smooth_quantum":       smooth_quantum,
        "smooth_quantum_kick":  smooth_quantum_kick,
        "theory_raw":           theory_raw,
        "theory_smooth":        theory_smooth,
        "equilibrium":          equilibrium,
        "damping":              damping,
        "effective_window":     effective_window}

################################################################################
# Validation helpers
################################################################################

########################################
# PASS, FAIL or LOWSTAT
########################################
def limit_status(value, limit, lowstat=False, minimum=False):
    if lowstat:
        return "LOWSTAT"
    if minimum:
        return "PASS" if value >= limit else "FAIL"
    return "PASS" if value <= limit else "FAIL"


########################################
# Normalized RMS difference
########################################
def normalized_rms_difference(reference, candidate, scale):
    if scale <= np.finfo(float).tiny:
        return np.nan
    return np.sqrt(np.mean((candidate - reference) ** 2)) / scale


########################################
# Correlation
########################################
def trajectory_correlation(reference, candidate):
    if len(reference) < 2:
        return np.nan
    if np.std(reference) == 0 or np.std(candidate) == 0:
        return np.nan
    return np.corrcoef(reference, candidate)[0, 1]


########################################
# Validate one mode
########################################
def validate_mode(mode):
    turns = mode["smooth_turns"]
    quantum = mode["smooth_quantum"]
    quantum_kick = mode["smooth_quantum_kick"]
    theory = mode["theory_smooth"]
    equilibrium = mode["equilibrium"]

    late_start = max(0, N_TURNS - EQ_WINDOW)
    late_mask = turns >= late_start

    trajectory_mask = theory >= TRAJECTORY_START_FRACTION * equilibrium
    n_trajectory = np.sum(trajectory_mask)
    lowstat_trajectory = n_trajectory < MIN_TRAJECTORY_POINTS
    lowstat_late = np.sum(late_mask) < MIN_TRAJECTORY_POINTS

    late_quantum = np.mean(quantum[late_mask]) if np.any(late_mask) else np.nan
    late_quantum_kick = (
        np.mean(quantum_kick[late_mask]) if np.any(late_mask) else np.nan)

    late_model_relative = abs(symmetric_relative_difference(
        np.array([late_quantum]), np.array([late_quantum_kick]))[0])
    late_quantum_reference = abs(late_quantum / equilibrium - 1.0)
    late_kick_reference = abs(late_quantum_kick / equilibrium - 1.0)

    if lowstat_trajectory:
        model_curve_nrmse = np.nan
        quantum_theory_nrmse = np.nan
        kick_theory_nrmse = np.nan
        quantum_correlation = np.nan
        kick_correlation = np.nan
    else:
        model_curve_nrmse = normalized_rms_difference(
            quantum[trajectory_mask],
            quantum_kick[trajectory_mask],
            equilibrium)
        quantum_theory_nrmse = normalized_rms_difference(
            theory[trajectory_mask],
            quantum[trajectory_mask],
            equilibrium)
        kick_theory_nrmse = normalized_rms_difference(
            theory[trajectory_mask],
            quantum_kick[trajectory_mask],
            equilibrium)
        quantum_correlation = trajectory_correlation(
            theory[trajectory_mask], quantum[trajectory_mask])
        kick_correlation = trajectory_correlation(
            theory[trajectory_mask], quantum_kick[trajectory_mask])

    checks = [
        {
            "name":     "late model relative difference",
            "value":    late_model_relative,
            "limit":    MODEL_LATE_REL_LIMIT,
            "status":   limit_status(
                late_model_relative, MODEL_LATE_REL_LIMIT, lowstat_late)},
        {
            "name":     "model curve NRMSE",
            "value":    model_curve_nrmse,
            "limit":    MODEL_CURVE_NRMSE_LIMIT,
            "status":   limit_status(
                model_curve_nrmse,
                MODEL_CURVE_NRMSE_LIMIT,
                lowstat_trajectory)},
        {
            "name":     "quantum late reference difference",
            "value":    late_quantum_reference,
            "limit":    REFERENCE_LATE_REL_LIMIT,
            "status":   limit_status(
                late_quantum_reference,
                REFERENCE_LATE_REL_LIMIT,
                lowstat_late)},
        {
            "name":     "quantum-kick late reference difference",
            "value":    late_kick_reference,
            "limit":    REFERENCE_LATE_REL_LIMIT,
            "status":   limit_status(
                late_kick_reference,
                REFERENCE_LATE_REL_LIMIT,
                lowstat_late)},
        {
            "name":     "quantum theory-curve NRMSE",
            "value":    quantum_theory_nrmse,
            "limit":    THEORY_CURVE_NRMSE_LIMIT,
            "status":   limit_status(
                quantum_theory_nrmse,
                THEORY_CURVE_NRMSE_LIMIT,
                lowstat_trajectory)},
        {
            "name":     "quantum-kick theory-curve NRMSE",
            "value":    kick_theory_nrmse,
            "limit":    THEORY_CURVE_NRMSE_LIMIT,
            "status":   limit_status(
                kick_theory_nrmse,
                THEORY_CURVE_NRMSE_LIMIT,
                lowstat_trajectory)},
        {
            "name":     "quantum trajectory correlation",
            "value":    quantum_correlation,
            "limit":    TRAJECTORY_CORRELATION_MIN,
            "status":   limit_status(
                quantum_correlation,
                TRAJECTORY_CORRELATION_MIN,
                lowstat_trajectory,
                minimum=True)},
        {
            "name":     "quantum-kick trajectory correlation",
            "value":    kick_correlation,
            "limit":    TRAJECTORY_CORRELATION_MIN,
            "status":   limit_status(
                kick_correlation,
                TRAJECTORY_CORRELATION_MIN,
                lowstat_trajectory,
                minimum=True)}]

    status = "PASS"
    if any(check["status"] == "FAIL" for check in checks):
        status = "FAIL"
    elif any(check["status"] == "LOWSTAT" for check in checks):
        status = "LOWSTAT"

    mode.update({
        "late_quantum":             late_quantum,
        "late_quantum_kick":        late_quantum_kick,
        "checks":                   checks,
        "status":                   status})

    return mode


########################################
# Validate monitor integrity
########################################
def validate_monitor_integrity(results):
    emittances = np.concatenate([
        results["gemitt_I"],
        results["gemitt_III"]])

    return {
        "survival":     (
            "PASS" if results["n_surviving"] == N_PARTICLES else "FAIL"),
        "counts":       (
            "PASS" if np.all(results["count"] == N_PARTICLES) else "FAIL"),
        "finite":       (
            "PASS" if np.all(np.isfinite(emittances)) else "FAIL"),
        "nonnegative":  (
            "PASS" if np.all(emittances >= 0) else "FAIL")}

################################################################################
# Reporting
################################################################################

########################################
# Print setup
########################################
def print_setup():
    print("\n" + "#" * 80)
    print("FCC-ee tt Emittance Comparison")
    print("#" * 80 + "\n")

    print(f"ENV_PATH                    = {ENV_PATH}")
    print(f"LINE_NAME                   = {LINE_NAME}")
    print(f"MONITOR_POINT               = {MONITOR_POINT}")
    print(f"P0C                         = {P0C:.6e} eV")
    print(f"N_PARTICLES                 = {N_PARTICLES:g}")
    print(f"N_TURNS                     = {N_TURNS:g}")
    print(f"SMOOTHING_WINDOW            = {SMOOTHING_WINDOW:g}")
    print(f"EQ_WINDOW                   = {EQ_WINDOW:g}")


########################################
# Print radiation reference
########################################
def print_radiation_reference(reference):
    print("\n" + "#" * 80)
    print("Mean-radiation Twiss Reference")
    print("#" * 80 + "\n")

    print(f"Energy loss per turn = {reference['energy_loss']:.6e} eV")
    print()
    print(
        f"{'mode':>6s}"
        f" {'equilibrium gemitt':>22s}"
        f" {'damping constant/turn':>24s}"
        f" {'amplitude damping turns':>24s}")

    for ii, mode_name in enumerate(reference["mode_names"]):
        damping = reference["damping_constants"][ii]
        print(
            f"{mode_name:>6s}"
            f" {reference['equilibrium_emittance'][ii]:22.6e}"
            f" {damping:24.6e}"
            f" {1.0 / damping:24.3f}")


########################################
# Print monitor integrity
########################################
def print_monitor_integrity(model_name, integrity):
    print(f"\n{model_name} monitor integrity")
    print("-" * (len(model_name) + 18))
    for name, status in integrity.items():
        print(f"{name:>16s}: {status}")


########################################
# Print mode result
########################################
def print_mode_result(mode):
    print("\n" + "#" * 80)
    print(f"Mode {mode['name']} Validation")
    print("#" * 80 + "\n")

    print(f"equilibrium emittance       = {mode['equilibrium']:.6e}")
    print(f"late quantum emittance      = {mode['late_quantum']:.6e}")
    print(f"late quantum-kick emittance = {mode['late_quantum_kick']:.6e}")
    print(f"effective smoothing window  = {mode['effective_window']:g} turns")
    print()
    print(
        f"{'check':>42s}"
        f" {'value':>14s}"
        f" {'limit':>14s}"
        f" {'status':>9s}")

    for check in mode["checks"]:
        print(
            f"{check['name']:>42s}"
            f" {check['value']:14.6e}"
            f" {check['limit']:14.6e}"
            f" {check['status']:>9s}")

    print()
    print(f"MODE {mode['name']} STATUS: {mode['status']}")


########################################
# Print overall status
########################################
def print_overall_status(integrity_quantum, integrity_kick, modes):
    integrity_statuses = (
        list(integrity_quantum.values()) + list(integrity_kick.values()))
    mode_statuses = [mode["status"] for mode in modes]

    print("\n" + "#" * 80)
    print("Overall Status")
    print("#" * 80 + "\n")

    if "FAIL" in integrity_statuses or "FAIL" in mode_statuses:
        overall_status = "FAIL"
    elif "LOWSTAT" in mode_statuses:
        overall_status = "LOWSTAT"
    else:
        overall_status = "PASS"

    print(f"OVERALL STATUS: {overall_status}")

    if RAISE_ON_FAIL and overall_status == "FAIL":
        raise RuntimeError("FCC emittance validation failed")

################################################################################
# Plotting
################################################################################

########################################
# Plot emittance trajectories
########################################
def plot_emittance_trajectories(modes):
    fig, axes = plt.subplots(
        len(modes), 1, figsize=(11, 3.4 * len(modes)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, modes):
        ax.plot(
            mode["turns"],
            mode["raw_quantum"],
            color="C0",
            alpha=0.18,
            linewidth=0.7)
        ax.plot(
            mode["turns"],
            mode["raw_quantum_kick"],
            color="C1",
            alpha=0.18,
            linewidth=0.7)
        ax.plot(
            mode["smooth_turns"],
            mode["smooth_quantum"],
            color="C0",
            label="quantum")
        ax.plot(
            mode["smooth_turns"],
            mode["smooth_quantum_kick"],
            color="C1",
            label="quantum-kick")
        ax.plot(
            mode["smooth_turns"],
            mode["theory_smooth"],
            color="black",
            linestyle="--",
            label="mean-radiation expectation")
        ax.axhline(
            mode["equilibrium"],
            color="0.5",
            linestyle=":",
            label="Twiss equilibrium")
        ax.set_ylabel(rf"$\epsilon_{{{mode['name']}}}$")
        ax.set_title(f"Normal mode {mode['name']}")
        ax.grid(True, alpha=0.35)
        ax.legend()

    axes[-1].set_xlabel("Turn")
    fig.suptitle("FCC-ee tt normal-mode emittance growth from zero amplitude")
    fig.tight_layout()


########################################
# Plot model residuals
########################################
def plot_model_residuals(modes):
    fig, axes = plt.subplots(
        len(modes), 1, figsize=(11, 2.8 * len(modes)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, modes):
        residual = symmetric_relative_difference(
            mode["smooth_quantum"], mode["smooth_quantum_kick"])
        theory_fraction = mode["theory_smooth"] / mode["equilibrium"]
        mask = theory_fraction >= TRAJECTORY_START_FRACTION

        ax.plot(
            mode["smooth_turns"][mask],
            residual[mask],
            color="black")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.axhline(
            MODEL_LATE_REL_LIMIT, color="0.5", linestyle="--")
        ax.axhline(
            -MODEL_LATE_REL_LIMIT, color="0.5", linestyle="--")
        ax.set_ylabel(rf"$\Delta\epsilon_{{{mode['name']}}}/\bar{{\epsilon}}$")
        ax.set_title(f"Normal mode {mode['name']}")
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("Turn")
    fig.suptitle("Smoothed quantum-kick relative to quantum")
    fig.tight_layout()

################################################################################
# Run
################################################################################

print_setup()

line_tapered, twiss = load_tapered_line()
reference = radiation_reference(twiss)
print_radiation_reference(reference)

quantum = track_model(line_tapered, "quantum")
quantum_kick = track_model(line_tapered, "quantum-kick")

integrity_quantum = validate_monitor_integrity(quantum)
integrity_kick = validate_monitor_integrity(quantum_kick)
print_monitor_integrity("quantum", integrity_quantum)
print_monitor_integrity("quantum-kick", integrity_kick)

modes = []
for mode_index, mode_name in enumerate(reference["mode_names"]):
    mode = prepare_mode_results(
        mode_name      = mode_name,
        mode_index     = mode_index,
        reference      = reference,
        quantum        = quantum,
        quantum_kick   = quantum_kick)
    mode = validate_mode(mode)
    modes.append(mode)
    print_mode_result(mode)

print_overall_status(integrity_quantum, integrity_kick, modes)

if PLOT_RESULTS:
    plt.close("all")
    plot_emittance_trajectories(modes)
    plot_model_residuals(modes)
    plt.show()
