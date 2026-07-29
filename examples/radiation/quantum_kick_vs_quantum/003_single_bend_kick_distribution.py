# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import time

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Monte Carlo settings
########################################
N_PARTICLES             = int(1E7)
BATCH_SIZE              = int(1E6)

########################################
# Single-bend case
########################################
CASE = {
    "name":     "FCC-ee tt-like high lambda",
    "p0c":      182.5e9,
    "length":   22.653765579198428,
    "angle":    0.0022799344662676477}

INITIAL_X               = 0.0
INITIAL_PX              = 1E-4
INITIAL_Y               = 0.0
INITIAL_PY              = -1E-4
INITIAL_DELTA           = 0.0

ALPHA_EM                = 7.2973525693E-3

########################################
# Acceptance limits
########################################
MEAN_Z_LIMIT            = 5.0
KS_LIMIT_FACTOR         = 1.95
TAIL_Z_LIMIT            = 5.0
MIN_TAIL_EVENTS_FOR_PASS = 20
RAISE_ON_FAIL           = False

TAIL_QUANTILES          = [0.9, 0.99, 0.999, 0.9999]

########################################
# Plot settings
########################################
PLOT_RESULTS            = True
PLOT_OBSERVABLES        = ["dpx", "dpy", "ddelta", "dloss"]

########################################
# Context
########################################
CONTEXT                 = xo.context_default

################################################################################
# Progress helpers
################################################################################

########################################
# Format duration
########################################
def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"

########################################
# Print progress line
########################################
def print_progress(label, n_done, n_total, time_start, force=False):
    time_now = time.time()
    if not force and time_now - print_progress.time_last < 2.0:
        return

    print_progress.time_last = time_now

    fraction = n_done / n_total if n_total > 0 else 1.0
    elapsed = time_now - time_start
    remaining = elapsed * (1.0 / fraction - 1.0) if fraction > 0 else 0.0

    bar_width = 28
    n_filled = int(round(bar_width * fraction))
    bar = "#" * n_filled + "." * (bar_width - n_filled)

    print(
        f"\r{label:<28s} [{bar}] "
        f"{100 * fraction:6.2f}% "
        f"{n_done:>12d}/{n_total:<12d} "
        f"elapsed {format_duration(elapsed):>7s} "
        f"left {format_duration(remaining):>7s}",
        end="",
        flush=True)

    if n_done >= n_total:
        print()


print_progress.time_last = 0.0

################################################################################
# Bend and particles
################################################################################

########################################
# Expected photon count
########################################
def expected_average_photon_count(case):
    beta0_gamma0 = case["p0c"] / xp.ELECTRON_MASS_EV
    return (
        2.5 / np.sqrt(3.0)
        * ALPHA_EM
        * beta0_gamma0
        * abs(case["angle"]))


########################################
# Build line
########################################
def build_single_bend_line(case, radiation_model):
    line = xt.Line(elements=[
        xt.Bend(
            length      = case["length"],
            angle       = case["angle"],
            k0_from_h   = True)])

    line.particle_ref = xp.Particles(
        p0c     = case["p0c"],
        mass0   = xp.ELECTRON_MASS_EV,
        q0      = -1)

    line.configure_radiation(model=radiation_model)

    return line


########################################
# Build particles
########################################
def build_particles(case, n_particles, i_start):
    particles = xp.Particles(
        _context    = CONTEXT,
        p0c         = case["p0c"],
        mass0       = xp.ELECTRON_MASS_EV,
        q0          = -1,
        particle_id = np.arange(i_start, i_start + n_particles),
        x           = INITIAL_X * np.ones(n_particles),
        px          = INITIAL_PX * np.ones(n_particles),
        y           = INITIAL_Y * np.ones(n_particles),
        py          = INITIAL_PY * np.ones(n_particles),
        delta       = INITIAL_DELTA * np.ones(n_particles))

    particles._init_random_number_generator()

    return particles


########################################
# Track one batch
########################################
def track_batch(line, case, n_particles, i_start):
    particles = build_particles(case, n_particles, i_start)

    time_start = time.perf_counter()
    line.track(particles)
    track_time = time.perf_counter() - time_start

    px      = CONTEXT.nparray_from_context_array(particles.px)
    py      = CONTEXT.nparray_from_context_array(particles.py)
    delta   = CONTEXT.nparray_from_context_array(particles.delta)

    dpx     = px - INITIAL_PX
    dpy     = py - INITIAL_PY
    ddelta  = delta - INITIAL_DELTA

    return {
        "dpx":          dpx,
        "dpy":          dpy,
        "ddelta":       ddelta,
        "dloss":        -ddelta,
        "track_time":   track_time}


################################################################################
# Sampling
################################################################################

########################################
# Allocate samples
########################################
def allocate_observable_samples():
    return {
        "dpx":      np.empty(N_PARTICLES),
        "dpy":      np.empty(N_PARTICLES),
        "ddelta":   np.empty(N_PARTICLES),
        "dloss":    np.empty(N_PARTICLES)}


########################################
# Collect one radiation mode
########################################
def collect_mode_samples(case, radiation_model):
    line = build_single_bend_line(case, radiation_model)
    samples = allocate_observable_samples()

    n_done = 0
    track_time_total = 0.0
    time_start = time.time()

    while n_done < N_PARTICLES:
        n_batch = min(BATCH_SIZE, N_PARTICLES - n_done)
        values = track_batch(
            line        = line,
            case        = case,
            n_particles = n_batch,
            i_start     = n_done)

        for key in samples:
            samples[key][n_done:n_done + n_batch] = values[key]

        track_time_total += values["track_time"]
        n_done += n_batch

        print_progress(
            label       = radiation_model,
            n_done      = n_done,
            n_total     = N_PARTICLES,
            time_start  = time_start)

    print_progress(
        label       = radiation_model,
        n_done      = N_PARTICLES,
        n_total     = N_PARTICLES,
        time_start  = time_start,
        force       = True)

    return samples, track_time_total


################################################################################
# Comparison helpers
################################################################################

########################################
# Empirical KS distance
########################################
def empirical_ks_distance(reference, candidate):
    reference_sorted = np.sort(reference)
    candidate_sorted = np.sort(candidate)
    support = np.sort(np.concatenate([reference_sorted, candidate_sorted]))

    cdf_reference = (
        np.searchsorted(reference_sorted, support, side="right")
        / reference_sorted.size)
    cdf_candidate = (
        np.searchsorted(candidate_sorted, support, side="right")
        / candidate_sorted.size)

    return np.max(np.abs(cdf_candidate - cdf_reference))


########################################
# Tail status
########################################
def tail_status(z_value, expected_ref_events):
    if expected_ref_events < MIN_TAIL_EVENTS_FOR_PASS:
        return "LOWSTAT"
    if z_value < TAIL_Z_LIMIT:
        return "PASS"
    return "FAIL"


########################################
# Tail checks
########################################
def tail_probability_checks(reference, candidate):
    checks = []

    for quantile in TAIL_QUANTILES:
        threshold = np.quantile(reference, quantile)
        p_ref = np.mean(reference > threshold)
        p_cand = np.mean(candidate > threshold)
        expected_ref_events = p_ref * reference.size

        sigma = np.sqrt(
            p_ref * (1.0 - p_ref) / reference.size
            + p_cand * (1.0 - p_cand) / candidate.size)
        z_value = abs(p_cand - p_ref) / sigma if sigma > 0 else 0.0

        checks.append({
            "quantile":     quantile,
            "threshold":    threshold,
            "events_ref":   expected_ref_events,
            "p_ref":        p_ref,
            "p_cand":       p_cand,
            "rel":          p_cand / p_ref - 1.0 if p_ref > 0 else 0.0,
            "z":            z_value,
            "status":       tail_status(z_value, expected_ref_events)})

    return checks


########################################
# Aggregate tail status
########################################
def aggregate_tail_status(tail_checks):
    statuses = [check["status"] for check in tail_checks]
    if "FAIL" in statuses:
        return "FAIL"
    if all(status == "LOWSTAT" for status in statuses):
        return "LOWSTAT"
    return "PASS"


########################################
# Compare one observable
########################################
def compare_observable(name, reference, candidate):
    mean_ref = np.mean(reference)
    mean_cand = np.mean(candidate)
    rms_ref = np.std(reference)
    rms_cand = np.std(candidate)

    mean_uncertainty = np.sqrt(
        np.var(reference) / reference.size
        + np.var(candidate) / candidate.size)
    mean_z = (
        abs(mean_cand - mean_ref) / mean_uncertainty
        if mean_uncertainty > 0 else 0.0)

    ks_distance = empirical_ks_distance(reference, candidate)
    ks_limit = KS_LIMIT_FACTOR * np.sqrt(
        (reference.size + candidate.size)
        / (reference.size * candidate.size))

    tail_checks = tail_probability_checks(reference, candidate)
    tail_status_all = aggregate_tail_status(tail_checks)
    active_tail_z = [
        check["z"] for check in tail_checks
        if check["status"] != "LOWSTAT"]
    max_tail_z = max(active_tail_z) if active_tail_z else 0.0

    mean_status = "PASS" if mean_z < MEAN_Z_LIMIT else "FAIL"
    ks_status = "PASS" if ks_distance < ks_limit else "FAIL"
    overall_status = (
        "PASS"
        if mean_status == "PASS"
        and ks_status == "PASS"
        and tail_status_all != "FAIL"
        else "FAIL")

    return {
        "name":         name,
        "mean_ref":     mean_ref,
        "mean_cand":    mean_cand,
        "mean_rel":     (
            mean_cand / mean_ref - 1.0
            if abs(mean_ref) > np.finfo(float).tiny else 0.0),
        "mean_z":       mean_z,
        "rms_ref":      rms_ref,
        "rms_cand":     rms_cand,
        "rms_rel":      (
            rms_cand / rms_ref - 1.0
            if abs(rms_ref) > np.finfo(float).tiny else 0.0),
        "ks":           ks_distance,
        "ks_limit":     ks_limit,
        "tail_checks":  tail_checks,
        "max_tail_z":   max_tail_z,
        "mean_status":  mean_status,
        "ks_status":    ks_status,
        "tail_status":  tail_status_all,
        "status":       overall_status}


################################################################################
# Reporting
################################################################################

########################################
# Print setup
########################################
def print_setup():
    print("\n" + "#" * 80)
    print("Single-Bend Kick Distribution")
    print("#" * 80 + "\n")

    print(f"case name           = {CASE['name']}")
    print(f"p0c                 = {CASE['p0c']:.6e} eV")
    print(f"length              = {CASE['length']:.12e} m")
    print(f"angle               = {CASE['angle']:.12e} rad")
    print(f"expected <N_gamma>  = {expected_average_photon_count(CASE):.6e}")
    print(f"N_PARTICLES         = {N_PARTICLES:g}")
    print(f"BATCH_SIZE          = {BATCH_SIZE:g}")
    print(f"TAIL_QUANTILES      = {TAIL_QUANTILES}")
    print(f"MEAN_Z_LIMIT        = {MEAN_Z_LIMIT:g}")
    print(f"KS_LIMIT_FACTOR     = {KS_LIMIT_FACTOR:g}")
    print(f"TAIL_Z_LIMIT        = {TAIL_Z_LIMIT:g}")


########################################
# Print mode summary
########################################
def print_mode_summary(mode_name, samples, track_time_total):
    print()
    print(f"{mode_name} summary")
    print("-" * (len(mode_name) + 8))

    print(f"track time total    = {track_time_total:.6e} s")
    print(f"tracking rate       = {N_PARTICLES / track_time_total:.6e} particles/s")

    for key in ["dpx", "dpy", "ddelta", "dloss"]:
        values = samples[key]
        print(
            f"{key:8s}"
            f" mean={np.mean(values): .6e}"
            f" rms={np.std(values): .6e}"
            f" min={np.min(values): .6e}"
            f" max={np.max(values): .6e}")


########################################
# Print comparison table
########################################
def print_comparison_table(results):
    print("\n" + "#" * 80)
    print("Quantum versus Quantum-Kick")
    print("#" * 80 + "\n")

    print(
        f"{'observable':>10s}"
        f" {'mean rel':>11s}"
        f" {'mean z':>8s}"
        f" {'mean':>6s}"
        f" {'rms rel':>11s}"
        f" {'KS stat':>10s}"
        f" {'KS lim':>10s}"
        f" {'KS':>6s}"
        f" {'max tail z':>11s}"
        f" {'tail':>8s}"
        f" {'overall':>8s}")

    for result in results:
        print(
            f"{result['name']:>10s}"
            f" {result['mean_rel']:+11.4e}"
            f" {result['mean_z']:8.3f}"
            f" {result['mean_status']:>6s}"
            f" {result['rms_rel']:+11.4e}"
            f" {result['ks']:10.4e}"
            f" {result['ks_limit']:10.4e}"
            f" {result['ks_status']:>6s}"
            f" {result['max_tail_z']:11.3f}"
            f" {result['tail_status']:>8s}"
            f" {result['status']:>8s}")


########################################
# Print tail table
########################################
def print_tail_table(results):
    print("\nTail probability checks")
    print("-----------------------")

    print(
        f"{'observable':>10s}"
        f" {'q':>9s}"
        f" {'threshold':>14s}"
        f" {'events':>8s}"
        f" {'P quantum':>11s}"
        f" {'P kick':>11s}"
        f" {'P rel':>11s}"
        f" {'z':>8s}"
        f" {'status':>8s}")

    for result in results:
        for check in result["tail_checks"]:
            print(
                f"{result['name']:>10s}"
                f" {check['quantile']:9.5f}"
                f" {check['threshold']:14.6e}"
                f" {check['events_ref']:8.1f}"
                f" {check['p_ref']:11.4e}"
                f" {check['p_cand']:11.4e}"
                f" {check['rel']:+11.4e}"
                f" {check['z']:8.3f}"
                f" {check['status']:>8s}")


########################################
# Print overall status
########################################
def print_overall_status(results):
    failed = [result for result in results if result["status"] != "PASS"]

    print("\n" + "#" * 80)
    print("Overall Status")
    print("#" * 80 + "\n")

    if len(failed) == 0:
        print("OVERALL STATUS: PASS")
    else:
        print("OVERALL STATUS: FAIL")
        print()
        print("Failing observables:")
        for result in failed:
            print(
                f"  {result['name']}: "
                f"mean={result['mean_status']}, "
                f"KS={result['ks_status']}, "
                f"tail={result['tail_status']}")

    if RAISE_ON_FAIL and len(failed) > 0:
        raise RuntimeError("Single-bend kick validation failed")


################################################################################
# Plotting
################################################################################

########################################
# CDF residual grid
########################################
def cdf_residual_grid(reference, candidate):
    combined = np.concatenate([reference, candidate])
    probabilities = np.linspace(1E-4, 1.0 - 1E-4, 800)
    x_grid = np.quantile(combined, probabilities)

    reference_sorted = np.sort(reference)
    candidate_sorted = np.sort(candidate)

    cdf_reference = (
        np.searchsorted(reference_sorted, x_grid, side="right")
        / reference_sorted.size)
    cdf_candidate = (
        np.searchsorted(candidate_sorted, x_grid, side="right")
        / candidate_sorted.size)

    return x_grid, cdf_reference, cdf_candidate, cdf_candidate - cdf_reference


########################################
# Plot comparisons
########################################
def plot_comparisons(samples_quantum, samples_quantum_kick, results):
    observables = [
        key for key in PLOT_OBSERVABLES
        if key in samples_quantum and key in samples_quantum_kick]

    fig, axes = plt.subplots(
        3,
        len(observables),
        figsize=(4.2 * len(observables), 9.0),
        gridspec_kw={"height_ratios": [2, 2, 1]})

    if len(observables) == 1:
        axes = axes.reshape(3, 1)

    result_by_name = {result["name"]: result for result in results}

    for ii, key in enumerate(observables):
        reference = samples_quantum[key]
        candidate = samples_quantum_kick[key]

        ax_hist = axes[0, ii]
        ax_cdf = axes[1, ii]
        ax_res = axes[2, ii]

        lo, hi = np.quantile(
            np.concatenate([reference, candidate]),
            [1E-4, 1.0 - 1E-4])
        bins = np.linspace(lo, hi, 250)

        ax_hist.hist(
            reference, bins=bins, density=True, histtype="step",
            linewidth=1.3, label="quantum")
        ax_hist.hist(
            candidate, bins=bins, density=True, histtype="step",
            linewidth=1.1, label="quantum-kick")
        ax_hist.set_title(key)
        ax_hist.set_ylabel("Density")
        ax_hist.grid(True)
        ax_hist.legend()

        x_grid, cdf_reference, cdf_candidate, residual = cdf_residual_grid(
            reference, candidate)

        ax_cdf.plot(x_grid, cdf_reference, label="quantum", linewidth=1.3)
        ax_cdf.plot(x_grid, cdf_candidate, label="quantum-kick", linewidth=1.1)
        ax_cdf.set_ylabel("CDF")
        ax_cdf.grid(True)

        ax_res.plot(x_grid, residual, color="black", linewidth=1.1)
        ax_res.axhline(0.0, color="0.5", linewidth=0.8)
        ax_res.axhline(
            result_by_name[key]["ks_limit"], color="0.6", linestyle="--")
        ax_res.axhline(
            -result_by_name[key]["ks_limit"], color="0.6", linestyle="--")
        ax_res.set_xlabel(key)
        ax_res.set_ylabel("CDF diff.")
        ax_res.grid(True)

        residual_max = max(
            np.max(np.abs(residual)), result_by_name[key]["ks_limit"])
        ax_res.set_ylim(-1.15 * residual_max, 1.15 * residual_max)

    fig.suptitle(
        f"{CASE['name']}: quantum versus quantum-kick "
        f"({N_PARTICLES:g} particles)")
    fig.tight_layout()


################################################################################
# Run
################################################################################

print_setup()

samples_quantum, time_quantum = collect_mode_samples(CASE, "quantum")
samples_quantum_kick, time_quantum_kick = collect_mode_samples(
    CASE, "quantum-kick")

print_mode_summary("quantum", samples_quantum, time_quantum)
print_mode_summary("quantum-kick", samples_quantum_kick, time_quantum_kick)

results = []
for observable in ["dpx", "dpy", "ddelta", "dloss"]:
    results.append(compare_observable(
        name        = observable,
        reference   = samples_quantum[observable],
        candidate   = samples_quantum_kick[observable]))

print_comparison_table(results)
print_tail_table(results)
print_overall_status(results)

if PLOT_RESULTS:
    plt.close("all")
    plot_comparisons(samples_quantum, samples_quantum_kick, results)
    plt.show()
