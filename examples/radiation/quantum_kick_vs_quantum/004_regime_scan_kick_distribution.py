# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

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
# Lambda scan
########################################
P0C                     = 182.5E9
BEND_LENGTH             = 22.653765579198428
TARGET_LAMBDAS          = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 9.0, 30.0]

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
PLOT_LAMBDAS            = [0.01, 0.3, 3.0, 30.0]

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
        f"\r{label:<30s} [{bar}] "
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
# Scan setup
################################################################################

########################################
# Angle from target lambda
########################################
def bend_angle_from_lambda(target_lambda):
    beta0_gamma0 = P0C / xp.ELECTRON_MASS_EV
    coefficient = 2.5 / np.sqrt(3.0) * ALPHA_EM * beta0_gamma0
    return target_lambda / coefficient


########################################
# Build cases
########################################
def build_scan_cases():
    cases = []
    for target_lambda in TARGET_LAMBDAS:
        angle = bend_angle_from_lambda(target_lambda)
        cases.append({
            "name":             f"lambda={target_lambda:g}",
            "target_lambda":    target_lambda,
            "p0c":              P0C,
            "length":           BEND_LENGTH,
            "angle":            angle})
    return cases


########################################
# Expected photon fractions
########################################
def poisson_probabilities_up_to(lam, n_max):
    probabilities = np.empty(n_max + 1)
    probabilities[0] = np.exp(-lam)
    for nn in range(1, n_max + 1):
        probabilities[nn] = probabilities[nn - 1] * lam / nn
    return probabilities


def photon_count_fractions(lam):
    probabilities = poisson_probabilities_up_to(lam, 32)
    p0 = probabilities[0]
    p1 = probabilities[1]
    p_direct = np.sum(probabilities[1:33])
    p_decomposed = max(0.0, 1.0 - np.sum(probabilities))
    return p0, p1, p_direct, p_decomposed


################################################################################
# Bend and particles
################################################################################

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
            label       = f"{case['name']} {radiation_model}",
            n_done      = n_done,
            n_total     = N_PARTICLES,
            time_start  = time_start)

    print_progress(
        label       = f"{case['name']} {radiation_model}",
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
    return ks_2samp(reference, candidate).statistic


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
        "ks_ratio":     ks_distance / ks_limit if ks_limit > 0 else np.nan,
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
def print_setup(cases):
    print("\n" + "#" * 80)
    print("Regime Scan Kick Distribution")
    print("#" * 80 + "\n")

    print(f"p0c                 = {P0C:.6e} eV")
    print(f"bend length         = {BEND_LENGTH:.12e} m")
    print(f"N_PARTICLES         = {N_PARTICLES:g}")
    print(f"BATCH_SIZE          = {BATCH_SIZE:g}")
    print(f"TAIL_QUANTILES      = {TAIL_QUANTILES}")
    print(f"MEAN_Z_LIMIT        = {MEAN_Z_LIMIT:g}")
    print(f"KS_LIMIT_FACTOR     = {KS_LIMIT_FACTOR:g}")
    print(f"TAIL_Z_LIMIT        = {TAIL_Z_LIMIT:g}")

    print()
    print("Scan cases")
    print("----------")
    print(
        f"{'lambda':>10s}"
        f" {'angle [rad]':>14s}"
        f" {'P0':>10s}"
        f" {'P1':>10s}"
        f" {'P direct':>10s}"
        f" {'P decomp':>10s}")
    for case in cases:
        p0, p1, p_direct, p_decomp = photon_count_fractions(
            case["target_lambda"])
        print(
            f"{case['target_lambda']:10.4g}"
            f" {case['angle']:14.6e}"
            f" {p0:10.4e}"
            f" {p1:10.4e}"
            f" {p_direct:10.4e}"
            f" {p_decomp:10.4e}")


########################################
# Print case summary
########################################
def print_case_summary(case, time_quantum, time_quantum_kick, results):
    p0, p1, p_direct, p_decomp = photon_count_fractions(case["target_lambda"])

    print("\n" + "#" * 80)
    print(f"Case summary: lambda = {case['target_lambda']:g}")
    print("#" * 80 + "\n")

    print(f"angle               = {case['angle']:.12e} rad")
    print(f"P(N=0)              = {p0:.6e}")
    print(f"P(N=1)              = {p1:.6e}")
    print(f"P(1<=N<=32)         = {p_direct:.6e}")
    print(f"P(N>32)             = {p_decomp:.6e}")
    print(f"quantum track time  = {time_quantum:.6e} s")
    print(f"kick track time     = {time_quantum_kick:.6e} s")
    print(f"speed ratio q/qkick = {time_quantum / time_quantum_kick:.6e}")

    print()
    print(
        f"{'observable':>10s}"
        f" {'mean rel':>11s}"
        f" {'mean z':>8s}"
        f" {'mean':>6s}"
        f" {'rms rel':>11s}"
        f" {'KS/lim':>10s}"
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
            f" {result['ks_ratio']:10.4e}"
            f" {result['ks_status']:>6s}"
            f" {result['max_tail_z']:11.3f}"
            f" {result['tail_status']:>8s}"
            f" {result['status']:>8s}")


########################################
# Print compact scan summary
########################################
def print_scan_summary(scan_results):
    print("\n" + "#" * 80)
    print("Compact Scan Summary")
    print("#" * 80 + "\n")

    print(
        f"{'lambda':>10s}"
        f" {'P0':>10s}"
        f" {'P direct':>10s}"
        f" {'P decomp':>10s}"
        f" {'mean z':>8s}"
        f" {'KS/lim':>10s}"
        f" {'tail z':>8s}"
        f" {'status':>8s}"
        f" {'speedup':>9s}")

    for item in scan_results:
        dloss_result = item["results_by_name"]["dloss"]
        p0, _p1, p_direct, p_decomp = photon_count_fractions(item["lambda"])
        print(
            f"{item['lambda']:10.4g}"
            f" {p0:10.4e}"
            f" {p_direct:10.4e}"
            f" {p_decomp:10.4e}"
            f" {dloss_result['mean_z']:8.3f}"
            f" {dloss_result['ks_ratio']:10.4e}"
            f" {dloss_result['max_tail_z']:8.3f}"
            f" {item['status']:>8s}"
            f" {item['speedup']:9.3f}")


########################################
# Print overall status
########################################
def print_overall_status(scan_results):
    failed = [item for item in scan_results if item["status"] != "PASS"]

    print("\n" + "#" * 80)
    print("Overall Status")
    print("#" * 80 + "\n")

    if len(failed) == 0:
        print("OVERALL STATUS: PASS")
    else:
        print("OVERALL STATUS: FAIL")
        print()
        print("Failing cases:")
        for item in failed:
            print(f"  lambda={item['lambda']:g}: {item['status']}")

    if RAISE_ON_FAIL and len(failed) > 0:
        raise RuntimeError("Regime scan validation failed")


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
# Plot scan metrics
########################################
def plot_scan_metrics(scan_results):
    lambdas = np.array([item["lambda"] for item in scan_results])
    mean_z = np.array([
        item["results_by_name"]["dloss"]["mean_z"]
        for item in scan_results])
    ks_ratio = np.array([
        item["results_by_name"]["dloss"]["ks_ratio"]
        for item in scan_results])
    tail_z = np.array([
        item["results_by_name"]["dloss"]["max_tail_z"]
        for item in scan_results])
    speedup = np.array([item["speedup"] for item in scan_results])

    p0 = []
    p1 = []
    p_direct = []
    p_decomp = []
    for lam in lambdas:
        this_p0, this_p1, this_p_direct, this_p_decomp = (
            photon_count_fractions(lam))
        p0.append(this_p0)
        p1.append(this_p1)
        p_direct.append(this_p_direct)
        p_decomp.append(this_p_decomp)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_frac, ax_mean, ax_ks, ax_speed = axes.ravel()

    ax_frac.semilogx(lambdas, p0, "o-", label="P(N=0)")
    ax_frac.semilogx(lambdas, p1, "o-", label="P(N=1)")
    ax_frac.semilogx(lambdas, p_direct, "o-", label="P(1<=N<=32)")
    ax_frac.semilogx(lambdas, p_decomp, "o-", label="P(N>32)")
    ax_frac.set_ylabel("Probability")
    ax_frac.set_title("Photon-count regimes")
    ax_frac.grid(True)
    ax_frac.legend()

    ax_mean.semilogx(lambdas, mean_z, "o-", label="mean z")
    ax_mean.semilogx(lambdas, tail_z, "s-", label="max tail z")
    ax_mean.axhline(MEAN_Z_LIMIT, color="0.5", linestyle="--")
    ax_mean.axhline(TAIL_Z_LIMIT, color="0.5", linestyle=":")
    ax_mean.set_ylabel("z")
    ax_mean.set_title("dloss mean and tail checks")
    ax_mean.grid(True)
    ax_mean.legend()

    ax_ks.semilogx(lambdas, ks_ratio, "o-")
    ax_ks.axhline(1.0, color="0.5", linestyle="--")
    ax_ks.set_xlabel(r"Target $\lambda=\langle N_\gamma\rangle$")
    ax_ks.set_ylabel("KS / limit")
    ax_ks.set_title("dloss CDF agreement")
    ax_ks.grid(True)

    ax_speed.semilogx(lambdas, speedup, "o-")
    ax_speed.set_xlabel(r"Target $\lambda=\langle N_\gamma\rangle$")
    ax_speed.set_ylabel("quantum time / quantum-kick time")
    ax_speed.set_title("Tracking-time diagnostic")
    ax_speed.grid(True)

    fig.tight_layout()


########################################
# Plot selected CDFs
########################################
def plot_selected_cdfs(plot_samples, scan_results):
    selected_lambdas = [lam for lam in PLOT_LAMBDAS if lam in plot_samples]
    if len(selected_lambdas) == 0:
        return

    fig, axes = plt.subplots(
        2,
        len(selected_lambdas),
        figsize=(4.2 * len(selected_lambdas), 6.0),
        gridspec_kw={"height_ratios": [3, 1]})

    if len(selected_lambdas) == 1:
        axes = axes.reshape(2, 1)

    result_by_lambda = {
        item["lambda"]: item["results_by_name"]["dloss"]
        for item in scan_results}

    for ii, lam in enumerate(selected_lambdas):
        reference = plot_samples[lam]["quantum"]
        candidate = plot_samples[lam]["quantum-kick"]

        ax_cdf = axes[0, ii]
        ax_res = axes[1, ii]

        x_grid, cdf_reference, cdf_candidate, residual = cdf_residual_grid(
            reference, candidate)

        ax_cdf.plot(x_grid, cdf_reference, label="quantum", linewidth=1.3)
        ax_cdf.plot(x_grid, cdf_candidate, label="quantum-kick", linewidth=1.1)
        ax_cdf.set_title(rf"$\lambda={lam:g}$")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.grid(True)
        ax_cdf.legend()

        ax_res.plot(x_grid, residual, color="black", linewidth=1.1)
        ax_res.axhline(0.0, color="0.5", linewidth=0.8)
        ax_res.axhline(
            result_by_lambda[lam]["ks_limit"], color="0.6", linestyle="--")
        ax_res.axhline(
            -result_by_lambda[lam]["ks_limit"], color="0.6", linestyle="--")
        ax_res.set_xlabel("Total normalized energy loss")
        ax_res.set_ylabel("CDF diff.")
        ax_res.grid(True)

        residual_max = max(
            np.max(np.abs(residual)),
            result_by_lambda[lam]["ks_limit"])
        ax_res.set_ylim(-1.15 * residual_max, 1.15 * residual_max)

    fig.suptitle("dloss CDF comparisons across selected regimes")
    fig.tight_layout()


################################################################################
# Run
################################################################################

cases = build_scan_cases()
print_setup(cases)

scan_results = []
plot_samples = {}

for ii_case, case in enumerate(cases):
    print("\n" + "#" * 80)
    print(
        f"Running case {ii_case + 1}/{len(cases)}: "
        f"lambda = {case['target_lambda']:g}")
    print("#" * 80 + "\n")

    samples_quantum, time_quantum = collect_mode_samples(case, "quantum")
    samples_quantum_kick, time_quantum_kick = collect_mode_samples(
        case, "quantum-kick")

    results = []
    for observable in ["dpx", "dpy", "ddelta", "dloss"]:
        results.append(compare_observable(
            name        = observable,
            reference   = samples_quantum[observable],
            candidate   = samples_quantum_kick[observable]))

    print_case_summary(case, time_quantum, time_quantum_kick, results)

    status = "PASS"
    if any(result["status"] != "PASS" for result in results):
        status = "FAIL"

    if PLOT_RESULTS and case["target_lambda"] in PLOT_LAMBDAS:
        plot_samples[case["target_lambda"]] = {
            "quantum":       samples_quantum["dloss"].copy(),
            "quantum-kick":  samples_quantum_kick["dloss"].copy()}

    scan_results.append({
        "lambda":           case["target_lambda"],
        "angle":            case["angle"],
        "time_quantum":     time_quantum,
        "time_quantum_kick": time_quantum_kick,
        "speedup":          time_quantum / time_quantum_kick,
        "results":          results,
        "results_by_name":  {result["name"]: result for result in results},
        "status":           status})

print_scan_summary(scan_results)
print_overall_status(scan_results)

if PLOT_RESULTS:
    plt.close("all")
    plot_scan_metrics(scan_results)
    plot_selected_cdfs(plot_samples, scan_results)
    plt.show()
