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
from scipy.stats import poisson

################################################################################
# User parameters
################################################################################

########################################
# Monte Carlo settings
########################################
SEED                    = 12345
N_EVENTS                = int(1E7)
BATCH_SIZE              = int(1E6)

########################################
# Photon-count regimes
########################################
TARGET_LAMBDAS          = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 9.0, 30.0]

DIRECT_TABLE_MAX        = 32
POWER2_TABLE_MAX        = 256

########################################
# Fixed-count inspection
########################################
MAX_FIXED_COUNT_CHECK   = 1024
MAX_FIXED_COUNT_PLOT    = 65
COUNTS_TO_PRINT         = [
    0, 1, 2, 3, 8, 16, 31, 32, 33, 35, 47, 63, 64, 65]

########################################
# Acceptance limits
########################################
MC_Z_LIMIT              = 5.0
ANALYTIC_TAIL_LIMIT     = 1E-14
RAISE_ON_FAIL           = False

########################################
# Plot settings
########################################
PLOT_RESULTS            = True

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
        f"\r{label:<24s} [{bar}] "
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
# Production decomposition model
################################################################################

########################################
# Largest available power-of-two chunk
########################################
def largest_power_of_two_chunk(n_photons):
    chunk = 1
    while chunk <= n_photons // 2:
        chunk *= 2
    return min(chunk, POWER2_TABLE_MAX)


########################################
# Decompose one fixed photon count
########################################
def decompose_photon_count(n_photons):
    n_left = int(n_photons)
    power2_chunks = []

    while n_left > DIRECT_TABLE_MAX:
        chunk = largest_power_of_two_chunk(n_left)
        power2_chunks.append(chunk)
        n_left -= chunk

    direct_chunk = n_left
    return power2_chunks, direct_chunk


########################################
# Count lookups for fixed counts
########################################
def lookup_counts_fixed(n_photons):
    power2_chunks, direct_chunk = decompose_photon_count(n_photons)
    n_power2 = len(power2_chunks)
    n_direct = int(direct_chunk > 0)
    return n_power2 + n_direct, n_power2, n_direct


########################################
# Count lookups for arrays
########################################
def lookup_counts_array(n_photons):
    n_left = np.asarray(n_photons, dtype=np.int64).copy()
    n_power2 = np.zeros(n_left.shape, dtype=np.int64)

    while np.any(n_left > DIRECT_TABLE_MAX):
        active = n_left > DIRECT_TABLE_MAX
        values = n_left[active]

        chunks = np.full(values.shape, DIRECT_TABLE_MAX, dtype=np.int64)
        chunks[values >= 64] = 64
        chunks[values >= 128] = 128
        chunks[values >= POWER2_TABLE_MAX] = POWER2_TABLE_MAX

        n_left[active] -= chunks
        n_power2[active] += 1

    n_direct = (n_left > 0).astype(np.int64)
    return n_power2 + n_direct, n_power2, n_direct

################################################################################
# Fixed-count checks
################################################################################

########################################
# Validate decomposition rules
########################################
def validate_fixed_count_decomposition():
    failures = []
    fixed_counts = np.arange(MAX_FIXED_COUNT_CHECK + 1, dtype=np.int64)
    array_lookup, array_power2, array_direct = lookup_counts_array(fixed_counts)

    for n_photons in range(MAX_FIXED_COUNT_CHECK + 1):
        power2_chunks, direct_chunk = decompose_photon_count(n_photons)
        n_lookup, n_power2, n_direct = lookup_counts_fixed(n_photons)

        reconstructed = sum(power2_chunks) + direct_chunk
        valid_power2 = all(
            chunk in [32, 64, 128, 256] for chunk in power2_chunks)

        checks = {
            "reconstructed count": reconstructed == n_photons,
            "valid power2 chunks": valid_power2,
            "direct range": 0 <= direct_chunk <= DIRECT_TABLE_MAX,
            "lookup accounting": n_lookup == n_power2 + n_direct,
            "direct accounting": n_direct == int(direct_chunk > 0),
            "vectorized total lookup count": (
                array_lookup[n_photons] == n_lookup),
            "vectorized power2 lookup count": (
                array_power2[n_photons] == n_power2),
            "vectorized direct lookup count": (
                array_direct[n_photons] == n_direct),
            "no more lookups than photons": (
                n_lookup <= n_photons if n_photons > 0 else n_lookup == 0),
            "one direct lookup through 32": (
                n_lookup == 1 and n_direct == 1 and n_power2 == 0
                if 1 <= n_photons <= DIRECT_TABLE_MAX else True)}

        for check_name, passed in checks.items():
            if not passed:
                failures.append((n_photons, check_name))

    return failures


########################################
# Print selected decompositions
########################################
def print_fixed_count_decompositions():
    print("\n" + "#" * 80)
    print("Fixed Photon-count Decomposition")
    print("#" * 80 + "\n")

    print(
        f"{'N photons':>10s}"
        f" {'power2 chunks':>28s}"
        f" {'direct':>8s}"
        f" {'lookups':>8s}"
        f" {'reduction':>11s}")

    for n_photons in COUNTS_TO_PRINT:
        power2_chunks, direct_chunk = decompose_photon_count(n_photons)
        n_lookup, _n_power2, _n_direct = lookup_counts_fixed(n_photons)
        reduction = n_photons / n_lookup if n_lookup > 0 else np.nan
        chunks_text = str(power2_chunks)
        direct_text = str(direct_chunk) if direct_chunk > 0 else "-"
        reduction_text = f"{reduction:.3f}" if n_lookup > 0 else "-"

        print(
            f"{n_photons:10d}"
            f" {chunks_text:>28s}"
            f" {direct_text:>8s}"
            f" {n_lookup:8d}"
            f" {reduction_text:>11s}")


########################################
# Print fixed-count status
########################################
def print_fixed_count_status(failures):
    print()
    if len(failures) == 0:
        print(
            f"FIXED-COUNT STATUS: PASS "
            f"(all counts 0..{MAX_FIXED_COUNT_CHECK} satisfy the production rules)")
        return "PASS"

    print("FIXED-COUNT STATUS: FAIL")
    for n_photons, check_name in failures:
        print(f"  N={n_photons}: {check_name}")
    return "FAIL"

################################################################################
# Analytic Poisson expectations
################################################################################

########################################
# Truncated Poisson support
########################################
def poisson_support(lam):
    n_max = max(64, int(np.ceil(lam + 12.0 * np.sqrt(lam + 1.0))))
    while poisson.sf(n_max, lam) > ANALYTIC_TAIL_LIMIT:
        n_max += 16

    counts = np.arange(n_max + 1, dtype=np.int64)
    probabilities = poisson.pmf(counts, lam)
    omitted_probability = poisson.sf(n_max, lam)
    return counts, probabilities, omitted_probability


########################################
# Metric values from photon counts
########################################
def metric_values_from_counts(n_photons):
    n_lookup, n_power2, n_direct = lookup_counts_array(n_photons)
    return {
        "photon_samples":   n_photons,
        "table_lookups":    n_lookup,
        "power2_lookups":   n_power2,
        "direct_lookups":   n_direct,
        "zero_fraction":    n_photons == 0,
        "direct_fraction":  (
            (n_photons >= 1) & (n_photons <= DIRECT_TABLE_MAX)),
        "decomp_fraction":  n_photons > DIRECT_TABLE_MAX}


########################################
# Analytic moments
########################################
def analytic_moments(lam):
    counts, probabilities, omitted_probability = poisson_support(lam)
    values_by_name = metric_values_from_counts(counts)

    moments = {}
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=float)
        mean = np.sum(probabilities * values)
        second = np.sum(probabilities * values * values)
        moments[name] = {
            "mean": mean,
            "variance": max(0.0, second - mean * mean)}

    moments["photon_samples"]["mean"] = lam
    moments["photon_samples"]["variance"] = lam

    return moments, omitted_probability

################################################################################
# Monte Carlo
################################################################################

########################################
# Empty accumulators
########################################
def empty_accumulators():
    names = [
        "photon_samples",
        "table_lookups",
        "power2_lookups",
        "direct_lookups",
        "zero_fraction",
        "direct_fraction",
        "decomp_fraction"]
    return {
        name: {"sum": 0.0, "sum2": 0.0}
        for name in names}


########################################
# Accumulate one batch
########################################
def accumulate_batch(accumulators, values_by_name):
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=float)
        accumulators[name]["sum"] += np.sum(values)
        accumulators[name]["sum2"] += np.sum(values * values)


########################################
# Run one lambda
########################################
def run_monte_carlo_lambda(rng, lam):
    accumulators = empty_accumulators()
    n_done = 0
    time_start = time.time()

    while n_done < N_EVENTS:
        n_batch = min(BATCH_SIZE, N_EVENTS - n_done)
        n_photons = rng.poisson(lam, size=n_batch)
        values_by_name = metric_values_from_counts(n_photons)
        accumulate_batch(accumulators, values_by_name)
        n_done += n_batch

        print_progress(
            label       = f"lambda={lam:g}",
            n_done      = n_done,
            n_total     = N_EVENTS,
            time_start  = time_start)

    print_progress(
        label       = f"lambda={lam:g}",
        n_done      = N_EVENTS,
        n_total     = N_EVENTS,
        time_start  = time_start,
        force       = True)

    sample_moments = {}
    for name, values in accumulators.items():
        mean = values["sum"] / N_EVENTS
        second = values["sum2"] / N_EVENTS
        sample_moments[name] = {
            "mean": mean,
            "variance": max(0.0, second - mean * mean)}

    return sample_moments

################################################################################
# Statistical comparison
################################################################################

########################################
# Compare one metric
########################################
def compare_metric(name, analytic, sampled):
    standard_error = np.sqrt(analytic["variance"] / N_EVENTS)
    difference = sampled["mean"] - analytic["mean"]

    if standard_error > 0:
        z_value = abs(difference) / standard_error
    else:
        z_value = 0.0 if difference == 0 else np.inf

    status = "PASS" if z_value < MC_Z_LIMIT else "FAIL"
    return {
        "name":             name,
        "analytic":         analytic["mean"],
        "sampled":          sampled["mean"],
        "difference":       difference,
        "standard_error":   standard_error,
        "z":                z_value,
        "status":           status}


########################################
# Compare one lambda
########################################
def compare_lambda(lam, analytic, sampled, omitted_probability):
    comparisons = []
    for name in analytic:
        comparisons.append(compare_metric(
            name        = name,
            analytic    = analytic[name],
            sampled     = sampled[name]))

    tail_status = (
        "PASS" if omitted_probability <= ANALYTIC_TAIL_LIMIT else "FAIL")
    status = "PASS"
    if tail_status == "FAIL" or any(
            comparison["status"] == "FAIL" for comparison in comparisons):
        status = "FAIL"

    return {
        "lambda":                   lam,
        "analytic":                 analytic,
        "sampled":                  sampled,
        "comparisons":              comparisons,
        "comparisons_by_name":      {
            comparison["name"]: comparison for comparison in comparisons},
        "omitted_probability":       omitted_probability,
        "tail_status":               tail_status,
        "status":                    status}

################################################################################
# Reporting
################################################################################

########################################
# Print setup
########################################
def print_setup():
    print("\n" + "#" * 80)
    print("Sampler Call Counts")
    print("#" * 80 + "\n")

    print(f"N_EVENTS             = {N_EVENTS:g}")
    print(f"BATCH_SIZE           = {BATCH_SIZE:g}")
    print(f"TARGET_LAMBDAS       = {TARGET_LAMBDAS}")
    print(f"DIRECT_TABLE_MAX     = {DIRECT_TABLE_MAX}")
    print(f"POWER2_TABLE_MAX     = {POWER2_TABLE_MAX}")
    print(f"MC_Z_LIMIT           = {MC_Z_LIMIT:g}")
    print(f"ANALYTIC_TAIL_LIMIT  = {ANALYTIC_TAIL_LIMIT:.1e}")


########################################
# Print one lambda
########################################
def print_lambda_result(result):
    print("\n" + "#" * 80)
    print(f"Lambda = {result['lambda']:g}")
    print("#" * 80 + "\n")

    print(
        f"Analytic omitted probability = "
        f"{result['omitted_probability']:.6e}  "
        f"[{result['tail_status']}]")
    print()
    print(
        f"{'metric':>18s}"
        f" {'analytic':>14s}"
        f" {'Monte Carlo':>14s}"
        f" {'difference':>14s}"
        f" {'MC s.e.':>12s}"
        f" {'z':>8s}"
        f" {'status':>8s}")

    for comparison in result["comparisons"]:
        print(
            f"{comparison['name']:>18s}"
            f" {comparison['analytic']:14.6e}"
            f" {comparison['sampled']:14.6e}"
            f" {comparison['difference']:+14.6e}"
            f" {comparison['standard_error']:12.4e}"
            f" {comparison['z']:8.3f}"
            f" {comparison['status']:>8s}")

    expected_photons = result["analytic"]["photon_samples"]["mean"]
    expected_lookups = result["analytic"]["table_lookups"]["mean"]
    reduction = (
        expected_photons / expected_lookups
        if expected_lookups > 0 else np.nan)

    print()
    print(f"Expected photon samples/event = {expected_photons:.6e}")
    print(f"Expected table lookups/event  = {expected_lookups:.6e}")
    print(f"Energy-sampler call reduction = {reduction:.6e}")
    print(f"LAMBDA STATUS: {result['status']}")


########################################
# Print compact summary
########################################
def print_compact_summary(results):
    print("\n" + "#" * 80)
    print("Compact Call-count Summary")
    print("#" * 80 + "\n")

    print(
        f"{'lambda':>9s}"
        f" {'photons':>11s}"
        f" {'lookups':>11s}"
        f" {'direct':>11s}"
        f" {'power2':>11s}"
        f" {'reduction':>11s}"
        f" {'P(1..32)':>11s}"
        f" {'P decomp':>11s}"
        f" {'status':>8s}")

    for result in results:
        analytic = result["analytic"]
        photons = analytic["photon_samples"]["mean"]
        lookups = analytic["table_lookups"]["mean"]
        reduction = photons / lookups if lookups > 0 else np.nan

        print(
            f"{result['lambda']:9.4g}"
            f" {photons:11.4e}"
            f" {lookups:11.4e}"
            f" {analytic['direct_lookups']['mean']:11.4e}"
            f" {analytic['power2_lookups']['mean']:11.4e}"
            f" {reduction:11.4f}"
            f" {analytic['direct_fraction']['mean']:11.4e}"
            f" {analytic['decomp_fraction']['mean']:11.4e}"
            f" {result['status']:>8s}")


########################################
# Print overall status
########################################
def print_overall_status(fixed_status, results):
    failed_lambdas = [
        result["lambda"] for result in results
        if result["status"] != "PASS"]

    print("\n" + "#" * 80)
    print("Overall Status")
    print("#" * 80 + "\n")

    if fixed_status == "PASS" and len(failed_lambdas) == 0:
        print("OVERALL STATUS: PASS")
        return

    print("OVERALL STATUS: FAIL")
    if fixed_status != "PASS":
        print("  Fixed-count decomposition failed")
    for lam in failed_lambdas:
        print(f"  Lambda={lam:g} failed")

    if RAISE_ON_FAIL:
        raise RuntimeError("Sampler call-count validation failed")

################################################################################
# Plotting
################################################################################

########################################
# Plot Poisson-regime results
########################################
def plot_regime_results(results):
    lambdas = np.array([result["lambda"] for result in results])
    sampled_lookups = np.array([
        result["sampled"]["table_lookups"]["mean"]
        for result in results])

    curve_lambdas = np.geomspace(lambdas.min(), lambdas.max(), 300)
    curve_moments = [
        analytic_moments(lam)[0] for lam in curve_lambdas]

    def curve_values(name):
        return np.array([
            moments[name]["mean"] for moments in curve_moments])

    photons = curve_values("photon_samples")
    lookups = curve_values("table_lookups")
    direct_lookups = curve_values("direct_lookups")
    power2_lookups = curve_values("power2_lookups")
    zero_fraction = curve_values("zero_fraction")
    direct_fraction = curve_values("direct_fraction")
    decomp_fraction = curve_values("decomp_fraction")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_calls, ax_reduction, ax_fraction, ax_components = axes.ravel()

    ax_calls.loglog(
        curve_lambdas, photons, label="quantum: individual photons")
    ax_calls.loglog(
        curve_lambdas, lookups, label="quantum-kick: table lookups")
    ax_calls.loglog(
        lambdas, sampled_lookups, "x", color="black", label="Monte Carlo")
    ax_calls.set_xlabel(r"$\lambda=\langle N_\gamma\rangle$")
    ax_calls.set_ylabel("Mean energy samples per event")
    ax_calls.set_title("Poisson-averaged sampling work")
    ax_calls.grid(True, which="both", alpha=0.35)
    ax_calls.legend()

    ax_reduction.semilogx(curve_lambdas, photons / lookups)
    ax_reduction.axhline(1.0, color="0.5", linestyle="--")
    ax_reduction.set_xlabel(r"$\lambda=\langle N_\gamma\rangle$")
    ax_reduction.set_ylabel("quantum calls / quantum-kick calls")
    ax_reduction.set_title("Call-count reduction (not timing speedup)")
    ax_reduction.grid(True, which="both", alpha=0.35)

    ax_fraction.semilogx(
        curve_lambdas, zero_fraction, label=r"$N_\gamma=0$")
    ax_fraction.semilogx(
        curve_lambdas, direct_fraction, label=r"$1\leq N_\gamma\leq32$")
    ax_fraction.semilogx(
        curve_lambdas, decomp_fraction, label=r"$N_\gamma>32$")
    ax_fraction.set_xlabel(r"$\lambda=\langle N_\gamma\rangle$")
    ax_fraction.set_ylabel("Event probability")
    ax_fraction.set_title("Probability of each sampling branch")
    ax_fraction.grid(True, which="both", alpha=0.35)
    ax_fraction.legend()

    ax_components.semilogx(
        curve_lambdas, lookups, color="black", label="total")
    ax_components.semilogx(
        curve_lambdas, direct_lookups, label="direct table")
    ax_components.semilogx(
        curve_lambdas, power2_lookups, label="power-of-two tables")
    ax_components.set_xlabel(r"$\lambda=\langle N_\gamma\rangle$")
    ax_components.set_ylabel("Mean table lookups per event")
    ax_components.set_title("Composition of quantum-kick work")
    ax_components.grid(True, which="both", alpha=0.35)
    ax_components.legend()

    fig.suptitle("Sampler work averaged over Poisson photon counts")
    fig.tight_layout()


########################################
# Plot fixed-count decomposition
########################################
def plot_fixed_count_decomposition():
    counts = np.arange(1, MAX_FIXED_COUNT_PLOT + 1)
    lookups, power2_lookups, direct_lookups = lookup_counts_array(counts)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_lookups, ax_reduction = axes

    ax_lookups.bar(
        counts,
        direct_lookups,
        width=0.9,
        label="direct table for remainder 1..32")
    ax_lookups.bar(
        counts,
        power2_lookups,
        width=0.9,
        bottom=direct_lookups,
        label="power-of-two table chunks")
    ax_lookups.step(
        counts,
        lookups,
        where="mid",
        color="black",
        linewidth=1.2,
        label="total table lookups")
    ax_lookups.axvline(32.5, color="0.5", linestyle="--")
    ax_lookups.axvline(63.5, color="0.5", linestyle="--")
    ax_lookups.set_ylim(0.0, 2.35)
    ax_lookups.set_yticks([0, 1, 2])
    ax_lookups.set_ylabel("Quantum-kick lookups")
    ax_lookups.set_title("How a fixed photon count is represented by tables")
    ax_lookups.grid(True, axis="y", alpha=0.35)
    ax_lookups.legend(loc="upper left")

    ax_reduction.plot(counts, counts / lookups, color="black")
    ax_reduction.axhline(1.0, color="0.5", linestyle="--")
    ax_reduction.axvline(32.5, color="0.5", linestyle="--")
    ax_reduction.axvline(63.5, color="0.5", linestyle="--")
    ax_reduction.set_xlabel("Fixed photon count")
    ax_reduction.set_ylabel("quantum calls / quantum-kick calls")
    ax_reduction.set_title("Call-count reduction (not timing speedup)")
    ax_reduction.grid(True, alpha=0.35)

    fig.tight_layout()

################################################################################
# Run
################################################################################

print_setup()

fixed_failures = validate_fixed_count_decomposition()
print_fixed_count_decompositions()
fixed_status = print_fixed_count_status(fixed_failures)

rng = np.random.default_rng(SEED)
results = []

for ii_lambda, lam in enumerate(TARGET_LAMBDAS):
    print("\n" + "#" * 80)
    print(
        f"Running lambda {ii_lambda + 1}/{len(TARGET_LAMBDAS)}: "
        f"lambda = {lam:g}")
    print("#" * 80 + "\n")

    analytic, omitted_probability = analytic_moments(lam)
    sampled = run_monte_carlo_lambda(rng, lam)
    result = compare_lambda(
        lam                 = lam,
        analytic            = analytic,
        sampled             = sampled,
        omitted_probability = omitted_probability)
    results.append(result)
    print_lambda_result(result)

print_compact_summary(results)
print_overall_status(fixed_status, results)

if PLOT_RESULTS:
    plt.close("all")
    plot_regime_results(results)
    plot_fixed_count_decomposition()
    plt.show()
