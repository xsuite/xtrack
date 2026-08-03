# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import importlib.util
import re
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# User parameters
################################################################################

########################################
# Monte Carlo settings
########################################
SEED                    = 12345
N_SAMPLES               = int(1E7)
PHOTON_BATCH_SIZE       = int(1E6)

FIXED_COUNTS            = [1, 2, 4, 8, 16, 32, 64, 128]
LAMBDAS                 = [1E-2, 3E-2, 1E-1, 3E-1, 1.0, 3.0, 9.0, 30.0]
TAIL_QUANTILES          = [0.99, 0.999, 0.9999]

########################################
# Acceptance limits
########################################
MEAN_Z_LIMIT            = 5.0
KS_LIMIT_FACTOR         = 1.95
TAIL_Z_LIMIT            = 5.0
MIN_TAIL_EVENTS_FOR_PASS    = 20
RAISE_ON_FAIL           = False

########################################
# Plot settings
########################################
PLOT_RESULTS            = True
PLOT_FIXED_COUNTS       = [1, 2, 4, 8, 16, 32, 64, 128]
PLOT_LAMBDAS            = [1E-2, 3E-2, 1E-1, 3E-1, 1.0, 3.0, 9.0, 30.0]

########################################
# File paths
########################################
REPO_ROOT               = Path(__file__).resolve().parents[3]
HEADER_PATH             = REPO_ROOT / "xtrack" / "headers" / "synrad_total_energy_tables.h"
GENERATOR_PATH          = REPO_ROOT / "xtrack" / "headers" / "_generate_synrad_total_energy_tables.py"

################################################################################
# Load generator helpers
################################################################################
spec = importlib.util.spec_from_file_location(
    "synrad_total_energy_table_generator", GENERATOR_PATH)
table_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table_generator)

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
    if fraction > 0:
        remaining = elapsed * (1.0 / fraction - 1.0)
    else:
        remaining = 0.0

    bar_width = 28
    n_filled = int(round(bar_width * fraction))
    bar = "#" * n_filled + "." * (bar_width - n_filled)

    print(
        f"\r{label:<34s} [{bar}] "
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
# Header parsing
################################################################################

########################################
# Read macro
########################################
def read_macro(header_text, name, value_type=float):
    match = re.search(
        rf"#define\s+{re.escape(name)}\s+([^\s]+)",
        header_text)
    if match is None:
        raise ValueError(f"Could not find macro {name}")

    return value_type(match.group(1))

########################################
# Read C double array
########################################
def read_double_array(header_text, name):
    match = re.search(
        rf"static const double\s+{re.escape(name)}\[[^\]]+\]\s*=\s*\{{(.*?)\}};",
        header_text,
        flags=re.S)
    if match is None:
        raise ValueError(f"Could not find array {name}")

    return np.fromstring(match.group(1).replace(",", " "), sep=" ")

########################################
# Load stored tables
########################################
def load_stored_tables(header_path):
    header_text = header_path.read_text()

    table_size         = read_macro(header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_TABLE_SIZE", int)
    direct_table_max   = read_macro(header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_DIRECT_TABLE_MAX", int)
    tail_prob_max      = read_macro(header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_TAIL_PROBABILITY_MAX")

    left_u      = read_double_array(header_text, "synrad_total_energy_left_u_grid")
    center_u    = read_double_array(header_text, "synrad_total_energy_center_u_grid")
    right_v     = read_double_array(header_text, "synrad_total_energy_right_v_grid")

    table_counts = list(range(1, direct_table_max + 1)) + [64, 128, 256]
    log_tables = {}
    for count in table_counts:
        log_table = read_double_array(
            header_text, f"synrad_total_energy_log_table_{count}")
        if log_table.size != table_size:
            raise ValueError(
                f"Table {count} has size {log_table.size}, expected {table_size}")
        log_tables[count] = log_table

    return {
        "table_size":        table_size,
        "direct_table_max":  direct_table_max,
        "tail_prob_max":     tail_prob_max,
        "left_u":            left_u,
        "center_u":          center_u,
        "right_v":           right_v,
        "log_tables":        log_tables}


################################################################################
# Sampling helpers
################################################################################

########################################
# Sample photon energies
########################################
def sample_photon_energy_normalized(rng, n_samples):
    xlow    = 1.0
    a1      = 2.149528241534391
    a2      = 1.770750801624037
    ratio   = 0.908250405131381

    samples = np.empty(n_samples)
    n_done  = 0

    while n_done < n_samples:
        n_try       = max(1024, int(1.35 * (n_samples - n_done)))
        use_low     = rng.random(n_try) < ratio
        candidate   = np.empty(n_try)
        approx      = np.empty(n_try)

        u_low = rng.random(np.count_nonzero(use_low))
        candidate[use_low] = u_low**3
        approx[use_low] = a1 / np.maximum(
            u_low * u_low, np.finfo(float).tiny)

        u_high = rng.random(np.count_nonzero(~use_low))
        candidate[~use_low] = (
            xlow - np.log(np.maximum(u_high, np.finfo(float).tiny)))
        approx[~use_low] = a2 * np.exp(-candidate[~use_low])

        accepted = table_generator.synrad(candidate) >= approx * rng.random(n_try)
        accepted_values = candidate[accepted]

        n_take = min(accepted_values.size, n_samples - n_done)
        samples[n_done:n_done + n_take] = accepted_values[:n_take]
        n_done += n_take

    return samples


########################################
# Sample fixed-count brute force
########################################
def sample_fixed_count_bruteforce(rng, count, n_samples, progress_label=None):
    if count == 0:
        return np.zeros(n_samples)

    values = np.empty(n_samples)
    n_done = 0
    n_batch_events = max(1, PHOTON_BATCH_SIZE // count)
    time_start = time.time()

    while n_done < n_samples:
        n_batch = min(n_batch_events, n_samples - n_done)
        photon_energy = sample_photon_energy_normalized(rng, count * n_batch)
        values[n_done:n_done + n_batch] = (
            photon_energy.reshape(n_batch, count).sum(axis=1))
        n_done += n_batch

        if progress_label is not None:
            print_progress(progress_label, n_done, n_samples, time_start)

    if progress_label is not None:
        print_progress(progress_label, n_samples, n_samples, time_start, force=True)

    return values


########################################
# Sample compound-Poisson brute force
########################################
def sample_compound_poisson_bruteforce(rng, lam, n_samples, progress_label=None):
    counts          = rng.poisson(lam, size=n_samples)
    values          = np.zeros(n_samples)

    n_done = 0
    time_start = time.time()
    while n_done < n_samples:
        mean_count = max(lam, 1.0)
        n_batch = min(
            max(1, int(PHOTON_BATCH_SIZE / mean_count)),
            n_samples - n_done)
        counts_batch = counts[n_done:n_done + n_batch]
        total_photons = int(np.sum(counts_batch))

        if total_photons > 0:
            photon_energy = sample_photon_energy_normalized(rng, total_photons)
            event_index = np.repeat(np.arange(n_batch), counts_batch)
            np.add.at(
                values[n_done:n_done + n_batch],
                event_index,
                photon_energy)

        n_done += n_batch

        if progress_label is not None:
            print_progress(progress_label, n_done, n_samples, time_start)

    if progress_label is not None:
        print_progress(progress_label, n_samples, n_samples, time_start, force=True)

    return values, counts


########################################
# Interpolate log table
########################################
def interpolate_log_table(u_values, log_table, stored):
    out             = np.empty_like(u_values)
    tail_prob_max   = stored["tail_prob_max"]

    left_mask   = u_values < tail_prob_max
    center_mask = (
        (u_values >= tail_prob_max)
        & (u_values <= 1.0 - tail_prob_max))
    right_mask  = u_values > 1.0 - tail_prob_max

    if np.any(left_mask):
        out[left_mask] = interpolate_segment(
            values          = u_values[left_mask],
            grid            = stored["left_u"],
            table           = log_table[:stored["left_u"].size],
            is_log_spaced   = True)

    if np.any(center_mask):
        left_size = stored["left_u"].size
        center_size = stored["center_u"].size
        out[center_mask] = interpolate_segment(
            values          = u_values[center_mask],
            grid            = stored["center_u"],
            table           = log_table[left_size:left_size + center_size],
            is_log_spaced   = False)

    if np.any(right_mask):
        right_size = stored["right_v"].size
        out[right_mask] = interpolate_segment(
            values          = 1.0 - u_values[right_mask],
            grid            = stored["right_v"],
            table           = log_table[-right_size:],
            is_log_spaced   = True)

    return out


########################################
# Interpolate one segment
########################################
def interpolate_segment(values, grid, table, is_log_spaced):
    out = np.empty_like(values)

    low_mask    = values <= grid[0]
    high_mask   = values >= grid[-1]
    mid_mask    = ~(low_mask | high_mask)

    out[low_mask]   = table[0]
    out[high_mask]  = table[-1]

    if np.any(mid_mask):
        values_mid = values[mid_mask]
        out_mid = np.empty_like(values_mid)

        if is_log_spaced:
            first_mask = values_mid < grid[1]
            rest_mask  = ~first_mask

            if np.any(first_mask):
                out_mid[first_mask] = (
                    table[0]
                    + values_mid[first_mask] / grid[1] * (table[1] - table[0]))

            if np.any(rest_mask):
                coord_grid = np.log(grid[1:])
                coord_val  = np.log(values_mid[rest_mask])
                out_mid[rest_mask] = np.interp(coord_val, coord_grid, table[1:])
        else:
            out_mid = np.interp(values_mid, grid, table)

        out[mid_mask] = out_mid

    return out


########################################
# Sample from one table
########################################
def sample_from_table(rng, count, n_samples, stored):
    u_values = rng.random(n_samples)
    log_values = interpolate_log_table(
        u_values     = u_values,
        log_table    = stored["log_tables"][count],
        stored       = stored)
    return np.exp(log_values)


########################################
# Decompose photon count
########################################
def decompose_photon_count(count, stored):
    if count == 0:
        return []

    n_left = int(count)
    chunks = []

    while n_left > stored["direct_table_max"]:
        chunk = 1
        while chunk <= n_left // 2 and chunk < 256:
            chunk *= 2
        chunks.append(chunk)
        n_left -= chunk

    if n_left > 0:
        chunks.append(n_left)

    return chunks


########################################
# Sample fixed count from tables
########################################
def sample_fixed_count_from_tables(rng, count, n_samples, stored,
                                   progress_label=None):
    values = np.zeros(n_samples)
    chunks = decompose_photon_count(count, stored)
    time_start = time.time()
    n_done = 0

    while n_done < n_samples:
        n_batch = min(PHOTON_BATCH_SIZE, n_samples - n_done)

        for chunk in chunks:
            values[n_done:n_done + n_batch] += sample_from_table(
                rng         = rng,
                count       = chunk,
                n_samples   = n_batch,
                stored      = stored)

        n_done += n_batch

        if progress_label is not None:
            print_progress(progress_label, n_done, n_samples, time_start)

    if progress_label is not None:
        print_progress(progress_label, n_samples, n_samples, time_start, force=True)

    return values


########################################
# Sample compound Poisson from tables
########################################
def sample_compound_poisson_from_tables(rng, lam, n_samples, stored,
                                        progress_label=None):
    counts      = rng.poisson(lam, size=n_samples)
    values      = np.zeros(n_samples)
    n_chunks    = np.zeros(n_samples, dtype=np.int64)

    unique_counts = np.unique(counts)
    time_start = time.time()
    n_done = 0

    for count in unique_counts:
        mask = counts == count
        n_for_count = np.count_nonzero(mask)

        if count == 0:
            n_done += n_for_count
            if progress_label is not None:
                print_progress(progress_label, n_done, n_samples, time_start)
            continue

        chunks = decompose_photon_count(count, stored)
        n_chunks[mask] = len(chunks)
        values_count = np.zeros(n_for_count)

        for chunk in chunks:
            values_count += sample_from_table(
                rng         = rng,
                count       = chunk,
                n_samples   = n_for_count,
                stored      = stored)

        values[mask] = values_count
        n_done += n_for_count

        if progress_label is not None:
            print_progress(progress_label, n_done, n_samples, time_start)

    if progress_label is not None:
        print_progress(progress_label, n_samples, n_samples, time_start, force=True)

    return values, counts, n_chunks


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
# Tail probabilities
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

        if sigma > 0:
            z_value = abs(p_cand - p_ref) / sigma
        else:
            z_value = 0.0

        checks.append({
            "quantile":     quantile,
            "threshold":    threshold,
            "p_ref":        p_ref,
            "p_cand":       p_cand,
            "events_ref":    expected_ref_events,
            "rel":          p_cand / p_ref - 1.0 if p_ref > 0 else 0.0,
            "z":            z_value,
            "status":       tail_status(z_value, expected_ref_events)})

    return checks


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
# Compare distributions
########################################
def compare_distributions(label, reference, candidate, mean_chunks=None,
                          keep_samples=False):
    mean_ref    = np.mean(reference)
    mean_cand   = np.mean(candidate)
    rms_ref     = np.std(reference)
    rms_cand    = np.std(candidate)

    mean_uncertainty = np.sqrt(
        np.var(reference) / reference.size
        + np.var(candidate) / candidate.size)
    if mean_uncertainty > 0:
        mean_z = abs(mean_cand - mean_ref) / mean_uncertainty
    else:
        mean_z = 0.0

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

    result = {
        "label":        label,
        "reference":    reference if keep_samples else None,
        "candidate":    candidate if keep_samples else None,
        "mean_rel":     mean_cand / mean_ref - 1.0 if mean_ref != 0 else 0.0,
        "mean_z":       mean_z,
        "rms_rel":      rms_cand / rms_ref - 1.0 if rms_ref != 0 else 0.0,
        "ks":           ks_distance,
        "ks_limit":     ks_limit,
        "tail_checks":  tail_checks,
        "max_tail_z":   max_tail_z,
        "mean_status":  mean_status,
        "ks_status":    ks_status,
        "tail_status":  tail_status_all,
        "status":       overall_status,
        "mean_chunks":  mean_chunks}

    return result


################################################################################
# Reporting
################################################################################

########################################
# Print stored table summary
########################################
def print_stored_table_summary(stored):
    print("\n" + "#" * 80)
    print("Stored Tables")
    print("#" * 80 + "\n")

    print(f"header path        = {HEADER_PATH}")
    print(f"table size         = {stored['table_size']}")
    print(f"direct table max   = {stored['direct_table_max']}")
    print(f"tail probability   = {stored['tail_prob_max']:.6e}")
    print(f"stored counts      = {sorted(stored['log_tables'])}")
    print(f"N_SAMPLES          = {N_SAMPLES:g}")
    print(f"PHOTON_BATCH_SIZE  = {PHOTON_BATCH_SIZE:g}")
    print(f"TAIL_QUANTILES     = {TAIL_QUANTILES}")
    print(f"MEAN_Z_LIMIT       = {MEAN_Z_LIMIT:g}")
    print(f"KS_LIMIT_FACTOR    = {KS_LIMIT_FACTOR:g}")
    print(f"TAIL_Z_LIMIT       = {TAIL_Z_LIMIT:g}")


########################################
# Print comparison table
########################################
def print_comparison_table(title, results, include_chunks=False):
    print("\n" + "#" * 80)
    print(title)
    print("#" * 80 + "\n")

    header = (
        f"{'case':>10s}"
        f" {'mean rel':>11s}"
        f" {'mean z':>8s}"
        f" {'mean':>6s}"
        f" {'rms rel':>11s}"
        f" {'KS stat':>10s}"
        f" {'KS lim':>10s}"
        f" {'KS':>6s}"
        f" {'max tail z':>11s}"
        f" {'tail':>8s}")
    if include_chunks:
        header += f" {'<chunks>':>10s}"
    header += f" {'overall':>8s}"
    print(header)

    for result in results:
        row = (
            f"{result['label']:>10s}"
            f" {result['mean_rel']:+11.4e}"
            f" {result['mean_z']:8.3f}"
            f" {result['mean_status']:>6s}"
            f" {result['rms_rel']:+11.4e}"
            f" {result['ks']:10.4e}"
            f" {result['ks_limit']:10.4e}"
            f" {result['ks_status']:>6s}"
            f" {result['max_tail_z']:11.3f}"
            f" {result['tail_status']:>8s}")
        if include_chunks:
            row += f" {result['mean_chunks']:10.4f}"
        row += f" {result['status']:>8s}"
        print(row)


########################################
# Print tail table
########################################
def print_tail_table(title, results):
    print("\n" + title)
    print("-" * len(title))

    print(
        f"{'case':>10s}"
        f" {'q':>9s}"
        f" {'threshold':>14s}"
        f" {'events':>8s}"
        f" {'P ref':>11s}"
        f" {'P table':>11s}"
        f" {'P rel':>11s}"
        f" {'z':>8s}"
        f" {'status':>8s}")

    for result in results:
        for check in result["tail_checks"]:
            print(
                f"{result['label']:>10s}"
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
def print_overall_status(fixed_results, poisson_results):
    all_results = fixed_results + poisson_results
    failed = [result for result in all_results if result["status"] != "PASS"]

    print("\n" + "#" * 80)
    print("Overall Status")
    print("#" * 80 + "\n")

    if len(failed) == 0:
        print("OVERALL STATUS: PASS")
    else:
        print("OVERALL STATUS: FAIL")
        print()
        print("Failing cases:")
        for result in failed:
            print(
                f"  {result['label']}: "
                f"mean={result['mean_status']}, "
                f"KS={result['ks_status']}, "
                f"tail={result['tail_status']}")

    if RAISE_ON_FAIL and len(failed) > 0:
        raise RuntimeError("Table sampling validation failed")


################################################################################
# Plotting
################################################################################

########################################
# Select results
########################################
def select_results(results, selected_labels):
    selected = []
    for result in results:
        if (
            result["label"] in {f"{label:g}" for label in selected_labels}
            and result["reference"] is not None
            and result["candidate"] is not None
        ):
            selected.append(result)
    return selected


########################################
# CDF grid
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
# Plot CDF comparisons
########################################
def plot_cdf_comparisons(results, selected_labels, title, label_prefix):
    selected = select_results(results, selected_labels)
    if len(selected) == 0:
        return

    fig, axes = plt.subplots(
        2,
        len(selected),
        figsize=(4.2 * len(selected), 6.0),
        gridspec_kw={"height_ratios": [3, 1]})

    if len(selected) == 1:
        axes = axes.reshape(2, 1)

    for ii, result in enumerate(selected):
        ax_cdf = axes[0, ii]
        ax_res = axes[1, ii]

        x_grid, cdf_reference, cdf_candidate, residual = cdf_residual_grid(
            result["reference"], result["candidate"])

        ax_cdf.plot(x_grid, cdf_reference, label="brute force", linewidth=1.3)
        ax_cdf.plot(x_grid, cdf_candidate, label="table", linewidth=1.1)
        ax_cdf.set_title(f"{label_prefix} = {result['label']}")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.grid(True)
        ax_cdf.legend()

        ax_res.plot(x_grid, residual, color="black", linewidth=1.1)
        ax_res.axhline(0.0, color="0.5", linewidth=0.8)
        ax_res.axhline(result["ks_limit"], color="0.6", linestyle="--")
        ax_res.axhline(-result["ks_limit"], color="0.6", linestyle="--")
        ax_res.set_xlabel("Total normalized energy loss")
        ax_res.set_ylabel("CDF diff.")
        ax_res.grid(True)

        residual_max = max(np.max(np.abs(residual)), result["ks_limit"])
        ax_res.set_ylim(-1.15 * residual_max, 1.15 * residual_max)

    fig.suptitle(title)
    fig.tight_layout()


################################################################################
# Run
################################################################################

print("\n" + "#" * 80)
print("Loading Stored Tables")
print("#" * 80 + "\n")

stored = load_stored_tables(HEADER_PATH)
print_stored_table_summary(stored)

################################################################################
# Fixed photon-count checks
################################################################################
print("\n" + "#" * 80)
print("Fixed Photon Counts")
print("#" * 80 + "\n")

rng_reference = np.random.default_rng(SEED)
rng_table     = np.random.default_rng(SEED + 1)

fixed_results = []
for ii_count, count in enumerate(FIXED_COUNTS):
    print(f"[{ii_count + 1}/{len(FIXED_COUNTS)}] Sampling fixed N = {count}")
    keep_samples = PLOT_RESULTS and count in PLOT_FIXED_COUNTS
    reference = sample_fixed_count_bruteforce(
        rng         = rng_reference,
        count       = count,
        n_samples   = N_SAMPLES,
        progress_label = f"N={count} brute force")
    candidate = sample_fixed_count_from_tables(
        rng         = rng_table,
        count       = count,
        n_samples   = N_SAMPLES,
        stored      = stored,
        progress_label = f"N={count} table")
    fixed_results.append(compare_distributions(
        label       = str(count),
        reference   = reference,
        candidate   = candidate,
        mean_chunks = len(decompose_photon_count(count, stored)),
        keep_samples = keep_samples))

print_comparison_table(
    "Fixed photon-count checks", fixed_results, include_chunks=True)
print_tail_table("Fixed photon-count tail checks", fixed_results)

################################################################################
# Compound-Poisson checks
################################################################################
print("\n" + "#" * 80)
print("Compound Poisson")
print("#" * 80 + "\n")

rng_reference = np.random.default_rng(SEED + 2)
rng_table     = np.random.default_rng(SEED + 3)

poisson_results = []
for ii_lam, lam in enumerate(LAMBDAS):
    print(f"[{ii_lam + 1}/{len(LAMBDAS)}] Sampling lambda = {lam:g}")
    keep_samples = PLOT_RESULTS and lam in PLOT_LAMBDAS
    reference, reference_counts = sample_compound_poisson_bruteforce(
        rng         = rng_reference,
        lam         = lam,
        n_samples   = N_SAMPLES,
        progress_label = f"lambda={lam:g} brute force")
    candidate, candidate_counts, n_chunks = sample_compound_poisson_from_tables(
        rng         = rng_table,
        lam         = lam,
        n_samples   = N_SAMPLES,
        stored      = stored,
        progress_label = f"lambda={lam:g} table")

    result = compare_distributions(
        label       = f"{lam:g}",
        reference   = reference,
        candidate   = candidate,
        mean_chunks = np.mean(n_chunks),
        keep_samples = keep_samples)
    result["zero_ref"] = np.mean(reference_counts == 0)
    result["one_ref"] = np.mean(reference_counts == 1)
    result["zero_table"] = np.mean(candidate_counts == 0)
    result["one_table"] = np.mean(candidate_counts == 1)
    result["direct_table"] = np.mean(
        (candidate_counts > 0) & (candidate_counts <= stored["direct_table_max"]))
    result["decomposed_table"] = np.mean(
        candidate_counts > stored["direct_table_max"])
    poisson_results.append(result)

print_comparison_table(
    "Compound-Poisson checks", poisson_results, include_chunks=True)
print_tail_table("Compound-Poisson tail checks", poisson_results)

print("\nPhoton-count fractions")
print("----------------------")
print(
    f"{'lambda':>10s}"
    f" {'P0 ref':>10s}"
    f" {'P0 table':>10s}"
    f" {'P1 ref':>10s}"
    f" {'P1 table':>10s}"
    f" {'P direct':>10s}"
    f" {'P decomp':>10s}")
for result in poisson_results:
    print(
        f"{result['label']:>10s}"
        f" {result['zero_ref']:10.4e}"
        f" {result['zero_table']:10.4e}"
        f" {result['one_ref']:10.4e}"
        f" {result['one_table']:10.4e}"
        f" {result['direct_table']:10.4e}"
        f" {result['decomposed_table']:10.4e}")

print_overall_status(fixed_results, poisson_results)

################################################################################
# Plots
################################################################################
if PLOT_RESULTS:
    plt.close("all")
    plot_cdf_comparisons(
        fixed_results,
        PLOT_FIXED_COUNTS,
        "Fixed photon-count total-energy CDFs",
        r"$N_\gamma$")
    plot_cdf_comparisons(
        poisson_results,
        PLOT_LAMBDAS,
        "Compound-Poisson total-energy CDFs",
        r"$\lambda$")
    plt.show()
