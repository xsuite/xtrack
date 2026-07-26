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

import matplotlib.pyplot as plt
import numpy as np

################################################################################
# User parameters
################################################################################

########################################
# Stored table checks
########################################
EXPECTED_DIRECT_TABLE_MAX   = 32
EXPECTED_TABLE_SIZE         = 6193
EXPECTED_LEFT_SIZE          = 2296
EXPECTED_CENTER_SIZE        = 1601
EXPECTED_RIGHT_SIZE         = 2296

GRID_ABS_TOL                = 1E-18
MONOTONIC_LOG_TOL           = 1E-14
BOUNDARY_LOG_TOL            = 1E-12

########################################
# Regeneration check
########################################
REGENERATE_DIRECT_TABLE_MAX = 4
REGENERATION_POWERS         = [1, 2, 4]
REGENERATED_LOG_TOL         = 5E-11

########################################
# Plot settings
########################################
PLOT_RESULTS                = True
PLOT_COUNTS                 = [1, 2, 4, 32, 128]

########################################
# File paths
########################################
REPO_ROOT                   = Path(__file__).resolve().parents[3]
HEADER_PATH                 = REPO_ROOT / "xtrack" / "headers" / "synrad_total_energy_tables.h"
GENERATOR_PATH              = REPO_ROOT / "xtrack" / "headers" / "_generate_synrad_total_energy_tables.py"

################################################################################
# Load generator helpers
################################################################################
spec = importlib.util.spec_from_file_location(
    "synrad_total_energy_table_generator", GENERATOR_PATH)
table_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table_generator)

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
# Find table counts
########################################
def find_log_table_counts(header_text):
    counts = re.findall(
        r"static const double\s+synrad_total_energy_log_table_([0-9]+)\[",
        header_text)
    return sorted({int(count) for count in counts})


########################################
# Load stored tables
########################################
def load_stored_tables(header_path):
    header_text = header_path.read_text()

    macros = {
        "table_size": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_TABLE_SIZE", int),
        "direct_table_max": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_DIRECT_TABLE_MAX", int),
        "tail_probability_min": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_TAIL_PROBABILITY_MIN"),
        "tail_probability_max": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_TAIL_PROBABILITY_MAX"),
        "tail_points_per_decade": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_TAIL_POINTS_PER_DECADE", int),
        "center_points": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_CENTER_POINTS", int),
        "left_size": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_LEFT_SIZE", int),
        "center_size": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_CENTER_SIZE", int),
        "right_size": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_RIGHT_SIZE", int),
        "left_offset": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_LEFT_OFFSET", int),
        "center_offset": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_CENTER_OFFSET", int),
        "right_offset": read_macro(
            header_text, "XTRACK_SYNRAD_TOTAL_ENERGY_RIGHT_OFFSET", int)}

    left_u      = read_double_array(header_text, "synrad_total_energy_left_u_grid")
    center_u    = read_double_array(header_text, "synrad_total_energy_center_u_grid")
    right_v     = read_double_array(header_text, "synrad_total_energy_right_v_grid")

    table_counts = find_log_table_counts(header_text)
    log_tables = {}
    for count in table_counts:
        log_tables[count] = read_double_array(
            header_text, f"synrad_total_energy_log_table_{count}")

    return {
        "macros":       macros,
        "left_u":       left_u,
        "center_u":     center_u,
        "right_v":      right_v,
        "table_counts": table_counts,
        "log_tables":   log_tables}


################################################################################
# Check helpers
################################################################################

########################################
# Store one check
########################################
def make_check(section, name, value, limit, passed):
    return {
        "section":  section,
        "name":     name,
        "value":    value,
        "limit":    limit,
        "status":   "PASS" if passed else "FAIL"}


########################################
# Monotonic checks
########################################
def is_increasing(values, tolerance=0.0):
    return bool(np.all(np.diff(values) >= -tolerance))


def is_decreasing(values, tolerance=0.0):
    return bool(np.all(np.diff(values) <= tolerance))


########################################
# Segment slices
########################################
def table_segments(log_table, stored):
    macros = stored["macros"]
    left = log_table[
        macros["left_offset"]:
        macros["left_offset"] + macros["left_size"]]
    center = log_table[
        macros["center_offset"]:
        macros["center_offset"] + macros["center_size"]]
    right = log_table[
        macros["right_offset"]:
        macros["right_offset"] + macros["right_size"]]
    return left, center, right


########################################
# Relative max difference
########################################
def max_abs_difference(values_a, values_b):
    return float(np.max(np.abs(values_a - values_b)))


################################################################################
# Reporting
################################################################################

########################################
# Print stored table summary
########################################
def print_stored_table_summary(stored):
    macros = stored["macros"]

    print("\n" + "#" * 80)
    print("Stored Table Metadata")
    print("#" * 80 + "\n")

    print(f"header path              = {HEADER_PATH}")
    print(f"table size               = {macros['table_size']}")
    print(f"direct table max         = {macros['direct_table_max']}")
    print(f"tail probability min     = {macros['tail_probability_min']:.6e}")
    print(f"tail probability max     = {macros['tail_probability_max']:.6e}")
    print(f"tail points per decade   = {macros['tail_points_per_decade']}")
    print(f"center points            = {macros['center_points']}")
    print(f"left / center / right    = "
          f"{macros['left_size']} / {macros['center_size']} / {macros['right_size']}")
    print(f"offsets                  = "
          f"{macros['left_offset']} / {macros['center_offset']} / {macros['right_offset']}")
    print(f"stored table counts      = {stored['table_counts']}")
    print(f"regenerated direct check = N=1..{REGENERATE_DIRECT_TABLE_MAX}")


########################################
# Print check table
########################################
def print_check_table(title, checks):
    print("\n" + "#" * 80)
    print(title)
    print("#" * 80 + "\n")

    print(
        f"{'section':>14s}"
        f" {'check':>32s}"
        f" {'value':>16s}"
        f" {'limit':>16s}"
        f" {'status':>8s}")

    for check in checks:
        print(
            f"{check['section']:>14s}"
            f" {check['name']:>32s}"
            f" {str(check['value']):>16s}"
            f" {str(check['limit']):>16s}"
            f" {check['status']:>8s}")


########################################
# Print overall status
########################################
def print_overall_status(*check_groups):
    failed = [
        check
        for checks in check_groups
        for check in checks
        if check["status"] != "PASS"]

    print("\n" + "#" * 80)
    print("Overall Status")
    print("#" * 80 + "\n")

    if len(failed) == 0:
        print("OVERALL STATUS: PASS")
    else:
        print("OVERALL STATUS: FAIL")
        print()
        print("Failing checks:")
        for check in failed:
            print(f"  {check['section']} / {check['name']}: {check['value']}")


################################################################################
# Stored table structural checks
################################################################################

########################################
# Check metadata
########################################
def check_metadata(stored):
    macros = stored["macros"]
    required_counts = sorted(
        set(range(1, macros["direct_table_max"] + 1)) | {64, 128, 256})

    checks = []
    checks.append(make_check(
        "metadata", "table size", macros["table_size"], EXPECTED_TABLE_SIZE,
        macros["table_size"] == EXPECTED_TABLE_SIZE))
    checks.append(make_check(
        "metadata", "direct table max", macros["direct_table_max"],
        EXPECTED_DIRECT_TABLE_MAX,
        macros["direct_table_max"] == EXPECTED_DIRECT_TABLE_MAX))
    checks.append(make_check(
        "metadata", "left size", macros["left_size"], EXPECTED_LEFT_SIZE,
        macros["left_size"] == EXPECTED_LEFT_SIZE))
    checks.append(make_check(
        "metadata", "center size", macros["center_size"], EXPECTED_CENTER_SIZE,
        macros["center_size"] == EXPECTED_CENTER_SIZE))
    checks.append(make_check(
        "metadata", "right size", macros["right_size"], EXPECTED_RIGHT_SIZE,
        macros["right_size"] == EXPECTED_RIGHT_SIZE))
    checks.append(make_check(
        "metadata", "left offset", macros["left_offset"], 0,
        macros["left_offset"] == 0))
    checks.append(make_check(
        "metadata", "center offset", macros["center_offset"],
        macros["left_size"],
        macros["center_offset"] == macros["left_size"]))
    checks.append(make_check(
        "metadata", "right offset", macros["right_offset"],
        macros["left_size"] + macros["center_size"],
        macros["right_offset"] == macros["left_size"] + macros["center_size"]))
    checks.append(make_check(
        "metadata", "table counts", stored["table_counts"], required_counts,
        stored["table_counts"] == required_counts))

    return checks


########################################
# Check probability grids
########################################
def check_probability_grids(stored):
    macros = stored["macros"]
    left_u = stored["left_u"]
    center_u = stored["center_u"]
    right_v = stored["right_v"]

    checks = []
    checks.append(make_check(
        "grid", "left size", left_u.size, macros["left_size"],
        left_u.size == macros["left_size"]))
    checks.append(make_check(
        "grid", "center size", center_u.size, macros["center_size"],
        center_u.size == macros["center_size"]))
    checks.append(make_check(
        "grid", "right size", right_v.size, macros["right_size"],
        right_v.size == macros["right_size"]))
    checks.append(make_check(
        "grid", "left finite", np.all(np.isfinite(left_u)), True,
        np.all(np.isfinite(left_u))))
    checks.append(make_check(
        "grid", "center finite", np.all(np.isfinite(center_u)), True,
        np.all(np.isfinite(center_u))))
    checks.append(make_check(
        "grid", "right finite", np.all(np.isfinite(right_v)), True,
        np.all(np.isfinite(right_v))))
    checks.append(make_check(
        "grid", "left increasing", is_increasing(left_u, GRID_ABS_TOL), True,
        is_increasing(left_u, GRID_ABS_TOL)))
    checks.append(make_check(
        "grid", "center increasing", is_increasing(center_u, GRID_ABS_TOL), True,
        is_increasing(center_u, GRID_ABS_TOL)))
    checks.append(make_check(
        "grid", "right-v increasing", is_increasing(right_v, GRID_ABS_TOL), True,
        is_increasing(right_v, GRID_ABS_TOL)))
    checks.append(make_check(
        "grid", "left starts at zero", left_u[0], 0.0,
        abs(left_u[0]) <= GRID_ABS_TOL))
    checks.append(make_check(
        "grid", "right starts at zero", right_v[0], 0.0,
        abs(right_v[0]) <= GRID_ABS_TOL))
    checks.append(make_check(
        "grid", "left tail boundary", left_u[-1],
        macros["tail_probability_max"],
        abs(left_u[-1] - macros["tail_probability_max"]) <= GRID_ABS_TOL))
    checks.append(make_check(
        "grid", "center lower boundary", center_u[0],
        macros["tail_probability_max"],
        abs(center_u[0] - macros["tail_probability_max"]) <= GRID_ABS_TOL))
    checks.append(make_check(
        "grid", "center upper boundary", center_u[-1],
        1.0 - macros["tail_probability_max"],
        abs(center_u[-1] - (1.0 - macros["tail_probability_max"])) <= GRID_ABS_TOL))
    checks.append(make_check(
        "grid", "right tail boundary", right_v[-1],
        macros["tail_probability_max"],
        abs(right_v[-1] - macros["tail_probability_max"]) <= GRID_ABS_TOL))

    return checks


########################################
# Check log tables
########################################
def check_log_tables(stored):
    macros = stored["macros"]
    checks = []

    for count in stored["table_counts"]:
        log_table = stored["log_tables"][count]
        left, center, right = table_segments(log_table, stored)

        finite = bool(np.all(np.isfinite(log_table)))
        left_mono = is_increasing(left, MONOTONIC_LOG_TOL)
        center_mono = is_increasing(center, MONOTONIC_LOG_TOL)
        right_mono = is_decreasing(right, MONOTONIC_LOG_TOL)
        left_center_jump = abs(left[-1] - center[0])
        center_right_jump = abs(center[-1] - right[-1])

        checks.append(make_check(
            f"N={count}", "table size", log_table.size, macros["table_size"],
            log_table.size == macros["table_size"]))
        checks.append(make_check(
            f"N={count}", "finite log values", finite, True, finite))
        checks.append(make_check(
            f"N={count}", "left monotonic", left_mono, True, left_mono))
        checks.append(make_check(
            f"N={count}", "center monotonic", center_mono, True, center_mono))
        checks.append(make_check(
            f"N={count}", "right monotonic in v", right_mono, True, right_mono))
        checks.append(make_check(
            f"N={count}", "left/center boundary", f"{left_center_jump:.3e}",
            f"< {BOUNDARY_LOG_TOL:.1e}",
            left_center_jump <= BOUNDARY_LOG_TOL))
        checks.append(make_check(
            f"N={count}", "center/right boundary", f"{center_right_jump:.3e}",
            f"< {BOUNDARY_LOG_TOL:.1e}",
            center_right_jump <= BOUNDARY_LOG_TOL))

    return checks


################################################################################
# Regeneration checks
################################################################################

########################################
# Build regenerated subset
########################################
def build_regenerated_subset():
    print("\n" + "#" * 80)
    print("Regenerating Small Direct Table Subset")
    print("#" * 80 + "\n")
    print(
        f"Regenerating direct tables through N={REGENERATE_DIRECT_TABLE_MAX}. "
        "This is intentionally much smaller than the full production table set.")

    time_start = time.time()
    runtime_grid = table_generator.make_segmented_probability_grid()
    regenerated_grid, quantile_tables = (
        table_generator.build_total_energy_quantile_tables(
            u_grid=runtime_grid,
            direct_table_max=REGENERATE_DIRECT_TABLE_MAX,
            powers=REGENERATION_POWERS))
    elapsed = time.time() - time_start

    print()
    print(f"Regeneration completed in {elapsed:.1f} s")

    log_tables = {
        count: np.log(values)
        for count, values in sorted(quantile_tables.items())
        if count <= REGENERATE_DIRECT_TABLE_MAX}

    return regenerated_grid, log_tables


########################################
# Check regenerated subset
########################################
def check_regenerated_subset(stored, regenerated_grid, regenerated_log_tables):
    checks = []

    checks.append(make_check(
        "regen-grid", "left grid",
        f"{max_abs_difference(stored['left_u'], regenerated_grid['left_u']):.3e}",
        f"< {GRID_ABS_TOL:.1e}",
        max_abs_difference(stored["left_u"], regenerated_grid["left_u"])
        <= GRID_ABS_TOL))
    checks.append(make_check(
        "regen-grid", "center grid",
        f"{max_abs_difference(stored['center_u'], regenerated_grid['center_u']):.3e}",
        f"< {GRID_ABS_TOL:.1e}",
        max_abs_difference(stored["center_u"], regenerated_grid["center_u"])
        <= GRID_ABS_TOL))
    checks.append(make_check(
        "regen-grid", "right grid",
        f"{max_abs_difference(stored['right_v'], regenerated_grid['right_v']):.3e}",
        f"< {GRID_ABS_TOL:.1e}",
        max_abs_difference(stored["right_v"], regenerated_grid["right_v"])
        <= GRID_ABS_TOL))

    for count in range(1, REGENERATE_DIRECT_TABLE_MAX + 1):
        stored_table = stored["log_tables"][count]
        regenerated_table = regenerated_log_tables[count]
        abs_diff = np.abs(stored_table - regenerated_table)
        max_diff = float(np.max(abs_diff))
        rms_diff = float(np.sqrt(np.mean(abs_diff**2)))
        i_max = int(np.argmax(abs_diff))

        checks.append(make_check(
            f"regen N={count}", "max |dlogQ|",
            f"{max_diff:.3e} at i={i_max}",
            f"< {REGENERATED_LOG_TOL:.1e}",
            max_diff <= REGENERATED_LOG_TOL))
        checks.append(make_check(
            f"regen N={count}", "rms |dlogQ|",
            f"{rms_diff:.3e}",
            f"< {REGENERATED_LOG_TOL:.1e}",
            rms_diff <= REGENERATED_LOG_TOL))

    return checks


################################################################################
# Plotting
################################################################################

########################################
# Reconstruct sorted probability grid
########################################
def sorted_probability_grid_and_log_table(count, stored):
    left, center, right = table_segments(stored["log_tables"][count], stored)
    u_sorted = np.concatenate((
        stored["left_u"],
        stored["center_u"],
        1.0 - stored["right_v"][::-1]))
    log_sorted = np.concatenate((
        left,
        center,
        right[::-1]))
    return u_sorted, log_sorted


########################################
# Plot inverse CDFs
########################################
def plot_inverse_cdfs(stored):
    counts = [count for count in PLOT_COUNTS if count in stored["log_tables"]]
    if len(counts) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_full, ax_tail = axes

    for count in counts:
        u_sorted, log_sorted = sorted_probability_grid_and_log_table(count, stored)
        values = np.exp(log_sorted)

        ax_full.semilogy(u_sorted, values, label=f"N={count}")

        _left, _center, right = table_segments(stored["log_tables"][count], stored)
        survival = stored["right_v"][1:]
        tail_values = np.exp(right[1:])
        ax_tail.loglog(survival, tail_values, label=f"N={count}")

    ax_full.set_xlabel("CDF probability")
    ax_full.set_ylabel("Total normalized energy quantile")
    ax_full.set_title("Stored inverse CDF tables")
    ax_full.grid(True)
    ax_full.legend()

    ax_tail.set_xlabel("Survival probability")
    ax_tail.set_ylabel("Total normalized energy quantile")
    ax_tail.set_title("Stored right-tail inverse CDFs")
    ax_tail.invert_xaxis()
    ax_tail.grid(True, which="both")
    ax_tail.legend()

    fig.tight_layout()


################################################################################
# Run
################################################################################

print("\n" + "#" * 80)
print("Loading Stored Tables")
print("#" * 80 + "\n")

stored = load_stored_tables(HEADER_PATH)
print_stored_table_summary(stored)

metadata_checks = check_metadata(stored)
grid_checks = check_probability_grids(stored)
log_table_checks = check_log_tables(stored)

print_check_table("Metadata checks", metadata_checks)
print_check_table("Probability grid checks", grid_checks)
print_check_table("Stored log-table checks", log_table_checks)

regenerated_grid, regenerated_log_tables = build_regenerated_subset()
regeneration_checks = check_regenerated_subset(
    stored, regenerated_grid, regenerated_log_tables)
print_check_table("Regenerated subset checks", regeneration_checks)

print_overall_status(
    metadata_checks,
    grid_checks,
    log_table_checks,
    regeneration_checks)

if PLOT_RESULTS:
    plt.close("all")
    plot_inverse_cdfs(stored)
    plt.show()
