"""001a - bar plots of the 001 benchmark, read from 001_bench.json."""

import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JSON_FILE = HERE / "001_benchmark_results.json"
MODES = ["scalar", "tpsa", "param"]
COLORS = {"xs": "#4C72B0", "ng": "#DD8452"}
SIDE_NAMES = {"xs": "Xsuite", "ng": "MAD-NG"}


def lighten(hex_color, amount=0.5):
    """Lighten a hex color by blending towards white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[k : k + 2], 16) / 255.0 for k in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1 - l) * amount
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


LIGHT_COLORS = {k: lighten(v, 0.6) for k, v in COLORS.items()}


def grouped_bars(rows, group_labels, divisors, ylabel, title, fname):
    """One group of xs/ng bar pairs per row, tracking at the base, setup stacked on top.

    Bars are means of the per-iteration times, so the two segments add up to the total
    exactly, and the error bar is the standard error of the total mean. The tick is the
    median of the same sample, so the gap below the bar top is the slow-outlier time
    this run picked up.
    """
    bar_width = 0.8 / (len(MODES) * 2)
    group_gap = 1.0
    fig, ax = plt.subplots(figsize=(3 + 2.3 * len(rows), 6))
    tick_positions, tick_labels, group_centres = [], [], []

    for i, row in enumerate(rows):
        base = i * group_gap
        for j, mode in enumerate(MODES):
            for k, side in enumerate(("xs", "ng")):
                body = row["track"][mode][side]["mean"] / divisors[i]
                total = row["total"][mode][side]["mean"] / divisors[i]
                setup = total - body
                total_sem = row["total"][mode][side]["sem"] / divisors[i]
                total_med = row["total"][mode][side]["med"] / divisors[i]
                pos = base + j * (2 * bar_width + 0.02) + k * bar_width
                first = i == 0 and j == 0
                ax.bar(
                    pos,
                    body,
                    width=bar_width,
                    color=COLORS[side],
                    label=f"{SIDE_NAMES[side]} track" if first else None,
                )
                ax.bar(
                    pos,
                    setup,
                    width=bar_width,
                    bottom=body,
                    color=LIGHT_COLORS[side],
                    label=f"{SIDE_NAMES[side]} setup" if first else None,
                    yerr=total_sem,
                    error_kw=dict(ecolor="0.25", capsize=2.5, lw=1.0),
                )
                ax.hlines(
                    total_med,
                    pos - bar_width / 2,
                    pos + bar_width / 2,
                    color="0.1",
                    lw=1.4,
                    zorder=5,
                    label="median" if first and k == 0 else None,
                )
                if k == 0:
                    tick_positions.append(pos + bar_width / 2)
                    tick_labels.append(mode)
        group_centres.append(
            base + ((len(MODES) - 1) * (2 * bar_width + 0.02)) / 2 + bar_width / 2
        )

    ax.set_xlim(-0.15, (len(rows) - 1) * group_gap + 0.95)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    for centre, label in zip(group_centres, group_labels):
        ax.annotate(
            label,
            xy=(centre, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -32),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


with open(JSON_FILE) as fid:
    data = json.load(fid)

if "fodo" in data:
    rows = data["fodo"]
    grouped_bars(
        rows,
        [f"{r['label']} FODO cells" for r in rows],
        [int(r["label"]) for r in rows],
        "Time per cell (ms)",
        "Mean track time per FODO cell\n"
        "(Xsuite vs MAD-NG, by tracking mode, renormalised per cell count)",
        HERE / "001_benchmark_plot.png",
    )

if "ir8" in data:
    rows = data["ir8"]
    grouped_bars(
        rows,
        [f"map {r['label']}" for r in rows],
        [1.0] * len(rows),
        "Time (ms)",
        f"Mean track time, HL-LHC IR8 s.ds.l8.b1 -> ip1.l1"
        f" ({rows[0]['n_elem']} elements)\n"
        f"Xsuite vs MAD-NG, by tracking mode and map order, 20 quadrupole knobs"
        f" ({rows[0]['n_driven']} driven fields).",
        HERE / "001_ir8_benchmark_plot.png",
    )
