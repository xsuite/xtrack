"""Plot reports emitted by ``001_benchmark_track_tpsa.py``."""

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
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


def grouped_bars(rows, group_labels, divisors, ylabel, title):
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
    return fig


def plot_report(report, output_file=None, show=True):
    """Plot one benchmark report, optionally saving it in addition to displaying it."""
    fig = grouped_bars(report["rows"], **report["plot"])
    if output_file is not None:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"saved {output_file}")
    if show:
        plt.show()
    return fig


def load_report(path):
    with open(path) as fid:
        return json.load(fid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        default=[
            HERE / "001_fodo_benchmark_results.json",
            HERE / "002_ir8_benchmark_results.json",
        ],
    )
    parser.add_argument("-o", "--output", type=Path, help="Write the plot to this file")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    args = parser.parse_args()

    if args.output is not None and len(args.reports) != 1:
        parser.error("--output requires exactly one report")

    for path in args.reports:
        plot_report(load_report(path), output_file=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
