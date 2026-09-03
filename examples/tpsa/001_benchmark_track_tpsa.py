"""Benchmark TPSA tracking on randomized FODO rings."""

from pathlib import Path

import numpy as np
import xtrack as xt

from benchmarking import bench_line, print_tables, to_madng, write_report
from plotting import plot_report

RING_SIZES = [8, 64, 512, 2048]
ORDER = 3
SEED = 100
KL = 0.05  # Fixed integrated quadrupole strength.
KQF, KQD = 1.0, -1.0
KNOBS = ["kqf", "kqd"]
KNOB_VALUES = [KQF, KQD]
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "001_fodo_benchmark_results.json"


def build_fodo(n_cells):
    """Randomized FODO ring, two knobs, k1*l fixed so the ring stays stable."""
    rng = np.random.default_rng(SEED)
    elements, names, weights = [], [], {}
    for cell in range(n_cells):
        l_qf, l_qd = rng.uniform(0.4, 0.6, size=2)
        l_d1, l_d2 = rng.uniform(1.8, 2.2, size=2)
        w_f, w_d = rng.uniform(0.98, 1.02, size=2)
        k_f, k_d = KL / l_qf * w_f, KL / l_qd * w_d
        elements += [
            xt.Quadrupole(length=l_qf, k1=k_f * KQF),
            xt.Drift(length=l_d1),
            xt.Quadrupole(length=l_qd, k1=k_d * KQD),
            xt.Drift(length=l_d2),
        ]
        names += [f"qf_{cell}", f"d1_{cell}", f"qd_{cell}", f"d2_{cell}"]
        weights[f"qf_{cell}"], weights[f"qd_{cell}"] = float(k_f), float(k_d)
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    line.vars["kqf"] = KQF
    line.vars["kqd"] = KQD
    for cell in range(n_cells):
        line.element_refs[f"qf_{cell}"].k1 = weights[f"qf_{cell}"] * line.vars["kqf"]
        line.element_refs[f"qd_{cell}"].k1 = weights[f"qd_{cell}"] * line.vars["kqd"]
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
    line.build_tracker()
    return line


def main():
    rows = []
    for n_cells in RING_SIZES:
        line = build_fodo(n_cells)
        mng, convert_s = to_madng(line)
        row = bench_line(line, mng, KNOBS, KNOB_VALUES, str(n_cells), ORDER)
        row["convert_s"] = convert_s
        rows.append(row)

    print_tables("fodo", rows)
    report = write_report(
        OUT_JSON,
        "fodo",
        rows,
        plot=dict(
            group_labels=[f"{row['label']} FODO cells" for row in rows],
            divisors=[int(row["label"]) for row in rows],
            ylabel="Time per cell (ms)",
            title=(
                "Mean track time per FODO cell\n"
                "(Xsuite vs MAD-NG, by tracking mode, renormalised per cell count)"
            ),
        ),
    )
    plot_report(report)


if __name__ == "__main__":
    main()
