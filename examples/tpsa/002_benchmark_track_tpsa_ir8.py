"""Benchmark TPSA tracking on the HL-LHC IR8 section."""

from pathlib import Path

import xtrack as xt

from benchmarking import bench_line, print_tables, to_madng, write_report
from plotting import plot_report

IR8_RANGE = ("s.ds.l8.b1", "ip1.l1")
IR8_ORDERS = [1, 2, 3, 4]
IR8_VARY = [
    "kq6.l8b1",
    "kq7.l8b1",
    "kq8.l8b1",
    "kq9.l8b1",
    "kq10.l8b1",
    "kqtl11.l8b1",
    "kqt12.l8b1",
    "kqt13.l8b1",
    "kq4.l8b1",
    "kq5.l8b1",
    "kq4.r8b1",
    "kq5.r8b1",
    "kq6.r8b1",
    "kq7.r8b1",
    "kq8.r8b1",
    "kq9.r8b1",
    "kq10.r8b1",
    "kqtl11.r8b1",
    "kqt12.r8b1",
    "kqt13.r8b1",
]
HERE = Path(__file__).resolve().parent
XTRACK_ROOT = HERE.parents[1]
LHC_JSON = XTRACK_ROOT / "test_data" / "hllhc15_thick" / "hllhc15_collider_thick.json"
LHC_MADX = XTRACK_ROOT / "test_data" / "hllhc15_thick" / "opt_round_150_1500.madx"
OUT_JSON = HERE / "002_ir8_benchmark_results.json"


def build_ir8():
    """The IR8 section of HL-LHC b1, with the collider knob expressions."""
    collider = xt.Environment.from_json(LHC_JSON)
    collider.vars.load(LHC_MADX, format="madx")
    collider.build_trackers()
    line = collider.lhcb1.select(*IR8_RANGE)
    # Paired with MADNG_INTEGRATION in benchmarking.py.
    line.configure_bend_model('bend-kick-bend', edge="full", integrator='uniform', num_multipole_kicks=1)
    line.configure_quadrupole_model('drift-kick-drift-exact', integrator='teapot', num_multipole_kicks=2)
    line.configure_sextupole_model('drift-kick-drift-exact')
    line.configure_octupole_model('drift-kick-drift-exact')
    line.configure_drift_model('exact')
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
    line.build_tracker()
    return line


def main():
    line = build_ir8()
    mng, convert_s = to_madng(line)
    values = [float(line.vars[name]._value) for name in IR8_VARY]
    rows = []
    for order in IR8_ORDERS:
        row = bench_line(line, mng, IR8_VARY, values, f"order {order}", order)
        row["convert_s"] = convert_s
        rows.append(row)

    print_tables("ir8", rows)
    report = write_report(
        OUT_JSON,
        "ir8",
        rows,
        plot=dict(
            group_labels=[f"map {row['label']}" for row in rows],
            divisors=[1.0] * len(rows),
            ylabel="Time (ms)",
            title=(
                "Mean track time, HL-LHC IR8 s.ds.l8.b1 -> ip1.l1 "
                f"({rows[0]['n_elem']} elements)\n"
                "Xsuite vs MAD-NG, by tracking mode and map order, 20 quadrupole knobs "
                f"({rows[0]['n_driven']} driven fields)."
            ),
        ),
    )
    plot_report(report)


if __name__ == "__main__":
    main()
