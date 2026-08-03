# copyright ################################# #
# This file is part of the Xtrack Package.    #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
"""Benchmark native and GTPSA tracking on an HL-LHC IR8 optics segment.

The benchmark intentionally does only tracking:

- normal ``xt.Particles`` through the normal line;
- scalar ``xtpsa.ParticlesTpsa`` through a non-parametric line;
- parametric ``xtpsa.ParticlesTpsa`` through a line with the IR8 quadrupole
  circuit variables promoted to GTPSA parameters.

The IR8 knob list follows the optics matching setup in
``fast_optics_jacobian/utils.py``, but no matching or twiss computation is done.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import numpy as np

import xtrack as xt
import xtrack.tpsa as xtpsa
import xgtpsa


HERE = Path(__file__).resolve().parent
XTRACK_ROOT = HERE.parents[1]
COLLIDER_JSON = (
    XTRACK_ROOT / "test_data" / "hllhc15_thick" / "hllhc15_collider_thick.json"
)

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

START = "s.ds.l8.b1"
END = "ip1.l1"

X0 = {
    "x": 1.0e-6,
    "px": 2.0e-7,
    "y": -2.0e-6,
    "py": 1.0e-7,
    "zeta": 0.0,
    "delta": 1.0e-5,
}


def _scalar_ref_kwargs(line):
    ref = line.particle_ref
    out = {}
    for name in ("p0c", "mass0", "q0"):
        out[name] = float(np.asarray(getattr(ref, name)).reshape(-1)[0])
    return out


def _normal_particle(line):
    return xt.Particles(**_scalar_ref_kwargs(line), **X0)


def _tpsa_particle(line, order, descriptor=None):
    return xtpsa.ParticlesTpsa(order=order, descriptor=descriptor,
                               **_scalar_ref_kwargs(line), **X0)


def _enable_line_variable_tpsa(line, names, descriptor):
    for ii, name in enumerate(names):
        if name not in line.vars:
            raise KeyError(f"variable {name!r} is not in the line")
        line.vars[name] = descriptor.param(ii + 1, float(line[name]))


def _time_tracks(label, line, particles, repeats, ele_start, ele_stop):
    timings = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for pp in particles:
            t0 = time.perf_counter()
            line.track(pp, ele_start=ele_start, ele_stop=ele_stop)
            timings.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()

    mean = statistics.mean(timings)
    best = min(timings)
    stdev = statistics.stdev(timings) if repeats > 1 else 0.0
    print(
        f"{label:28s} "
        f"best {1e3 * best:9.3f} ms   "
        f"mean {1e3 * mean:9.3f} ms   "
        f"stdev {1e3 * stdev:8.3f} ms"
    )
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    print(f"Loading {COLLIDER_JSON}")
    collider = xt.load(COLLIDER_JSON)

    normal_line = collider.lhcb1
    scalar_line = normal_line.copy()
    param_line = normal_line.copy()

    start_idx = normal_line.element_names.index(args.start)
    end_idx = normal_line.element_names.index(args.end)
    if end_idx <= start_idx:
        raise ValueError(
            f"range {args.start!r}->{args.end!r} wraps around the cyclic line "
            "and is not supported by the current TPSA backend"
        )

    print(
        f"Tracking range: {args.start} -> {args.end} "
        f"({end_idx - start_idx} elements)"
    )
    print(f"TPSA map order: {args.order}")
    print(f"Parametric variables: {len(IR8_VARY)}")

    normal_line.build_tracker(use_prebuilt_kernels=False)
    scalar_line.build_tracker(use_prebuilt_kernels=False)
    param_line.build_tracker(use_prebuilt_kernels=False)

    param_descriptor = xgtpsa.Descriptor(
        6, args.order, num_params=len(IR8_VARY), param_order=1
    )
    _enable_line_variable_tpsa(param_line, IR8_VARY, param_descriptor)

    for _ in range(args.warmup):
        normal_line.track(_normal_particle(normal_line),
                          ele_start=args.start, ele_stop=args.end)
        scalar_line.track(_tpsa_particle(scalar_line, args.order),
                          ele_start=args.start, ele_stop=args.end)
        param_line.track(_tpsa_particle(param_line, args.order, param_descriptor),
                         ele_start=args.start, ele_stop=args.end)

    normal_particles = [_normal_particle(normal_line) for _ in range(args.repeats)]
    scalar_maps = [
        _tpsa_particle(scalar_line, args.order) for _ in range(args.repeats)
    ]
    param_maps = [
        _tpsa_particle(param_line, args.order, param_descriptor)
        for _ in range(args.repeats)
    ]

    print()
    _time_tracks("normal particles", normal_line, normal_particles,
                 args.repeats, args.start, args.end)
    _time_tracks("TPSA scalar line", scalar_line, scalar_maps,
                 args.repeats, args.start, args.end)
    _time_tracks("TPSA parametric line", param_line, param_maps,
                 args.repeats, args.start, args.end)


if __name__ == "__main__":
    main()
