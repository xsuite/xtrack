import gc
import json
import sys
import time
from pathlib import Path
from string import Template

import numpy as np
import xtrack as xt
import xtrack.tpsa as xtpsa
from xdeps.refs import AttrRef
from xtrack.tpsa._knobs import KnobParameters
from madng_tpsa import Descriptor

RING_SIZES = [8, 64, 512, 2048]
REPS = 50
WARMUP = 3
ORDER = 3
SEED = 100
KL = 0.05  # fixed integrated quad strength
KQF, KQD = 1.0, -1.0
KNOBS = ["kqf", "kqd"]
KNOB_VALUES = [KQF, KQD]
IR8_RANGE = ("s.ds.l8.b1", "ip1.l1")
IR8_ORDERS = [1, 2, 3]
HERE = Path(__file__).resolve().parent
XTRACK_ROOT = HERE.parents[1]
LHC_JSON = XTRACK_ROOT / "test_data" / "hllhc15_thick" / "hllhc15_collider_thick.json"
LHC_MADX = XTRACK_ROOT / "test_data" / "hllhc15_thick" / "opt_round_150_1500.madx"
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
X0 = dict(x=1e-6, px=2e-7, y=-1e-6, py=1e-7, zeta=0.0, delta=0.0)
MODES = ("scalar", "tpsa", "param")
TRACK_OPTS = "method=2, nslice=1, model='TKT', save=false"
OUT_JSON = HERE / "001_benchmark_results.json"


def stats(times):
    """Mean of a sample of times in ms, its standard error, and where the median sits.

    Timings are a tight core plus a long tail of machine noise, so the mean sits above
    the median and sem is much smaller than std. Both are kept: sem is the error bar,
    std and the quartiles describe the spread of the sample itself.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    std = float(np.std(times, ddof=1))
    return dict(
        mean=float(times.mean()),
        sem=std / np.sqrt(times.size),
        med=float(np.median(times)),
        std=std,
        q25=float(np.percentile(times, 25)),
        q75=float(np.percentile(times, 75)),
    )


def timings_ms(call, reps=REPS, warmup=WARMUP):
    """Per-iteration wall time of call() in ms, after warmup runs."""
    for _ in range(warmup):
        call()
    times = np.empty(reps)
    gc.disable()
    try:
        for i in range(reps):
            t0 = time.perf_counter()
            call()
            times[i] = (time.perf_counter() - t0) * 1e3
    finally:
        gc.enable()
    return times


def split_timings_ms(build, track, reps=REPS, warmup=WARMUP):
    """Per-iteration (setup, track) times in ms, both read off the same iteration."""
    for _ in range(warmup):
        track(build())
    setup, tracked = np.empty(reps), np.empty(reps)
    gc.disable()
    try:
        for i in range(reps):
            t0 = time.perf_counter()
            particles = build()
            t1 = time.perf_counter()
            track(particles)
            t2 = time.perf_counter()
            setup[i] = (t1 - t0) * 1e3
            tracked[i] = (t2 - t1) * 1e3
    finally:
        gc.enable()
    return setup, tracked


def knob_driven_fields(line, names):
    """``(element, attribute)`` pairs the knob expressions manipulate, straight from xdeps.
    """
    fields = set()
    for name in names:
        for ref in line.vars[name]._find_dependant_targets():
            if isinstance(ref, AttrRef) and ref._owner._owner is line.element_refs:
                fields.add((ref._owner._key, ref._key))
    return sorted(fields)


MADNG_TEMPLATE = Template("""
local track, damap in MAD
local c0 = py:recv()
local reps, warmup = $reps, $warmup
local knobs = $knob_names

function fresh_scalar()
    return { x=c0[1], px=c0[2], y=c0[3], py=c0[4], t=0, pt=0 }
end

function fresh_tpsa()
    local X = damap { nv=6, mo=$order }
    X:set0(c0)
    return X
end

function fresh_param()
    local X = damap { nv=6, mo=$order, np=#knobs, po=1, pn=knobs }
    X:set0(c0)
    $seed_knobs
    return X
end

function restore_knobs()
    $restore_knobs
end

function bench(fresh)
    for i=1,warmup do track { sequence=MADX.seq, X0=fresh(), $track_opts } end
    collectgarbage()
    local ts, tt = MAD.vector(reps), MAD.vector(reps)
    for i=1,reps do
        local t0 = os.clock()
        local X = fresh()
        local t1 = os.clock()
        track { sequence=MADX.seq, X0=X, $track_opts }
        local t2 = os.clock()
        ts[i] = (t1 - t0) * 1e3
        tt[i] = (t2 - t1) * 1e3
    end
    return ts, tt
end
""")


def get_madng_str(knob_names, knob_values, order):
    """The Lua benchmark harness, specialized to one knob set."""
    pairs = list(zip(knob_names, knob_values))
    return MADNG_TEMPLATE.substitute(
        reps=REPS,
        warmup=WARMUP,
        order=order,
        track_opts=TRACK_OPTS,
        knob_names="{" + ",".join(f"'{n}'" for n in knob_names) + "}",
        seed_knobs="\n    ".join(f"MADX['{n}'] = {v!r} + X['{n}']" for n, v in pairs),
        restore_knobs="\n    ".join(f"MADX['{n}'] = {v!r}" for n, v in pairs),
    )


def madng_split(mng, fresh):
    """(setup, track) per-iteration times of one Lua mode, ms.

    os.clock has microsecond granularity, so the cheapest setups are quantized.
    Lua collects once before the loop, but its collector stays on, unlike gc.disable on the Python side.
    """
    mng.send(f"local ts, tt = bench({fresh}) py:send(ts) py:send(tt)")
    return mng.recv().reshape(-1), mng.recv().reshape(-1)


def bench_line(line, mng, knob_names, knob_values, label, order):
    """Track and setup times for the three modes, on both codes, for one line."""
    p0c = float(np.asarray(line.particle_ref.p0c).reshape(-1)[0])
    mass0 = float(np.asarray(line.particle_ref.mass0).reshape(-1)[0])
    kwargs = dict(p0c=p0c, mass0=mass0, **X0)

    plain = Descriptor(6, order)
    parametric = Descriptor(6, order, params=knob_names, param_order=1)
    knobs = KnobParameters(line, knob_names, parametric)
    n_driven = len(knob_driven_fields(line, knob_names))

    def refresh():
        knobs.refresh(knob_values)

    def build_param():
        refresh()
        return xtpsa.ParticlesTpsa(order=order, descriptor=parametric, **kwargs)

    build = dict(
        scalar=lambda: xt.Particles(**kwargs),
        tpsa=lambda: xtpsa.ParticlesTpsa(order=order, descriptor=plain, **kwargs),
        param=build_param,
    )
    xs = {m: split_timings_ms(f, line.track) for m, f in build.items()}
    xs_knobs = timings_ms(refresh, reps=5, warmup=1)
    knobs.teardown()

    mng.send(get_madng_str(knob_names, knob_values, order)).send(list(X0.values()))
    ng = {m: madng_split(mng, f"fresh_{m}") for m in MODES}
    mng.send("restore_knobs()")

    samples = {m: dict(xs=xs[m], ng=ng[m]) for m in MODES}

    def part(index):
        return {m: {s: stats(v[index]) for s, v in samples[m].items()} for m in MODES}

    return dict(
        label=label,
        n_elem=len(line.element_names),
        n_driven=n_driven,
        xs_knobs=stats(xs_knobs),
        setup=part(0),
        track=part(1),
        total={
            m: {s: stats(v[0] + v[1]) for s, v in samples[m].items()} for m in MODES
        },
    )


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


def build_ir8():
    """The IR8 section of HL-LHC b1, knobs and expressions shared with the collider."""
    collider = xt.Environment.from_json(LHC_JSON)
    collider.vars.load(LHC_MADX, format="madx")
    collider.build_trackers()
    line = collider.lhcb1.select(*IR8_RANGE)
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
    line.build_tracker()
    return line


def to_madng(line):
    """MAD-NG instance for a line, with the conversion time."""
    t0 = time.perf_counter()
    mng = line.to_madng(sequence_name="seq")
    return mng, time.perf_counter() - t0


def run_fodo():
    """One row per ring size, at ORDER."""
    rows = []
    for n_cells in RING_SIZES:
        line = build_fodo(n_cells)
        mng, convert_s = to_madng(line)
        row = bench_line(line, mng, KNOBS, KNOB_VALUES, str(n_cells), ORDER)
        row["convert_s"] = convert_s
        rows.append(row)
    return rows


def run_ir8():
    """One row per map order, on the same line and the same MAD-NG instance."""
    line = build_ir8()
    mng, convert_s = to_madng(line)
    values = [float(line.vars[name]._value) for name in IR8_VARY]
    rows = []
    for order in IR8_ORDERS:
        row = bench_line(line, mng, IR8_VARY, values, f"order {order}", order)
        row["convert_s"] = convert_s
        rows.append(row)
    return rows


def fmt(entry):
    """mean ± standard error of the mean, ms, in a fixed-width cell."""
    return f"{entry['mean']:>10.4f} ±{entry['sem']:>8.4f}"


def print_part(name, rows, key, caption):
    print(f"\n[{name}] {caption}")
    header = f"{'case':>10}" + "".join(
        f"{'xs_' + m:>21}{'ng_' + m:>21}" for m in MODES
    )
    print(header)
    for r in rows:
        line = f"{r['label']:>10}"
        for mode in MODES:
            line += fmt(r[key][mode]["xs"]) + fmt(r[key][mode]["ng"])
        print(line)


def print_tables(name, rows):
    print(
        f"\n[{name}] {REPS} iterations, mean ± standard error of the mean in ms "
        "(xs = xtrack+GTPSA, ng = MAD-NG in-process)"
    )
    print(f"{'case':>10}{'elems':>7}{'driven':>8}{'xs_knobs (ms)':>23}")
    for r in rows:
        print(
            f"{r['label']:>10}{r['n_elem']:>7}{r['n_driven']:>8}"
            + fmt(r["xs_knobs"])
        )

    print_part(name, rows, "total", "setup + track, one iteration")
    print_part(name, rows, "setup", "object setup only")
    print_part(name, rows, "track", "track only, setup subtracted")

    print(f"\n[{name}] ng/xs track ratio, and the xdeps part of xs_param")
    print(
        f"{'case':>10}"
        + "".join(f"{'ng/xs ' + m:>13}" for m in MODES)
        + f"{'us per field':>14}"
    )
    for r in rows:
        line = f"{r['label']:>10}"
        for mode in MODES:
            t = r["track"][mode]
            line += f"{t['ng']['mean'] / t['xs']['mean']:>13.2f}"
        print(line + f"{r['xs_knobs']['mean'] * 1e3 / r['n_driven']:>14.2f}")

    print(f"\n[{name}] track sample spread, median [q25, q75] ms, for reference")
    print(f"{'case':>10}" + "".join(f"{'xs_' + m:>24}{'ng_' + m:>24}" for m in MODES))
    for r in rows:
        line = f"{r['label']:>10}"
        for mode in MODES:
            for side in ("xs", "ng"):
                e = r["track"][mode][side]
                line += f"{e['med']:>9.3f} [{e['q25']:>6.3f},{e['q75']:>7.3f}]"
        print(line)


cases = sys.argv[1:] or ["fodo", "ir8"]
out = {}
if "fodo" in cases:
    out["fodo"] = run_fodo()
if "ir8" in cases:
    out["ir8"] = run_ir8()

for name, rows in out.items():
    print_tables(name, rows)

with open(OUT_JSON, "w") as fid:
    json.dump(out, fid, indent=1)
print(f"\nwrote {OUT_JSON}")
