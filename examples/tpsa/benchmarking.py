"""Shared timing and reporting helpers for the TPSA tracking benchmarks."""

import gc
import json
import time
from pathlib import Path
from string import Template

import numpy as np
import xtrack as xt
import xtrack.tpsa as xtpsa
from madng_tpsa import Descriptor
from xdeps.refs import AttrRef
from xtrack.tpsa._knobs import KnobParameters

REPS = 50
WARMUP = 3
MODES = ("scalar", "tpsa", "param")
X0 = dict(x=1e-6, px=2e-7, y=-1e-6, py=1e-7, zeta=0.0, delta=0.0)
TRACK_OPTS = "save=false"

# Per element kind, mirroring the xsuite magnet configuration of the harness.
# model TKT = thick-kick-thick (xsuite bend-kick-bend, mat-kick-mat), DKD =
# drift-kick-drift over an exact drift. method is the integrator order, method 2
# with nslice n is n uniform kicks, so nslice mirrors num_multipole_kicks.
MADNG_INTEGRATION = {
    "sbend": dict(model="TKT", method=2, nslice=1),
    "rbend": dict(model="TKT", method=2, nslice=1),
    "quadrupole": dict(model="DKD", method="'teapot2'", nslice=1),
    "sextupole": dict(model="TKT", method=2, nslice=1),
    "octupole": dict(model="TKT", method=2, nslice=1),
}


def stats(times):
    """Mean of sample times in ms, its standard error, and sample spread statistics."""
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
    """Per-iteration wall time of ``call`` in ms, after warmup runs."""
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
    """Per-iteration setup and tracking times in ms, sampled together."""
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
    """``(element, attribute)`` pairs manipulated by the knob expressions."""
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
        knob_names="{" + ",".join(f"'{name}'" for name in knob_names) + "}",
        seed_knobs="\n    ".join(f"MADX['{name}'] = {value!r} + X['{name}']" for name, value in pairs),
        restore_knobs="\n    ".join(f"MADX['{name}'] = {value!r}" for name, value in pairs),
    )


def madng_split(mng, fresh):
    """Return setup and tracking samples, in ms, for one MAD-NG mode."""
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
    xs = {mode: split_timings_ms(build_mode, line.track) for mode, build_mode in build.items()}
    xs_knobs = timings_ms(refresh, reps=5, warmup=1)
    knobs.teardown()

    mng.send(get_madng_str(knob_names, knob_values, order)).send(list(X0.values()))
    ng = {mode: madng_split(mng, f"fresh_{mode}") for mode in MODES}
    mng.send("restore_knobs()")

    samples = {mode: dict(xs=xs[mode], ng=ng[mode]) for mode in MODES}

    def part(index):
        return {
            mode: {side: stats(value[index]) for side, value in samples[mode].items()}
            for mode in MODES
        }

    return dict(
        label=label,
        n_elem=len(line.element_names),
        n_driven=n_driven,
        xs_knobs=stats(xs_knobs),
        setup=part(0),
        track=part(1),
        total={
            mode: {side: stats(value[0] + value[1]) for side, value in samples[mode].items()}
            for mode in MODES
        },
    )


def set_madng_integration(mng, integration=MADNG_INTEGRATION, sequence_name="seq"):
    """Set model, method and nslice on the sequence elements, by element kind.

    Element attributes win over the track command options in MAD-NG, so this is
    what pins the integration scheme per magnet family. Kinds left out of
    ``integration`` keep the MAD-NG defaults.
    """
    entries = ", ".join(
        f"{kind}={{model='{opt['model']}', method={opt['method']}, nslice={opt['nslice']}}}"
        for kind, opt in integration.items()
    )
    mng.send(f"""
        local integration = {{{entries}}}
        MADX.{sequence_name}:foreach(function(element)
            local opt = integration[element.kind]
            if opt then
                element.model, element.method, element.nslice =
                    opt.model, opt.method, opt.nslice
            end
        end)
        """)


def to_madng(line, integration=MADNG_INTEGRATION):
    """Create a MAD-NG instance for a line and return its conversion time."""
    t0 = time.perf_counter()
    mng = line.to_madng(sequence_name="seq")
    set_madng_integration(mng, integration)
    return mng, time.perf_counter() - t0


def fmt(entry):
    """Mean plus standard error in ms, in a fixed-width cell."""
    return f"{entry['mean']:>10.4f} +/-{entry['sem']:>8.4f}"


def print_part(name, rows, key, caption):
    print(f"\n[{name}] {caption}")
    header = f"{'case':>10}" + "".join(f"{'xs_' + mode:>21}{'ng_' + mode:>21}" for mode in MODES)
    print(header)
    for row in rows:
        line = f"{row['label']:>10}"
        for mode in MODES:
            line += fmt(row[key][mode]["xs"]) + fmt(row[key][mode]["ng"])
        print(line)


def print_tables(name, rows):
    print(
        f"\n[{name}] {REPS} iterations, mean +/- standard error of the mean in ms "
        "(xs = xtrack+GTPSA, ng = MAD-NG in-process)"
    )
    print(f"{'case':>10}{'elems':>7}{'driven':>8}{'xs_knobs (ms)':>23}")
    for row in rows:
        print(f"{row['label']:>10}{row['n_elem']:>7}{row['n_driven']:>8}" + fmt(row["xs_knobs"]))

    print_part(name, rows, "total", "setup + track, one iteration")
    print_part(name, rows, "setup", "object setup only")
    print_part(name, rows, "track", "track only, setup subtracted")

    print(f"\n[{name}] ng/xs track ratio, and the xdeps part of xs_param")
    print(f"{'case':>10}" + "".join(f"{'ng/xs ' + mode:>13}" for mode in MODES) + f"{'us per field':>14}")
    for row in rows:
        line = f"{row['label']:>10}"
        for mode in MODES:
            track = row["track"][mode]
            line += f"{track['ng']['mean'] / track['xs']['mean']:>13.2f}"
        print(line + f"{row['xs_knobs']['mean'] * 1e3 / row['n_driven']:>14.2f}")

    print(f"\n[{name}] track sample spread, median [q25, q75] ms, for reference")
    print(f"{'case':>10}" + "".join(f"{'xs_' + mode:>24}{'ng_' + mode:>24}" for mode in MODES))
    for row in rows:
        line = f"{row['label']:>10}"
        for mode in MODES:
            for side in ("xs", "ng"):
                entry = row["track"][mode][side]
                line += f"{entry['med']:>9.3f} [{entry['q25']:>6.3f},{entry['q75']:>7.3f}]"
        print(line)


def write_report(path, scenario, rows, plot):
    """Write one self-contained benchmark report and return its contents."""
    report = dict(scenario=scenario, rows=rows, plot=plot)
    with open(path, "w") as fid:
        json.dump(report, fid, indent=1)
    print(f"\nwrote {path}")
    return report
