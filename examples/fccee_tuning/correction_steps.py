"""Twiss evaluation and variable access for FCC-ee correction."""

from time import perf_counter
import numpy as np


def orbit_guess(tw):
    return dict(
        x=float(tw.x[0]),
        px=float(tw.px[0]),
        y=float(tw.y[0]),
        py=float(tw.py[0]),
        zeta=0.0,
        delta=0.0,
    )


def values(line, names):
    return np.array([line[name] for name in names])


def set_values(line, names, vals):
    line.vars.update({name: float(value) for name, value in zip(names, vals)})


def compute_twiss(
    seed=None, co_guess=None, radiation_integrals=False, only_orbit=False, *, line, stats, steps
):
    kwargs = {} if seed is None else dict(co_guess=seed.particle_on_co)
    if co_guess is not None:
        kwargs['co_guess'] = co_guess
    started = perf_counter()
    stats['calls'] += 1
    if only_orbit:
        stats['orbit_calls'] += 1
    try:
        return line.twiss(
            chrom=False,
            steps_R_matrix=steps.copy(),
            radiation_integrals=radiation_integrals,
            only_orbit=only_orbit,
            **kwargs,
        )
    finally:
        elapsed = perf_counter() - started
        stats['seconds'] += elapsed
        if only_orbit:
            stats['orbit_seconds'] += elapsed


def ensure_full_twiss(tw, *, full_twiss):
    # Call only while the line still has the settings used to compute tw.
    # Use the tracked orbit as a guess; keep the ordinary closed-orbit search.
    if 'betx' in tw._col_names:
        return tw
    return full_twiss(tw)
