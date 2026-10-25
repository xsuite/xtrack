"""Tune correction: targets, response, metrics, goals and reporting."""

from copy import copy
from types import SimpleNamespace
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer


DEFAULT_TUNE_CORR_SETTINGS = SimpleNamespace(
    tune_tol=1e-5,
)


def build_tune_corr_optimizer(line, reference, *, settings=None, **twiss_kwargs):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if settings is None:
        settings = copy(DEFAULT_TUNE_CORR_SETTINGS)
    circuits = line.metadata['correction_knobs']['tunes']
    knobs = circuits['knobs']
    response = build_tune_response(
        reference, line=line, tune_knobs=knobs, **twiss_kwargs)
    targets = [
        xt.Target('qx', value=reference.qx,
                  tol=settings.tune_tol, tag='qx'),
        xt.Target('qy', value=reference.qy,
                  tol=settings.tune_tol, tag='qy')
    ]
    vary = []
    for knob in knobs:
        vary.append(xt.Vary(knob, step=1e-7))
    optimizer = CorrectionOptimizer(
        line=line, reference_twiss=reference, settings=settings,
        method='4d', delta0=0., chrom=False, radiation_integrals=False,
        allow_twiss_failure=False,
        co_guess=dict(x=0, px=0, y=0, py=0, zeta=0, delta=0),
        **twiss_kwargs,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name='tunes',
        nominal_knob_values=circuits['nominal_values'],
        vary=vary, targets=targets)
    optimizer.rcond = 0.
    optimizer.set_response(response)
    optimizer.tune_tol = settings.tune_tol

    return optimizer


def install_tune_knobs(line, *, families=('kqf2', 'kqd1')):
    """Install new family trims initialized to zero and return their names.

    Append to the existing family expressions so all dependent magnets follow
    the corrections without changing their individual strength expressions.
    """
    knobs = [f'd{family}_tune' for family in families]
    switch = 'on_tune_corr'
    assert all(knob not in line.vars for knob in knobs), 'Tune correction knobs already exist'
    line[switch] = 1.
    for family, knob in zip(families, knobs):
        line[knob] = 0.
        line.ref[family] += line.ref[switch] * line.ref[knob]
    line.metadata.setdefault('correction_knobs', {})['tunes'] = dict(
        knobs=knobs, nominal_values=dict.fromkeys(knobs, 0.),
        family_knobs=list(families), units='m^-2')
    return knobs


def compute_metrics(optimizer, tw):
    reference = optimizer.reference
    tunes = np.array([tw.qx, tw.qy])
    return dict(tunes=tunes, tune_error=tunes - np.array([reference.qx, reference.qy]))


def tune_goal(tw, reference, *, tolerance):
    error = np.array([tw.qx - reference.qx, tw.qy - reference.qy])
    return bool(np.all(np.abs(error) < tolerance))


def goal(optimizer, tw):
    return tune_goal(tw, optimizer.reference, tolerance=optimizer.tune_tol)


def print_metrics(optimizer, tw, *, log=print):
    m = optimizer.compute_metrics(tw)
    log('Tunes:', m['tunes'], 'error', m['tune_error'])


def build_tune_response(tw, *, line, tune_knobs, **twiss_kwargs):
    """Numerical two-family tune response, with all probe settings restored."""
    baseline = np.array(line.get(tune_knobs))
    jac = np.empty((2, 2))
    try:
        for j in range(2):
            line[tune_knobs[j]] = baseline[j] + 1e-7
            probe = line.twiss(method='4d', delta0=0., chrom=False, radiation_integrals=False,
                               co_guess=tw.particle_on_co, **twiss_kwargs)
            jac[:, j] = (np.array([probe.qx, probe.qy]) - np.array([tw.qx, tw.qy])) / 1e-7
            line.set(tune_knobs, baseline)
    finally:
        line.set(tune_knobs, baseline)
    return jac
