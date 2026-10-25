"""Chromaticity correction using four sextupole family offsets."""

from copy import copy
from types import SimpleNamespace

import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer


DEFAULT_CHROMATICITY_CORR_SETTINGS = SimpleNamespace(
    chromaticity_tol=0.1,
    response_step=1e-4,
    rcond=1e-8,
)

SEXTUPOLE_FAMILY_KNOBS = ('ksf1', 'ksf2', 'ksd1', 'ksd2')


def build_chromaticity_corr_optimizer(line, reference, *, settings=None, **twiss_kwargs):
    """Build a fixed family response and target the aligned machine's chromaticities.

    The shared reference includes chromaticities at nominal nonlinear strength.
    The workflow uses this correction during refinement, after increasing errors.
    """
    if settings is None:
        settings = copy(DEFAULT_CHROMATICITY_CORR_SETTINGS)
    twiss_kwargs.setdefault('delta_chrom', 2e-4)
    knobs = line.metadata['correction_knobs']['chromaticity']['knobs']
    targets = [
        xt.Target('dqx', value=reference.dqx,
                  tol=settings.chromaticity_tol, tag='dqx'),
        xt.Target('dqy', value=reference.dqy,
                  tol=settings.chromaticity_tol, tag='dqy')
    ]
    vary = []
    for knob in knobs:
        vary.append(xt.Vary(knob, step=settings.response_step))
    optimizer = CorrectionOptimizer(
        line=line, reference_twiss=reference, settings=settings,
        method='4d', delta0=0., chrom=True, radiation_integrals=False,
        allow_twiss_failure=False,
        co_guess=dict(x=0, px=0, y=0, py=0, zeta=0, delta=0),
        **twiss_kwargs,
        compute_metrics=compute_metrics, goal=goal, print_metrics=print_metrics,
        name='chromaticity',
        vary=vary, targets=targets)
    optimizer.ideal_chromaticities = np.array([reference.dqx, reference.dqy])
    optimizer.chromaticity_tol = settings.chromaticity_tol
    optimizer.rcond = settings.rcond
    optimizer.set_response(build_chromaticity_response(
        line, reference, knobs=knobs, step=settings.response_step, **twiss_kwargs))
    return optimizer


def install_chromaticity_knobs(line):
    """Install new family trims initialized to zero and return their names.

    Append to the existing family expressions. All dependent sextupoles inherit
    the trims and their existing on_nonlinear scaling.
    """
    families = SEXTUPOLE_FAMILY_KNOBS
    knobs = [f'd{family}_chromaticity' for family in families]
    switch = 'on_chromaticity_corr'
    assert all(knob not in line.vars for knob in knobs), 'Chromaticity correction knobs already exist'
    line[switch] = 1.
    for family, knob in zip(families, knobs):
        line[knob] = 0.
        line.ref[family] += line.ref[switch] * line.ref[knob]
    line.metadata.setdefault('correction_knobs', {})['chromaticity'] = dict(
        knobs=knobs, family_knobs=list(families), units='m^-3')
    return knobs


def build_chromaticity_response(line, reference, *, knobs, step, **twiss_kwargs):
    """Numerical d(dqx,dqy)/d(family offset), restoring every probe."""
    twiss_kwargs.setdefault('delta_chrom', 2e-4)
    baseline = np.array(line.get(knobs))
    response = np.empty((2, len(knobs)))
    try:
        for column, knob in enumerate(knobs):
            line[knob] = baseline[column] + step
            plus = line.twiss(method='4d', delta0=0., chrom=True, radiation_integrals=False,
                               co_guess=reference.particle_on_co, **twiss_kwargs)
            line[knob] = baseline[column] - step
            minus = line.twiss(method='4d', delta0=0., chrom=True, radiation_integrals=False,
                               co_guess=reference.particle_on_co, **twiss_kwargs)
            response[:, column] = np.array([plus.dqx - minus.dqx,
                                             plus.dqy - minus.dqy]) / (2 * step)
            line[knob] = baseline[column]
    finally:
        line.set(knobs, baseline)
    return response


def compute_metrics(optimizer, tw):
    if 'dqx' not in tw or 'dqy' not in tw:
        return dict(chromaticities=None, chromaticity_error=None)
    values = np.array([tw.dqx, tw.dqy])
    return dict(chromaticities=values,
                chromaticity_error=values - optimizer.ideal_chromaticities)


def goal(optimizer, tw):
    error = optimizer.compute_metrics(tw)['chromaticity_error']
    return error is not None and bool(np.all(np.abs(error) < optimizer.chromaticity_tol))


def print_metrics(optimizer, tw, *, log=print):
    metrics = optimizer.compute_metrics(tw)
    if metrics['chromaticities'] is not None:
        log('Chromaticities:', metrics['chromaticities'], 'error', metrics['chromaticity_error'])
