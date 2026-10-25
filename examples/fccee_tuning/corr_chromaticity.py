"""Chromaticity correction using four sextupole family offsets."""

from copy import copy
from functools import partial
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


def build_chromaticity_corr_optimizer(line, reference, *, full_twiss, settings=None):
    """Build a fixed family response and target the aligned machine's chromaticities.

    Evaluate the ideal machine with nominal nonlinear strengths. The workflow
    uses this correction during refinement, after increasing the errors.
    """
    if settings is None:
        settings = copy(DEFAULT_CHROMATICITY_CORR_SETTINGS)
    knobs = line.metadata['correction_knobs']['chromaticity']['knobs']
    chromatic_twiss = partial(full_twiss, chrom=True)
    nonlinear = float(line['on_nonlinear'])
    try:
        line['on_nonlinear'] = 1.
        ideal = chromatic_twiss(co_guess=reference.particle_on_co)
        action = xt.Action(chromatic_twiss)
        optimizer = CorrectionOptimizer(
            line, ideal, settings=settings,
            compute_metrics=compute_metrics, goal=goal, print_metrics=print_metrics,
            name='chromaticity',
            vary=[xt.Vary(knob, step=settings.response_step) for knob in knobs],
            targets=[xt.Target(quantity, value=ideal[quantity], action=action,
                               tol=settings.chromaticity_tol, tag=quantity)
                     for quantity in ('dqx', 'dqy')])
        optimizer.ideal_chromaticities = np.array([ideal.dqx, ideal.dqy])
        optimizer.chromaticity_tol = settings.chromaticity_tol
        optimizer.rcond = settings.rcond
        optimizer.set_response(build_chromaticity_response(
            line, ideal, knobs=knobs, full_twiss=chromatic_twiss,
            step=settings.response_step))
    finally:
        line['on_nonlinear'] = nonlinear
    return optimizer


def build_chromaticity_response(line, reference, *, knobs, full_twiss, step):
    """Numerical d(dqx,dqy)/d(family offset), restoring every probe."""
    baseline = np.array(line.get(knobs))
    response = np.empty((2, len(knobs)))
    try:
        for column, knob in enumerate(knobs):
            line[knob] = baseline[column] + step
            plus = full_twiss(co_guess=reference.particle_on_co)
            line[knob] = baseline[column] - step
            minus = full_twiss(co_guess=reference.particle_on_co)
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
