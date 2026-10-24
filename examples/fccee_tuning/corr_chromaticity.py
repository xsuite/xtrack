"""Chromaticity correction using four arc sextupole family offsets."""

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

ARC_SEXTUPOLE_FAMILIES = ('sf1a', 'sf2a', 'sd1a', 'sd2a')


def install_chromaticity_knobs(line):
    """Add or reuse one independent k2 offset per arc sextupole family.

    Original family variables and IR/straight sextupoles are unchanged. Offsets
    are added outside on_nonlinear, like the other correction trims: their
    physical strength is scaled once by the workflow's scale_corrections.
    """
    table = line.get_table()
    sextupoles = list(dict.fromkeys(table.env_name[table.element_type == 'Sextupole']))
    groups = [[name for name in sextupoles if name.split('.')[0] == family]
              for family in ARC_SEXTUPOLE_FAMILIES]
    knobs = [f'dk{family[:-1]}_chromaticity' for family in ARC_SEXTUPOLE_FAMILIES]
    switch = 'on_chromaticity_corr'
    for family, names, knob in zip(ARC_SEXTUPOLE_FAMILIES, groups, knobs):
        if not names:
            raise ValueError(f'No arc sextupoles found for {family}')
        if knob in line.vars:
            if line.ref[knob].xdeps.expr is not None or switch not in line.vars:
                raise ValueError(f'Conflicting chromaticity knob {knob}')
            for name in names:
                dependencies = line.ref[name].k2.xdeps.expr_dependencies
                if line.ref[knob] not in dependencies or line.ref[switch] not in dependencies:
                    raise ValueError(f'Existing chromaticity knob {knob} is not attached to {name}')
    if switch not in line.vars:
        line[switch] = 1.
    for names, knob in zip(groups, knobs):
        if knob in line.vars:
            continue
        line[knob] = 0.
        for name in names:
            line.ref[name].k2 += line.ref[switch] * line.ref[knob]
    line.metadata.setdefault('correction_knobs', {})['chromaticity'] = dict(
        knobs=knobs, families=list(ARC_SEXTUPOLE_FAMILIES),
        sextupole_groups=groups, units='m^-3')
    return knobs


def build_chromaticity_corr_optimizer(line, reference, *, full_twiss, settings=None):
    """Build a fixed family response and target the aligned machine's chromaticities.

    Evaluate the ideal machine with nominal nonlinear strengths. The workflow
    uses this correction during refinement, after increasing the errors.
    """
    if settings is None:
        settings = copy(DEFAULT_CHROMATICITY_CORR_SETTINGS)
    knobs = install_chromaticity_knobs(line)
    chromatic_twiss = partial(full_twiss, chrom=True)
    nonlinear = float(line['on_nonlinear'])
    try:
        line['on_nonlinear'] = 1.
        ideal = chromatic_twiss(reference)
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
            plus = full_twiss(reference)
            line[knob] = baseline[column] - step
            minus = full_twiss(reference)
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
