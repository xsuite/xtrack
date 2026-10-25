"""Second-order chromaticity correction with arc quadrupole knl[3] offsets."""

from copy import copy
from functools import partial
from types import SimpleNamespace

import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer


DEFAULT_SECOND_ORDER_CHROM_SETTINGS = SimpleNamespace(
    second_order_chrom_tol=1.,       # Tolerance on Q''/2, in both planes.
    response_step=0.02,              # Integrated K3L per physical magnet [m^-3].
    delta_chrom=2e-4,                # Avoid cancellation in the second derivative.
    rcond=1e-8,
)
ARC_QUADRUPOLE_FAMILIES = ('qf2a', 'qd1a')
KNOBS = ('dko_arc_f', 'dko_arc_d')
SWITCH = 'on_arc_octupole_corr'


def install_second_order_chrom_knobs(line):
    """Install/reuse additive K3L offsets, length-weighted across physical halves.

    Original errors and expressions are preserved. Trims are outside
    on_nonlinear: the workflow scales their physical strength exactly once.
    """
    records = line.metadata['tuning_errors']['magnets'].values()
    groups = [[g['names'] for g in records if g['category'] == 'arc_quad'
               and g['names'][0].split('.')[0] == family]
              for family in ARC_QUADRUPOLE_FAMILIES]
    circuits = line.metadata.setdefault('correction_knobs', {})
    existing = circuits.get('second_order_chrom')
    if existing is not None:
        if (existing['knobs'] != list(KNOBS) or existing['magnet_groups'] != groups
                or SWITCH not in line.vars):
            raise ValueError('Conflicting second-order chromaticity circuit metadata')
    elif SWITCH in line.vars or any(knob in line.vars for knob in KNOBS):
        raise ValueError('Second-order chromaticity knobs exist without circuit metadata')
    for knob, magnets in zip(KNOBS, groups):
        if not magnets:
            raise ValueError(f'No arc quadrupoles found for {knob}')
        if existing is not None and (knob not in line.vars or line.ref[knob].xdeps.expr is not None):
            raise ValueError(f'Conflicting second-order chromaticity knob {knob}')
        for names in magnets:
            for name in names:
                if (not isinstance(line.get(name), xt.Quadrupole)
                        or line[name].length <= 0 or len(line[name].knl) < 4
                        or line[name].order < 3):
                    raise ValueError(f'{name} must be a positive-length quadrupole with order >= 3')
                if existing is not None:
                    deps = line.ref[name].knl[3].xdeps.expr_dependencies
                    if line.ref[knob] not in deps or line.ref[SWITCH] not in deps:
                        raise ValueError(f'Second-order chromaticity knob {knob} is not attached to {name}')
    if existing is None:
        line[SWITCH] = 1.
        for knob, magnets in zip(KNOBS, groups):
            line[knob] = 0.
            for names in magnets:
                length = sum(line[name].length for name in names)
                for name in names:
                    line.ref[name].knl[3] += (line.ref[SWITCH] * line.ref[knob]
                                            * line[name].length / length)
        circuits['second_order_chrom'] = dict(
            knobs=list(KNOBS), families=list(ARC_QUADRUPOLE_FAMILIES),
            magnet_groups=groups, units='total K3L per physical quadrupole [m^-3]')
    circuits.setdefault('switches', {})['second_order_chrom'] = SWITCH
    return list(KNOBS)


def build_second_order_chrom_optimizer(line, reference, *, full_twiss, settings=None):
    if settings is None:
        settings = copy(DEFAULT_SECOND_ORDER_CHROM_SETTINGS)
    knobs = install_second_order_chrom_knobs(line)
    chromatic_twiss = partial(full_twiss, chrom=True)
    nonlinear = float(line['on_nonlinear'])
    try:
        line['on_nonlinear'] = 1.
        ideal = chromatic_twiss(co_guess=reference.particle_on_co)
        action = xt.Action(chromatic_twiss)
        optimizer = CorrectionOptimizer(
            line, ideal, settings=settings,
            compute_metrics=compute_metrics, goal=goal, print_metrics=print_metrics,
            name='second_order_chrom',
            vary=[xt.Vary(knob, step=settings.response_step) for knob in knobs],
            targets=[xt.Target(quantity, value=ideal[quantity], action=action,
                               tol=2*settings.second_order_chrom_tol, tag=quantity)
                     for quantity in ('ddqx', 'ddqy')])
        optimizer.ideal_second_order_chrom = np.array([ideal.ddqx, ideal.ddqy])/2
        optimizer.second_order_chrom_tol = settings.second_order_chrom_tol
        optimizer.rcond = settings.rcond
        optimizer.set_response(build_second_order_chrom_response(
            line, ideal, knobs=knobs, full_twiss=chromatic_twiss, step=settings.response_step))
    finally:
        line['on_nonlinear'] = nonlinear
    return optimizer


def build_second_order_chrom_response(line, reference, *, knobs, full_twiss, step):
    """Response of Q'' (Twiss convention), restoring all probes even on failure."""
    baseline = np.array(line.get(knobs))
    response = np.empty((2, len(knobs)))
    try:
        for column, knob in enumerate(knobs):
            line[knob] = baseline[column] + step
            plus = full_twiss(co_guess=reference.particle_on_co)
            line[knob] = baseline[column] - step
            minus = full_twiss(co_guess=reference.particle_on_co)
            response[:, column] = np.array([plus.ddqx-minus.ddqx,
                                             plus.ddqy-minus.ddqy])/(2*step)
            line[knob] = baseline[column]
    finally:
        line.set(knobs, baseline)
    return response


def compute_metrics(optimizer, tw):
    if 'ddqx' not in tw or 'ddqy' not in tw:
        return dict(second_order_chrom=None, second_order_chrom_error=None)
    values = np.array([tw.ddqx, tw.ddqy])/2
    return dict(second_order_chrom=values,
                second_order_chrom_error=values-optimizer.ideal_second_order_chrom)


def goal(optimizer, tw):
    error = optimizer.compute_metrics(tw)['second_order_chrom_error']
    return error is not None and bool(np.all(np.abs(error) < optimizer.second_order_chrom_tol))


def print_metrics(optimizer, tw, *, log=print):
    metrics = optimizer.compute_metrics(tw)
    if metrics['second_order_chrom'] is not None:
        log('Second-order chromaticities Q\'\'/2:', metrics['second_order_chrom'],
            'error', metrics['second_order_chrom_error'])
