"""Second-order chromaticity correction with arc quadrupole knl[3] offsets."""

from copy import copy
from types import SimpleNamespace

import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer


DEFAULT_SECOND_ORDER_CHROM_SETTINGS = SimpleNamespace(
    second_order_chrom_tol=1.,       # Tolerance on Q''/2, in both planes.
    response_step=0.01,              # k3 [m^-4]; integrated probe is step * length.
    delta_chrom=2e-4,                # Avoid cancellation in the second derivative.
    rcond=1e-8,
)
ARC_QUADRUPOLE_KNOBS = ('kqf2', 'kqd1')
KNOBS = ('dko_arc_f', 'dko_arc_d')
SWITCH = 'on_arc_octupole_corr'


def build_second_order_chrom_optimizer(line, reference, *, settings=None, **twiss_kwargs):
    """Target the second-order chromaticities of the shared clean reference."""
    if settings is None:
        settings = copy(DEFAULT_SECOND_ORDER_CHROM_SETTINGS)
    twiss_kwargs.setdefault('delta_chrom', settings.delta_chrom)
    knobs = line.metadata['correction_knobs']['second_order_chrom']['knobs']
    targets = [
        xt.Target('ddqx', value=reference.ddqx,
                  tol=2 * settings.second_order_chrom_tol, tag='ddqx'),
        xt.Target('ddqy', value=reference.ddqy,
                  tol=2 * settings.second_order_chrom_tol, tag='ddqy')
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
        name='second_order_chrom',
        vary=vary, targets=targets)
    optimizer.ideal_second_order_chrom = np.array([reference.ddqx, reference.ddqy])/2
    optimizer.second_order_chrom_tol = settings.second_order_chrom_tol
    optimizer.rcond = settings.rcond
    optimizer.set_response(build_second_order_chrom_response(
        line, reference, knobs=knobs, step=settings.response_step, **twiss_kwargs))
    return optimizer


def install_second_order_chrom_knobs(line):
    """Install shared k3 trims on quadrupoles controlled by the arc family knobs.

    Use xdeps to find all controlled quadrupoles, including through derived
    variables. Each receives an additive knl[3] = switch * trim * length term.
    The knobs represent k3 [m^-4], so no physical-magnet grouping is needed.
    Existing knl[3] expressions are preserved; k1 and its correction trims do
    not modulate the octupole term. Trims remain outside on_nonlinear.
    """
    assert all(knob not in line.vars for knob in KNOBS), 'Second-order chromaticity knobs already exist'
    table = line.get_table()
    quad_refs = {line.ref[name]: name for name in
                 table.rows.match(element_type='Quadrupole').env_name}
    groups = [[quad_refs[ref] for ref in line.ref[family].xdeps.controlled_targets
               if ref in quad_refs] for family in ARC_QUADRUPOLE_KNOBS]
    for family, names in zip(ARC_QUADRUPOLE_KNOBS, groups):
        if not names:
            raise ValueError(f'No quadrupoles controlled by {family}')
        for name in names:
            if line[name].length <= 0 or len(line[name].knl) < 4 or line[name].order < 3:
                raise ValueError(f'{name} must be a positive-length quadrupole with order >= 3')
    line[SWITCH] = 1.
    for knob, names in zip(KNOBS, groups):
        line[knob] = 0.
        for name in names:
            line.ref[name].knl[3] += line.ref[SWITCH] * line.ref[knob] * line[name].length
    circuits = line.metadata.setdefault('correction_knobs', {})
    circuits['second_order_chrom'] = dict(
        knobs=list(KNOBS), family_knobs=list(ARC_QUADRUPOLE_KNOBS),
        quadrupole_groups=groups, units='m^-4')
    circuits.setdefault('switches', {})['second_order_chrom'] = SWITCH
    return list(KNOBS)


def build_second_order_chrom_response(line, reference, *, knobs, step, **twiss_kwargs):
    """Response of Q'' (Twiss convention), restoring all probes even on failure."""
    twiss_kwargs.setdefault('delta_chrom', DEFAULT_SECOND_ORDER_CHROM_SETTINGS.delta_chrom)
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
