"""Tune correction: targets, response, metrics, goals and reporting."""

from copy import copy
from types import SimpleNamespace
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer


DEFAULT_TUNE_CORR_SETTINGS = SimpleNamespace(
    tune_tol=1e-5,
)


def install_tune_knobs(line):
    """Install tune trims, reset them to zero, and return their names.

    Preserve the k1 family expressions and use metadata to reuse installed trims.
    On first installation, measure each family's k1 coefficients through xdeps.
    """
    circuits = line.metadata['correction_knobs']['tunes']
    families = circuits.get('family_knobs', circuits['knobs'])
    family_nominal_values = circuits.get('family_nominal_values', circuits['nominal_values'])
    knobs = [f'd{family}_tune' for family in families]
    installed = circuits['knobs'] if 'family_knobs' in circuits else []
    coefficients = circuits.get('k1_coefficients')
    if coefficients is None:
        table = line.get_table()
        quad_refs = {line.ref[name]: name for name in
                     table.rows.match(element_type='Quadrupole').env_name}
        coefficients = []
        for family in families:
            if line.ref[family].xdeps.expr is not None:
                raise ValueError(f'Tune family {family} must be an independent variable')
            names = [quad_refs[ref] for ref in line.ref[family].xdeps.controlled_targets
                     if ref in quad_refs]
            if not names:
                raise ValueError(f'No quadrupoles controlled by {family}')
            value = float(line[family])
            before = np.array([line.get(name).k1 for name in names])
            try:
                line[family] = value + 1e-6
                slopes = (np.array([line.get(name).k1 for name in names]) - before) / 1e-6
            finally:
                line[family] = value
            coefficients.append({name: float(slope) for name, slope in zip(names, slopes)
                                 if slope != 0})
    for knob, slopes in zip(knobs, coefficients):
        line[knob] = 0.
        if knob not in installed:
            for name, slope in slopes.items():
                line.ref[name].k1 += slope * line.ref[knob]
    line.metadata['correction_knobs']['tunes'] = dict(
        knobs=knobs, nominal_values=dict.fromkeys(knobs, 0.),
        family_knobs=list(families), family_nominal_values=family_nominal_values,
        k1_coefficients=coefficients, units='m^-2')
    return knobs


def build_tune_corr_optimizer(line, reference, *, full_twiss, settings=None):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if settings is None:
        settings = copy(DEFAULT_TUNE_CORR_SETTINGS)
    action = xt.Action(full_twiss)
    knobs = install_tune_knobs(line)
    circuits = line.metadata['correction_knobs']['tunes']
    response = build_tune_response(
        reference, line=line, tune_knobs=knobs, full_twiss=full_twiss)
    optimizer = CorrectionOptimizer(
        line, reference, settings=settings,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name='tunes',
        nominal_knob_values=circuits['nominal_values'],
        vary=[xt.Vary(knob, step=1e-7) for knob in knobs],
        targets=[action.target(q, value=reference[q], tol=settings.tune_tol, tag=q)
                 for q in ('qx', 'qy')])
    optimizer.rcond = 0.
    optimizer.set_response(response)
    optimizer.tune_tol = settings.tune_tol

    return optimizer


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


def build_tune_response(tw, *, line, tune_knobs, full_twiss):
    """Numerical two-family tune response, with all probe settings restored."""
    baseline = np.array(line.get(tune_knobs))
    jac = np.empty((2, 2))
    try:
        for j in range(2):
            line[tune_knobs[j]] = baseline[j] + 1e-7
            probe = full_twiss(co_guess=tw.particle_on_co)
            jac[:, j] = (np.array([probe.qx, probe.qy]) - np.array([tw.qx, tw.qy])) / 1e-7
            line.set(tune_knobs, baseline)
    finally:
        line.set(tune_knobs, baseline)
    return jac
