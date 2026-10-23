"""Tune targets, response, metrics, goals and standalone correction stage."""

from functools import partial
from types import SimpleNamespace
import numpy as np
import xtrack as xt

import correction_setup as setup
from correction_optimizer import CorrectionOptimizer


def tune_settings(*, tune_tol=1e-5, max_iterations=7):
    """Return editable settings for this correction."""
    return SimpleNamespace(
        tune_tol=tune_tol,
        max_iterations=max_iterations,
    )


def build_tune_corr_optimizer(line, reference, *, full_twiss, settings=None):
    """Build a correction optimizer with local targets, response and diagnostics."""
    settings = settings if settings is not None else tune_settings()
    action = xt.Action(full_twiss)
    circuits = line.metadata['correction_knobs']['tunes']
    knobs = circuits['knobs']
    response = setup.build_tune_response(
        reference, line=line, tune_knobs=knobs, full_twiss=full_twiss)
    optimizer = CorrectionOptimizer(
        line, reference, settings=settings,
        compute_metrics=compute_metrics, goal=goal,
        configure_goals=configure_goals, print_metrics=print_metrics,
        name='tunes',
        nominal_knob_values=circuits['nominal_values'],
        vary=[xt.Vary(knob, step=1e-7) for knob in knobs],
        targets=[action.target(q, value=reference[q], tol=settings.tune_tol / 2, tag=q)
                 for q in ('qx', 'qy')])
    optimizer.rcond = 0.
    optimizer.set_response(response)
    optimizer.configure_goals(final_refinement=False, error_fraction=0.)

    optimizer.correct = partial(correct, optimizer)
    return optimizer


def compute_metrics(optimizer, tw):
    reference = optimizer.reference
    tunes = np.array([tw.qx, tw.qy])
    return dict(tunes=tunes, tune_error=tunes - np.array([reference.qx, reference.qy]))


def tune_goal(tw, reference, *, tolerance):
    error = np.array([tw.qx - reference.qx, tw.qy - reference.qy])
    return bool(np.all(np.abs(error) < tolerance))


def configure_goals(optimizer, *, final_refinement, error_fraction):
    optimizer.tune_tol = optimizer.settings.tune_tol
    optimizer.cleanup_tol = optimizer.settings.tune_tol / 2
    for target in optimizer.targets:
        target.tol = optimizer.cleanup_tol


def goal(optimizer, tw):
    return tune_goal(tw, optimizer.reference, tolerance=optimizer.tune_tol)


def correct(optimizer, tw, *, full_twiss):
    """Tighter tune cleanup; the caller supplies Twiss and owns rollback."""
    for iteration in range(optimizer.settings.max_iterations):
        if np.all(optimizer.target_status(ret=True, action_results=[tw]).tol_met):
            return tw
        optimizer.run_lstsq(action_results=[tw], tag_outcome=False)
        tw = full_twiss(tw)
    if np.all(optimizer.target_status(ret=True, action_results=[tw]).tol_met):
        return tw
    raise RuntimeError('Tune correction iteration limit')


def print_metrics(optimizer, tw, *, log=print):
    m = optimizer.compute_metrics(tw)
    log('Tunes:', m['tunes'], 'error', m['tune_error'])
