"""Build the tune optimizer and its two-knob finite-difference response."""

import numpy as np
import xtrack as xt

import correction_setup as setup


def compute_metrics(tw, reference):
    tunes = np.array([tw.qx, tw.qy])
    return dict(tunes=tunes, tune_error=tunes - np.array([reference.qx, reference.qy]))


def tune_goal(tw, reference, *, tolerance):
    metrics = compute_metrics(tw, reference)
    return bool(np.all(np.abs(metrics['tune_error']) < tolerance))


def build_optimizer(line, reference, action, *, tune_tol, full_twiss):
    """Return an optimizer targeting the nominal tunes with a fixed response."""
    knobs = line.metadata['correction_knobs']['tunes']['knobs']
    response = setup.build_tune_response(
        reference, line=line, tune_knobs=knobs, full_twiss=full_twiss)
    optimizer = line.match(
        solve=False,
        name='tunes',
        vary=[xt.Vary(knob, step=1e-7) for knob in knobs],
        targets=[action.target(q, value=reference[q], tol=tune_tol / 2, tag=q)
                 for q in ('qx', 'qy')],
        action_results={action: reference},
        assert_within_tol=False,
        show_call_counter=False,
    )
    optimizer.set_response(response)
    return optimizer
