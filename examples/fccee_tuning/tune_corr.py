"""Tune targets, response, metrics and standalone correction stage."""

import numpy as np
import xtrack as xt

import correction_setup as setup


def compute_metrics(tw, reference):
    tunes = np.array([tw.qx, tw.qy])
    return dict(tunes=tunes, tune_error=tunes - np.array([reference.qx, reference.qy]))


def tune_goal(tw, reference, *, tolerance):
    metrics = compute_metrics(tw, reference)
    return bool(np.all(np.abs(metrics['tune_error']) < tolerance))


def correct(tw, *, optimizer, action, tolerance, full_twiss, snapshot, restore):
    """Correct tunes using supplied Twiss and caller-owned evaluation/rollback."""
    for target in optimizer.targets:
        target.tol = tolerance
    for iteration in range(7):
        status = optimizer.target_status(ret=True, action_results={action: tw})
        if np.all(status.tol_met):
            return tw
        before = snapshot()
        try:
            optimizer.run_lstsq(action_results={action: tw},
                                rcond=0.0, tag_outcome=False)
            trial = full_twiss(tw)
            trial_status = optimizer.target_status(ret=True, action_results={action: trial})
            if trial_status.penalty < status.penalty:
                tw = trial
                continue
        except (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError) as err:
            restore(before)
            raise RuntimeError('Tune correction failed') from err
        restore(before)
        raise RuntimeError('No improving tune step')
    status = optimizer.target_status(ret=True, action_results={action: tw})
    if np.all(status.tol_met):
        return tw
    raise RuntimeError('Tune correction iteration limit')


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
