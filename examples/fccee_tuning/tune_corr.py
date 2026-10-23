"""Tune targets, response, metrics, goals and standalone correction stage."""

from dataclasses import dataclass
import numpy as np
import xtrack as xt

import correction_setup as setup
from correction_optimizer import CorrectionOptimizer


@dataclass
class TuneSettings:
    tune_tol: float = 1e-5
    max_iterations: int = 7


def compute_metrics(tw, reference):
    tunes = np.array([tw.qx, tw.qy])
    return dict(tunes=tunes, tune_error=tunes - np.array([reference.qx, reference.qy]))


def tune_goal(tw, reference, *, tolerance):
    metrics = compute_metrics(tw, reference)
    return bool(np.all(np.abs(metrics['tune_error']) < tolerance))


class TuneCorrection(CorrectionOptimizer):
    def __init__(self, line, reference, *, full_twiss, settings=None):
        settings = settings if settings is not None else TuneSettings()
        action = xt.Action(full_twiss)
        circuits = line.metadata['correction_knobs']['tunes']
        knobs = circuits['knobs']
        response = setup.build_tune_response(
            reference, line=line, tune_knobs=knobs, full_twiss=full_twiss)
        super().__init__(
            line, reference, settings=settings, name='tunes',
            nominal_knob_values=circuits['nominal_values'],
            vary=[xt.Vary(knob, step=1e-7) for knob in knobs],
            targets=[action.target(q, value=reference[q], tol=settings.tune_tol / 2, tag=q)
                     for q in ('qx', 'qy')])
        self.rcond = 0.
        self.set_response(response)
        self.configure_goals(final_refinement=False, error_fraction=0.)

    def configure_goals(self, *, final_refinement, error_fraction):
        self.tune_tol = self.settings.tune_tol
        self.cleanup_tol = self.settings.tune_tol / 2
        for target in self.targets:
            target.tol = self.cleanup_tol

    def compute_metrics(self, tw):
        return compute_metrics(tw, self.reference)

    def goal(self, tw):
        return tune_goal(tw, self.reference, tolerance=self.tune_tol)

    def correct(self, tw, *, full_twiss):
        """Tighter tune cleanup; the caller supplies Twiss and owns rollback."""
        for iteration in range(self.settings.max_iterations):
            if np.all(self.target_status(ret=True, action_results=[tw]).tol_met):
                return tw
            self.run_lstsq(action_results=[tw], tag_outcome=False)
            tw = full_twiss(tw)
        if np.all(self.target_status(ret=True, action_results=[tw]).tol_met):
            return tw
        raise RuntimeError('Tune correction iteration limit')

    def print_metrics(self, tw, *, log=print):
        m = self.compute_metrics(tw)
        log('Tunes:', m['tunes'], 'error', m['tune_error'])
