"""State operations can be checked without building the full FCC-ee response."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import xtrack as xt


@pytest.fixture
def correction(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / 'examples' / 'fccee_tuning'
    monkeypatch.syspath_prepend(str(examples))
    module = importlib.import_module('correction_workflow')
    # The full-lattice comparison exercises construction and physics. Here we
    # isolate snapshots/history from the expensive FCC-specific preparation.
    state = object.__new__(module.FCCeeCorrection)
    state.line = xt.Line(elements=[xt.Drift(length=1)])
    state.line.vars.update(
        dict(orbit=2.0, quad=3.0, skew=4.0, tune=12.0, on_errors=0.2, on_nonlinear=0.2)
    )
    state.line.vars['bpm_offset'] = 0.001 * state.line.ref['on_errors']
    state.line.metadata['tuning_errors'] = dict(seed=42)
    state.orbit_knobs, state.quad_knobs, state.skew_knobs, state.tune_knobs = (
        ['orbit'],
        ['quad'],
        ['skew'],
        ['tune'],
    )
    state.correction_knobs = ['orbit', 'quad', 'skew', 'tune']
    state.control_names = state.correction_knobs + ['on_errors', 'on_nonlinear']
    state.nominal_knob_values = dict(orbit=0.0, quad=0.0, skew=0.0, tune=10.0)
    state.settings = module.CorrectionSettings()
    state.log = lambda *args, **kwargs: None
    state.history = []
    state.final_refinement = False
    state.accepted_twiss = SimpleNamespace(
        x=[1.0], px=[2.0], y=[3.0], py=[4.0], qx=1.2, qy=2.3, delta=[0.0], c_minus=1e-4
    )
    state.line.twiss_default['co_guess'] = dict(x=1.0, px=2.0, y=3.0, py=4.0, zeta=0.0, delta=0.0)
    metrics = dict(
        orbit_rms=np.array([1e-6, 2e-6]),
        beta_rms=np.array([0.01, 0.02]),
        beta_max=np.array([0.1, 0.2]),
        tunes=np.array([1.2, 2.3]),
        tune_error=np.zeros(2),
        dx_rms=0.001,
        dx_ip=0.00001,
        dy_rms=0.0001,
        dy_ip=0.00001,
        coupling_rms=np.array([0.001, 0.002]),
        coupling_ip=0.001,
        c_minus=0.0001,
    )
    state.measure = lambda tw, **kwargs: metrics
    yield state
    state.__dict__.clear()


def test_scaling_is_separate_from_setting_strength(correction):
    correction.scale_corrections(0.4)
    assert correction.error_fraction == 0.2
    assert correction.line['on_nonlinear'] == 0.2
    assert correction._values(correction.correction_knobs).tolist() == [4.0, 6.0, 8.0, 14.0]
    correction.set_error_fraction(0.4)
    assert correction.line['on_errors'] == correction.line['on_nonlinear'] == 0.4
    assert correction.line['bpm_offset'] == 0.0004
    before = correction.snapshot()
    with pytest.raises(ValueError):
        correction.scale_corrections(float('nan'))
    with pytest.raises(ValueError):
        correction.set_error_fraction(1.1)
    assert correction.snapshot().knob_values == before.knob_values
    correction.set_error_fraction(0)
    correction.scale_corrections(0.1)  # No previous finite-fraction solution to extrapolate.
    assert correction._values(correction.correction_knobs).tolist() == [4.0, 6.0, 8.0, 14.0]


def test_snapshot_restores_all_trial_state_and_preserves_expressions(correction):
    before = correction.snapshot()
    correction.scale_corrections(0.5)
    correction.set_error_fraction(0.5)
    correction.line.twiss_default['co_guess']['x'] = 999
    correction.final_refinement = True
    correction.record(correction.accepted_twiss, iterations=2)
    correction.restore(before)
    assert correction.snapshot().knob_values == before.knob_values
    assert correction.line.twiss_default['co_guess'] == before.co_guess
    assert correction.accepted_twiss is before.accepted_twiss
    assert correction.history == [] and not correction.final_refinement
    correction.set_error_fraction(0.3)
    assert correction.line['bpm_offset'] == 0.0003


def test_history_is_independent_and_keeps_refinement(correction):
    correction.record(correction.accepted_twiss, iterations=2)
    first = deepcopy(correction.history[0])
    correction.line['quad'] = 99
    correction.measure(None)['beta_rms'][0] = 0.5
    correction.final_refinement = True
    correction.record(correction.accepted_twiss, iterations=1)
    assert correction.history[0] == first
    assert len(correction.history) == 2
    assert correction.history[-1]['final_refinement']
    assert correction.history_rows == [correction.history[-1]['row']]
    rows = correction.history_rows
    rows[0][0] = 999
    assert correction.history[0] == first


def test_twiss_evaluation_is_explicit(correction, monkeypatch):
    module = importlib.import_module('correction_workflow')
    correction.steps_r_matrix = dict(dx=1e-8)
    correction.stats = {}
    calls = []

    def compute(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(module.steps, 'compute_twiss', compute)
    first = correction.twiss()
    second = correction.twiss()
    assert second is not first
    assert len(calls) == 2
    assert calls[0][0][0] is correction.accepted_twiss
    correction.orbit_twiss(second)
    assert calls[-1][0][0] is second
    assert calls[-1][1]['only_orbit']

    # A supplied point satisfying the goals requires no further Twiss.
    correction.orbit_goal = correction.optical_goal = correction.coupling_goal = lambda tw: True
    tw, iterations = correction.correct_at_fixed_strengths(second)
    assert tw is second and iterations == 0
    assert len(calls) == 3


def test_aggregate_goal_targets_preserve_thresholds(correction):
    correction.reference = correction.accepted_twiss
    metrics = correction.measure(None)
    correction.orbit_errors = lambda tw: [np.array([value]) for value in metrics['orbit_rms']]
    correction._build_goal_optimizers()
    assert correction.optical_goal(None)
    assert correction.coupling_goal(None)
    metrics['orbit_rms'][:] = correction.error_fraction * correction.settings.orbit_tol
    assert correction.orbit_goal(None)  # Inclusive orbit limit.
    metrics['orbit_rms'][0] = np.nextafter(metrics['orbit_rms'][0], np.inf)
    assert not correction.orbit_goal(None)
    metrics['beta_rms'][0] = correction.settings.beta_rms_tol
    assert not correction.optical_goal(None)  # Strict optics limit.
    metrics['beta_rms'][0] = 0.01
    metrics['dy_ip'] = 0.00005
    assert correction.coupling_goal(None)
    correction.final_refinement = True
    assert not correction.coupling_goal(None)
    correction.settings.dispersion_ip_tol = 0.00006
    assert correction.coupling_goal(None)  # Settings remain live.
