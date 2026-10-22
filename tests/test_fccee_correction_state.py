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


def test_optics_targets_use_relative_beta_and_physical_alpha(correction, monkeypatch):
    import xdeps as xd

    module = importlib.import_module('optics_corr')
    columns = ['name', 's', 'betx', 'bety', 'alfx', 'alfy', 'dx']
    reference = xd.Table(dict(
        name=np.array(['bpm']), s=np.array([0.]), line_length=100.,
        betx=np.array([10.]), bety=np.array([20.]),
        alfx=np.array([1.]), alfy=np.array([-2.]), dx=np.array([0.1]),
        qx=0.31, qy=0.32), col_names=columns)
    trial = xd.Table(dict(
        name=np.array(['bpm']), s=np.array([0.]), line_length=100.,
        betx=np.array([12.]), bety=np.array([16.]),
        alfx=np.array([1.5]), alfy=np.array([-1.]), dx=np.array([0.15]),
        qx=0.32, qy=0.29), col_names=columns)
    response = np.array([[1., 0.2], [0.3, 2.], [1., -1.], [0.5, 1.],
                         [2., 0.], [0., 1.], [1., 1.]])
    physical_response = response.copy()
    physical_response[2:4] += np.array([1., -2.])[:, None] * response[:2]
    physical_response[:2] *= np.array([10., 20.])[:, None]
    scales = 1 / (4 * np.pi * np.max(np.abs(physical_response[-2:]), axis=0))
    settings = correction.settings
    weights = np.array([settings.ir_beta_weight, settings.ir_beta_weight,
                        settings.alpha_weight, settings.alpha_weight,
                        settings.horizontal_dispersion_weight, 1000., 1000.])
    monkeypatch.setattr(module, 'build_normal_quad_optics_response_matrix',
                        lambda *args: physical_response.copy())
    monkeypatch.setattr(module.setup, 'check_optics_response', lambda *args, **kwargs: None)
    correction.line.metadata['correction_knobs'] = dict(
        optics=dict(knobs=['quad', 'skew'], quadrupoles=[]))

    def no_tracking():
        raise AssertionError('Unexpected Twiss')

    action = xt.Action(no_tracking)
    opt = module.build_optimizer(
        correction.line, reference, action, beta_at=['bpm'], alpha_at=['bpm'],
        dispersion_at=['bpm'], interaction_points=['bpm'], ip_names=[],
        settings=correction.settings, full_twiss=no_tracking, log=correction.log)
    supplied = {action: trial}
    status = opt.target_status(ret=True, action_results=supplied)
    # Physical values in status; normalized relative-beta objective in the solve.
    np.testing.assert_allclose(status.current_val, [12., 16., 1.5, -1., 0.15, 0.32, 0.29])
    np.testing.assert_allclose(status.target_val, [10., 20., 1., -2., 0.1, 0.31, 0.32])
    relative_residuals = np.array([0.2, -0.2, 0.5, 1., 0.05, 0.01, -0.03])
    target_weights = np.array([target.weight for target in opt.targets])
    np.testing.assert_allclose(target_weights * status.residue, weights * relative_residuals)
    # Recover alpha from a model linear in relative beta and normalized alpha,
    # then finite-difference the physical observable to check the row conversion.
    def model(knobs):
        changes = response @ knobs
        changes[2:4] = (np.array([1., -2.]) + changes[2:4]) * (1 + changes[:2])
        return changes

    probe = 1e-6 * np.eye(2)
    physical_response = np.column_stack([
        (model(direction) - model(-direction)) / (2e-6) for direction in probe])
    np.testing.assert_allclose(
        target_weights[:, None] * opt.response.matrix, weights[:, None] * physical_response,
        atol=1e-8)
    expected = scales * np.linalg.lstsq(
        weights[:, None] * physical_response * scales,
        -weights * relative_residuals, rcond=1e-12)[0]
    np.testing.assert_allclose(opt.get_correction(action_results=supplied, rcond=1e-12), expected)
    correction.optimizers = SimpleNamespace(optics=opt)
    correction.twiss_action = action
    correction.optics_weights = target_weights
    assert correction.optical_cost(trial) == pytest.approx(
        np.linalg.norm((weights * relative_residuals)[:-2]))
