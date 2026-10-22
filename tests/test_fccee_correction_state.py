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
    state.compute_metrics = lambda tw, **kwargs: metrics
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
    correction.compute_metrics(None)['beta_rms'][0] = 0.5
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


def test_module_goals_preserve_thresholds(correction, monkeypatch):
    correction.reference = correction.accepted_twiss
    metrics = correction.compute_metrics(None)
    correction.monitor_indices = np.array([0])
    correction.coupling_indices = correction.ip_indices = np.array([0])
    monkeypatch.setattr(importlib.import_module('orbit_corr'), 'compute_metrics',
                        lambda *args, **kwargs: metrics)
    monkeypatch.setattr(importlib.import_module('coupling_corr'), 'compute_metrics',
                        lambda *args, **kwargs: metrics)
    assert correction.coupling_goal(None)
    metrics['orbit_rms'][:] = correction.error_fraction * correction.settings.orbit_tol
    assert correction.orbit_goal(None)  # Inclusive orbit limit.
    metrics['orbit_rms'][0] = np.nextafter(metrics['orbit_rms'][0], np.inf)
    assert not correction.orbit_goal(None)
    metrics['dy_ip'] = 0.00005
    assert correction.coupling_goal(None)
    correction.final_refinement = True
    assert not correction.coupling_goal(None)
    correction.settings.dispersion_ip_tol = 0.00006
    assert correction.coupling_goal(None)  # Settings remain live.


def test_optics_goal_computes_errors_without_aggregate_metrics(correction, monkeypatch):
    correction.reference = SimpleNamespace(
        betx=np.ones(1000), bety=np.ones(1000), dx=np.zeros(1000), qx=0., qy=0.)
    correction.dispersion_indices = np.array([0, 1])
    correction.ip_indices = np.array([1])
    tw = deepcopy(correction.reference)
    monkeypatch.setattr(correction, 'compute_metrics', lambda *a, **k: pytest.fail('Unexpected aggregate metrics'))
    monkeypatch.setattr(correction, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    assert correction.optical_goal(tw)

    # Beta checks cover the full table, including points outside the BPM/IP list.
    for plane in ('betx', 'bety'):
        beta = getattr(tw, plane)
        beta[:] = 1.04
        assert not correction.optical_goal(tw)  # RMS above 3%, maximum below 50%.
        beta[:] = 1.
        beta[-1] = 1.5
        assert not correction.optical_goal(tw)  # At maximum limit, RMS below 3%.
        beta[-1] = 1.
    for tune in ('qx', 'qy'):
        setattr(tw, tune, -correction.settings.tune_tol)
        assert not correction.optical_goal(tw)  # Strict and sign-independent.
        setattr(tw, tune, 0.)

    tw.dx[-1] = 1.
    assert correction.optical_goal(tw)  # Dispersion is checked only at BPMs/IPs.
    tw.dx[0] = 0.01
    assert not correction.optical_goal(tw)  # RMS fails although IP error is zero.
    tw.dx[0] = 0.
    tw.dx[1] = -correction.settings.horizontal_dispersion_ip_tol
    assert correction.optical_goal(tw)  # Looser ramp limit.
    correction.final_refinement = True
    assert not correction.optical_goal(tw)  # Exactly at the final IP limit.
    tw.dx[1] = np.nextafter(tw.dx[1], 0.)
    assert correction.optical_goal(tw)
    correction.settings.horizontal_dispersion_ip_tol *= 0.5
    assert not correction.optical_goal(tw)  # Settings remain live.


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
                        settings.horizontal_dispersion_weight
                        * settings.horizontal_dispersion_ip_weight, 1000., 1000.])
    monkeypatch.setattr(module, 'build_normal_quad_optics_response_matrix',
                        lambda *args: physical_response.copy())

    def no_tracking():
        raise AssertionError('Unexpected Twiss')

    monkeypatch.setattr(correction.line, 'twiss', no_tracking)
    opt = module.build_optimizer(
        correction.line, reference, knobs=['quad', 'skew'],
        observation_names=['bpm'], interaction_points=['bpm'], ip_names=['bpm'],
        settings=correction.settings)
    supplied = [trial]
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
    correction.optics_weights = target_weights
    assert status.penalty == pytest.approx(
        np.linalg.norm(weights * relative_residuals))


@pytest.mark.parametrize('response, accepted', [(0.1, False), (1., True)])
def test_optics_keeps_worsening_steps_until_goal_or_budget(correction, monkeypatch, response, accepted):
    import xdeps as xd

    correction.line['quad'] = 1.
    def no_tracking():
        raise AssertionError('Optimizer must use supplied results')

    action = xt.Action(no_tracking)
    baseline = {'error': 0.75, 'inactive': 0.}
    opt = xd.Optimize(
        vary=[xd.Vary('quad', correction.line.vars, step=1e-6)],
        targets=[action.target('error', value=0., tol=1e-12, weight=2.),
                 action.target('inactive', value=0., tol=1e-12)],
        action_results={action: baseline}, show_call_counter=False)
    opt.targets[1].active = False
    opt.set_response(np.array([[response], [0.]]))
    correction.optimizers = SimpleNamespace(optics=opt)
    trials = []

    def evaluate_trial(tw):
        knob = float(correction.line['quad'])
        trials.append(knob)
        return {'error': knob**2 - 0.25, 'inactive': 1000.}

    monkeypatch.setattr(correction, '_full_twiss', evaluate_trial)
    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw: abs(tw['error']) < 0.2)
    monkeypatch.setattr(correction, 'orbit_goal', lambda tw: True)
    monkeypatch.setattr(correction, 'optical_goal', lambda tw: False)
    monkeypatch.setattr(correction, 'coupling_goal', lambda tw: True)
    monkeypatch.setattr(correction, 'correct_tunes', lambda tw: tw)
    before = correction.snapshot()
    if accepted:
        result, iterations = correction.correct_at_fixed_strengths(baseline)
        assert iterations == 1
        np.testing.assert_allclose(trials, [0.25])
        assert correction.line['quad'] == pytest.approx(0.25)
        assert result['error'] == pytest.approx(-0.1875)
    else:
        with pytest.raises(RuntimeError, match='goals not met after 5 iterations'):
            correction.correct_at_fixed_strengths(baseline)
        assert len(trials) == 5
        assert trials[0] == pytest.approx(-6.5)
        assert all(abs(b) > abs(a) for a, b in zip(trials, trials[1:]))
        assert correction.line['quad'] == trials[-1]
        # The ramp/refinement caller owns rollback after a stage fails.
        correction.restore(before)
        assert correction.line['quad'] == 1.
    assert len(opt.log()) == 1


def test_explicit_stage_order_passes_latest_twiss(correction, monkeypatch):
    calls = []
    def optimizer(name):
        def apply(*, action_results, **kwargs):
            calls.append((name, action_results[0]))
        return SimpleNamespace(run_lstsq=apply)

    correction.optimizers = SimpleNamespace(**{
        name: optimizer(name) for name in ('orbit_x', 'orbit_y', 'optics', 'coupling')})
    def stage(name, advance=1):
        def run(tw):
            calls.append((name, tw))
            return tw + advance
        return run

    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw: tw == 6)
    for name in ('orbit_goal', 'optical_goal', 'coupling_goal'):
        monkeypatch.setattr(correction, name, lambda tw: False)
    monkeypatch.setattr(correction, '_full_twiss', stage('full'))
    monkeypatch.setattr(correction, 'orbit_twiss', stage('orbit'))
    monkeypatch.setattr(correction, 'correct_tunes', stage('tune'))
    monkeypatch.setattr(correction, 'settle_coupling_orbit', stage('settle', advance=0))
    monkeypatch.setattr(correction, 'snapshot', lambda: pytest.fail('Unexpected stage snapshot'))
    assert correction.correct_at_fixed_strengths(0) == (6, 1)
    assert calls == [('orbit_x', 0), ('orbit_y', 0), ('full', 0), ('tune', 1),
                     ('optics', 2), ('full', 2), ('tune', 3), ('settle', 4),
                     ('coupling', 4), ('orbit', 4), ('settle', 5), ('tune', 5)]
    assert correction.correct_at_fixed_strengths(6) == (6, 0)
    assert len(calls) == 12  # No work at an already accepted point.


@pytest.mark.parametrize('final_refinement, limit', [(False, 5), (True, 40)])
@pytest.mark.parametrize('converges', [False, True])
def test_cycle_budget_checks_goals_after_last_cycle(correction, monkeypatch,
                                                   final_refinement, limit, converges):
    correction.final_refinement = final_refinement
    logged = []
    tune_calls = []
    monkeypatch.setattr(correction, 'log_iteration', lambda iteration, tw: logged.append(iteration))
    monkeypatch.setattr(correction, 'goals_met', lambda tw: converges and logged[-1] == limit)
    for name in ('orbit_goal', 'optical_goal', 'coupling_goal'):
        monkeypatch.setattr(correction, name, lambda tw: True)
    monkeypatch.setattr(correction, 'correct_tunes', lambda tw: tune_calls.append(tw) or tw)
    if converges:
        assert correction.correct_at_fixed_strengths(0) == (0, limit)
    else:
        with pytest.raises(RuntimeError, match=f'goals not met after {limit} iterations'):
            correction.correct_at_fixed_strengths(0)
    assert logged == list(range(limit + 1))
    assert len(tune_calls) == 3 * limit


@pytest.mark.parametrize('twiss_fails', [False, True])
def test_standalone_tune_correction(correction, monkeypatch, twiss_fails):
    import xdeps as xd

    action = xt.Action(lambda: pytest.fail('Unexpected optimizer evaluation'))
    baseline = np.array([12., 3.])
    opt = xd.Optimize(
        vary=[xd.Vary(name, correction.line.vars, step=1e-6) for name in ('tune', 'quad')],
        targets=[action.target(i, value=value, tol=1e-15)
                 for i, value in enumerate([1., 2.])],
        action_results={action: baseline}, show_call_counter=False)
    opt.set_response(np.eye(2))
    correction.optimizers = SimpleNamespace(tunes=opt)
    trials = []
    def evaluate(tw):
        values = np.array([float(correction.line[name]) for name in ('tune', 'quad')])
        trials.append(values)
        if twiss_fails:
            raise ValueError('Invalid trial optics')
        return values

    monkeypatch.setattr(correction, '_full_twiss', evaluate)
    if twiss_fails:
        before = correction.snapshot()
        with pytest.raises(ValueError, match='Invalid trial optics'):
            correction.correct_tunes(baseline)
        assert len(trials) == 1
        correction.restore(before)
        np.testing.assert_allclose([correction.line[name] for name in ('tune', 'quad')], baseline)
    else:
        result = correction.correct_tunes(baseline)
        np.testing.assert_allclose(result, [1., 2.])
        assert len(trials) == 1
        assert correction.correct_tunes(result) is result
        assert len(trials) == 1  # Goal check does not run Twiss again.
    assert len(opt.log()) == 1


def test_tunes_allow_non_improving_iterations_until_limit(correction, monkeypatch):
    import xdeps as xd

    correction.line['tune'] = 1.
    action = xt.Action(lambda: pytest.fail('Unexpected optimizer evaluation'))
    opt = xd.Optimize(
        vary=[xd.Vary('tune', correction.line.vars, step=1e-6)],
        targets=[action.target(0, value=0., tol=1e-12)],
        action_results={action: [2.]}, show_call_counter=False)
    # The actual response is twice this value, so full steps oscillate with
    # unchanged penalty. Only the tune iteration limit should reject them.
    opt.set_response(np.array([[1.]]))
    correction.optimizers = SimpleNamespace(tunes=opt)
    trials = []
    def evaluate(tw):
        value = 2 * float(correction.line['tune'])
        trials.append(value)
        return [value]
    monkeypatch.setattr(correction, '_full_twiss', evaluate)
    before = correction.snapshot()
    with pytest.raises(RuntimeError, match='Tune correction iteration limit'):
        correction.correct_tunes([2.])
    assert trials == [-2., 2., -2., 2., -2., 2., -2.]
    correction.restore(before)
    assert correction.line['tune'] == 1.


@pytest.mark.parametrize('outcome', ['improves', 'worsens', 'fails'])
def test_settling_restores_failed_trial_before_continuing(correction, monkeypatch, outcome):
    calls = []
    before = correction.snapshot()
    def apply(**kwargs):
        correction.line['orbit'] += 1.
    correction.optimizers = SimpleNamespace(
        orbit_x=SimpleNamespace(run_lstsq=apply),
        orbit_y=SimpleNamespace(run_lstsq=lambda **kwargs: None))
    def evaluate(kind):
        def run(tw):
            calls.append(kind)
            if outcome == 'fails':
                correction.line.twiss_default['co_guess'] = {'x': 999.}
                raise ValueError('Invalid orbit')
            if outcome == 'worsens':
                return -1
            return tw + 1
        return run

    monkeypatch.setattr(correction, 'correctable_orbit_errors',
                        lambda tw: [np.array([1.]), np.array([2.])])
    monkeypatch.setattr(correction, 'orbit_twiss', evaluate('orbit'))
    monkeypatch.setattr(correction, '_full_twiss', evaluate('full'))
    def ensure_full(tw):
        if outcome == 'fails':
            assert correction.line['orbit'] == before.knob_values['orbit']
            assert correction.line.twiss_default['co_guess'] == before.co_guess
        return ('full', tw)
    monkeypatch.setattr(correction, 'ensure_full_twiss', ensure_full)
    monkeypatch.setattr(correction, 'correct_tunes', lambda tw: pytest.fail('Nested tune correction'))
    result = correction.settle_coupling_orbit(0)
    if outcome == 'improves':
        assert result == ('full', 3)
        assert calls == ['orbit', 'orbit', 'full']
    elif outcome == 'worsens':
        assert result == ('full', -1)
        assert calls == ['orbit', 'orbit', 'full']
    else:
        assert result == ('full', 0)
        assert calls == ['orbit']

    calls.clear()
    monkeypatch.setattr(correction, 'correctable_orbit_errors',
                        lambda tw: [np.array([0.]), np.array([0.])])
    assert correction.settle_coupling_orbit(0) == ('full', 0)
    assert calls == []


def test_module_metrics_match_reference_helpers(correction, monkeypatch):
    import xdeps as xd

    obs = importlib.import_module('correction_observables')
    orbit = importlib.import_module('orbit_corr')
    columns = ['name', 'betx', 'bety', 'dx', 'dy', 'f1001', 'f1010', 'x', 'y']
    reference = xd.Table(dict(
        name=np.array(['bpm1', 'other', 'bpm2', 'ip']),
        betx=np.array([10., 20., 30., 40.]), bety=np.array([15., 25., 35., 45.]),
        dx=np.array([.1, .2, .3, .4]), dy=np.array([.01, .02, .03, .04]),
        f1001=np.array([.01j, .02, -.03j, .04]),
        f1010=np.array([-.01, .02j, .03, -.04j]),
        x=np.zeros(4), y=np.zeros(4), qx=.31, qy=.32, c_minus=0.), col_names=columns)
    tw = xd.Table({key: deepcopy(reference[key])
                   for key in columns + ['qx', 'qy', 'c_minus']}, col_names=columns)
    tw.betx[:] *= [1.01, 1.2, .98, 1.03]
    tw.bety[:] *= [.99, .8, 1.02, .97]
    tw.dx[:] += [.001, .1, -.002, .00004]
    tw.dy[:] += [.0001, .2, -.0002, -.00001]
    tw.f1001[:] += [.001 + .002j, .3, -.002j, .0004]
    tw.f1010[:] += [.002j, -.4j, .001, .0002j]
    tw.x[:] = [1e-6, 1., -2e-6, 0.]
    tw.y[:] = [-3e-6, 2., 4e-6, 0.]
    tw['qx'], tw['qy'], tw['c_minus'] = .310001, .319998, -.0002
    correction.reference = reference
    correction.monitor_indices = np.array([0, 2])
    correction.dispersion_indices = correction.coupling_indices = np.array([0, 2, 3])
    correction.ip_indices = np.array([3])
    correction.monitor_alignment = lambda: (
        np.array([.1e-6, -.2e-6]), np.array([-.3e-6, .4e-6]), np.array([.1, -.2]))
    del correction.compute_metrics  # Use the real aggregator instead of the state-test stub.
    monkeypatch.setattr(correction, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    expected = obs.measure(
        tw, tw_ref=reference, dispersion_indices=correction.dispersion_indices,
        ip_indices=correction.ip_indices,
        orbit_errors=lambda table: obs.orbit_errors(
            table, monitor_indices=correction.monitor_indices,
            monitor_alignment=correction.monitor_alignment))
    actual = correction.compute_metrics(tw)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14, atol=0)
    np.testing.assert_allclose(correction.coupling_vector(tw), obs.coupling_vector(
        tw, tw_ref=reference, coupling_indices=correction.coupling_indices))
    np.testing.assert_allclose(correction.lab_orbit_errors(tw), obs.lab_orbit_errors(
        tw, monitor_indices=correction.monitor_indices,
        monitor_alignment=correction.monitor_alignment))
    # Individual modules need only their own observables, not a full diagnostic table.
    orbit_only = SimpleNamespace(x=tw.x, y=tw.y)
    np.testing.assert_allclose(orbit.compute_metrics(
        orbit_only, monitor_indices=correction.monitor_indices,
        monitor_alignment=correction.monitor_alignment)['orbit_rms'], expected['orbit_rms'])
    assert not hasattr(correction, 'goal_optimizers')
