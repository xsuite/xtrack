"""State operations can be checked without building the full FCC-ee response."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import xtrack as xt


def stub_optimizer(**kwargs):
    defaults = dict(goal=lambda tw: True, configure_goals=lambda **kw: None,
                    print_metrics=lambda tw, **kw: None,
                    correct=lambda tw, **kw: tw, step_scale=.5, orbit_tol=5e-8)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


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
    state.optimizers = SimpleNamespace(**{name: stub_optimizer() for name in
        ('optics', 'orbit_x', 'orbit_y', 'coupling', 'tunes')})
    state.final_refinement = False
    state.accepted_twiss = SimpleNamespace(
        x=[1.0], px=[2.0], y=[3.0], py=[4.0], qx=1.2, qy=2.3, delta=[0.0], c_minus=1e-4
    )
    state.line.twiss_default['co_guess'] = dict(x=1.0, px=2.0, y=3.0, py=4.0, zeta=0.0, delta=0.0)
    metrics = dict(
        orbit_x_rms=1e-6, orbit_y_rms=2e-6,
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
    tw, iterations = correction.correct_at_fixed_strengths(second)
    assert tw is second and iterations == 0
    assert len(calls) == 3


def diagnostic_optimizer(module_name, class_name, settings_name, reference):
    # Test metric/goal methods independently of the expensive response setup.
    module = importlib.import_module(module_name)
    opt = object.__new__(getattr(module, class_name))
    opt.settings = getattr(module, settings_name)()
    opt.reference = reference
    return opt


def test_mode_and_restore_refresh_active_goals(correction):
    reference = correction.accepted_twiss
    optics = diagnostic_optimizer('optics_corr', 'OpticsCorrection', 'OpticsSettings', reference)
    coupling = diagnostic_optimizer('coupling_corr', 'CouplingCorrection', 'CouplingSettings', reference)
    orbit = diagnostic_optimizer('orbit_corr', 'OrbitCorrection', 'OrbitSettings', reference)
    correction.optimizers = SimpleNamespace(optics=optics, coupling=coupling, orbit=orbit)
    correction.configure_goals()
    orbit.plane = 'x'
    orbit.compute_metrics = lambda tw: {'orbit_x_rms': orbit.orbit_tol}
    assert orbit.goal(None)  # Inclusive orbit limit.
    orbit.compute_metrics = lambda tw: {'orbit_x_rms': np.nextafter(orbit.orbit_tol, np.inf)}
    assert not orbit.goal(None)
    assert orbit.orbit_tol == pytest.approx(2e-6)
    assert optics.dx_ip_tol == 5e-4
    assert coupling.dy_ip_tol == 1e-4
    before = correction.snapshot()
    correction.set_error_fraction(.8)
    correction.final_refinement = True
    assert orbit.orbit_tol == pytest.approx(8e-6)
    assert optics.dx_ip_tol == 1e-4
    assert coupling.dy_ip_tol == 2e-5
    correction.restore(before)
    assert not correction.final_refinement
    assert orbit.orbit_tol == pytest.approx(2e-6)
    assert optics.dx_ip_tol == 5e-4
    assert coupling.dy_ip_tol == 1e-4
    correction.set_error_fraction(0.)
    assert orbit.orbit_tol == 2e-9
    # Settings are configuration; active parameters refresh explicitly.
    coupling.settings.ramp_dispersion_ip_tol = 6e-5
    correction.configure_goals()
    assert coupling.dy_ip_tol == 6e-5


@pytest.mark.parametrize('key, tolerance', [
    ('coupling_rms', 'rdt_rms_tol'), ('coupling_ip', 'rdt_ip_tol'),
    ('c_minus', 'c_minus_tol'), ('dy_rms', 'dy_rms_tol'), ('dy_ip', 'dy_ip_tol')])
def test_coupling_goal_limits_are_strict(correction, key, tolerance):
    opt = diagnostic_optimizer('coupling_corr', 'CouplingCorrection',
                               'CouplingSettings', correction.accepted_twiss)
    metrics = dict(coupling_rms=np.zeros(2), coupling_ip=0., c_minus=0.,
                   dy_rms=0., dy_ip=0.)
    opt.compute_metrics = lambda tw: metrics
    for final in (False, True):
        opt.configure_goals(final_refinement=final, error_fraction=1.)
        limit = getattr(opt, tolerance)
        metrics[key] = -limit if key == 'c_minus' else limit
        assert not opt.goal(None)
        metrics[key] = np.nextafter(metrics[key], 0.)
        assert opt.goal(None)
        metrics[key] = 0.


def test_additional_correction_participates_in_workflow(correction):
    from dataclasses import dataclass

    base = importlib.import_module('correction_optimizer').CorrectionOptimizer

    @dataclass
    class Settings:
        tolerance: float = 0.1

    class ChromaticityCorrection(base):
        def __init__(self, line):
            action = xt.Action(lambda: pytest.fail('Unexpected action execution'))
            super().__init__(line, {'chromaticity': 0.}, settings=Settings(),
                             name='chromaticity',
                             nominal_knob_values={'sext': 1.},
                             vary=[xt.Vary('sext', step=1e-6)],
                             targets=[action.target('chromaticity', value=0., tol=1e-9)])
            self.set_response([[1.]])

        def configure_goals(self, *, final_refinement, error_fraction):
            self.tolerance = self.settings.tolerance if final_refinement else 1.

        def compute_metrics(self, tw):
            return {'chromaticity': tw['chromaticity']}

        def goal(self, tw):
            return abs(tw['chromaticity']) < self.tolerance

    correction.line['sext'] = 3.
    opt = ChromaticityCorrection(correction.line)
    correction.optimizers = SimpleNamespace(chromaticity=opt)
    correction._collect_controls()
    correction.configure_goals()
    assert correction.correction_knobs == ['sext']
    assert correction.nominal_knob_values == {'sext': 1.}
    before = correction.snapshot()
    correction.scale_corrections(.4)
    assert correction.line['sext'] == 5.  # Scale the deviation from nominal once.
    correction.set_error_fraction(.4)
    correction.final_refinement = True
    assert not correction.goals_met({'chromaticity': .5})
    correction.restore(before)
    assert correction.line['sext'] == 3.
    assert opt.tolerance == 1.
    assert correction.goals_met({'chromaticity': .5})
    del correction.compute_metrics
    assert correction.compute_metrics({'chromaticity': .5}) == {'chromaticity': .5}
    lines = []
    correction.log = lambda *args: lines.append(args)
    correction.log_iteration(0, {'chromaticity': .5})
    assert lines[-1] == ('chromaticity:', {'chromaticity': .5})
    assert len(opt.log()) == 1
    # A knob shared by two corrections is scaled/restored only once.
    correction.optimizers.other = ChromaticityCorrection(correction.line)
    correction._collect_controls()
    assert correction.correction_knobs == ['sext']
    correction.optimizers.other.nominal_knob_values['sext'] = 2.
    with pytest.raises(ValueError, match='Conflicting nominal'):
        correction._collect_controls()


def test_optics_goal_computes_errors_without_aggregate_metrics(correction, monkeypatch):
    reference = SimpleNamespace(
        betx=np.ones(1000), bety=np.ones(1000), dx=np.zeros(1000), qx=0., qy=0.)
    opt = diagnostic_optimizer('optics_corr', 'OpticsCorrection', 'OpticsSettings', reference)
    opt.dispersion_indices = np.array([0, 1])
    opt.ip_indices = np.array([1])
    correction.optimizers = SimpleNamespace(optics=opt)
    correction.configure_goals()
    tw = deepcopy(reference)
    monkeypatch.setattr(correction, 'compute_metrics', lambda *a, **k: pytest.fail('Unexpected aggregate metrics'))
    monkeypatch.setattr(correction, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    assert opt.goal(tw)
    for plane in ('betx', 'bety'):
        beta = getattr(tw, plane)
        beta[:] = 1.04
        assert not opt.goal(tw)
        beta[:] = 1.
        beta[-1] = 1.5
        assert not opt.goal(tw)
        beta[-1] = 1.
    for tune in ('qx', 'qy'):
        setattr(tw, tune, -opt.tune_tol)
        assert not opt.goal(tw)
        setattr(tw, tune, 0.)
    tw.dx[-1] = 1.
    assert opt.goal(tw)
    tw.dx[0] = 0.01
    assert not opt.goal(tw)
    tw.dx[0] = 0.
    tw.dx[1] = -opt.settings.horizontal_dispersion_ip_tol
    assert opt.goal(tw)
    correction.final_refinement = True
    assert not opt.goal(tw)
    tw.dx[1] = np.nextafter(tw.dx[1], 0.)
    assert opt.goal(tw)
    opt.dx_ip_tol *= .5
    assert not opt.goal(tw)  # Active tolerances are directly editable.


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
    settings = module.OpticsSettings()
    weights = np.array([settings.ir_beta_weight, settings.ir_beta_weight,
                        settings.alpha_weight, settings.alpha_weight,
                        settings.horizontal_dispersion_weight
                        * settings.horizontal_dispersion_ip_weight, 1000., 1000.])
    monkeypatch.setattr(module, 'build_normal_quad_optics_response_matrix',
                        lambda *args: physical_response.copy())

    def no_tracking():
        raise AssertionError('Unexpected Twiss')

    monkeypatch.setattr(correction.line, 'twiss', no_tracking)
    opt = module.OpticsCorrection(
        correction.line, reference, knobs=['quad', 'skew'],
        observation_names=['bpm'], interaction_points=['bpm'], ip_names=['bpm'],
        settings=settings)
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
    opt.goal = lambda tw: False
    opt.configure_goals = lambda **kw: None
    correction.optimizers.optics = opt
    trials = []

    def evaluate_trial(tw):
        knob = float(correction.line['quad'])
        trials.append(knob)
        return {'error': knob**2 - 0.25, 'inactive': 1000.}

    monkeypatch.setattr(correction, '_full_twiss', evaluate_trial)
    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw: abs(tw['error']) < 0.2)
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
        return stub_optimizer(run_lstsq=apply, goal=lambda tw: False)

    correction.optimizers = SimpleNamespace(**{
        name: optimizer(name) for name in ('orbit_x', 'orbit_y', 'optics', 'coupling')})
    def stage(name, advance=1):
        def run(tw):
            calls.append((name, tw))
            return tw + advance
        return run

    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw: tw == 6)
    monkeypatch.setattr(correction, '_full_twiss', stage('full'))
    monkeypatch.setattr(correction, 'orbit_twiss', stage('orbit'))
    correction.optimizers.tunes = stub_optimizer(correct=lambda tw, **kw: stage('tune')(tw))
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
    correction.optimizers.tunes.correct = lambda tw, **kw: tune_calls.append(tw) or tw
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
    tune_module = importlib.import_module('tune_corr')
    opt = object.__new__(tune_module.TuneCorrection)
    xd.Optimize.__init__(opt,
        vary=[xd.Vary(name, correction.line.vars, step=1e-6) for name in ('tune', 'quad')],
        targets=[action.target(i, value=value, tol=1e-15)
                 for i, value in enumerate([1., 2.])],
        action_results={action: baseline}, show_call_counter=False)
    opt.set_response(np.eye(2))
    opt.settings = tune_module.TuneSettings()
    opt.configure_goals(final_refinement=False, error_fraction=0.)
    correction.optimizers.tunes = opt
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
            opt.correct(baseline, full_twiss=correction._full_twiss)
        assert len(trials) == 1
        correction.restore(before)
        np.testing.assert_allclose([correction.line[name] for name in ('tune', 'quad')], baseline)
    else:
        result = opt.correct(baseline, full_twiss=correction._full_twiss)
        np.testing.assert_allclose(result, [1., 2.])
        assert len(trials) == 1
        assert opt.correct(result, full_twiss=correction._full_twiss) is result
        assert len(trials) == 1  # Goal check does not run Twiss again.
    assert len(opt.log()) == 1


def test_tunes_allow_non_improving_iterations_until_limit(correction, monkeypatch):
    import xdeps as xd

    correction.line['tune'] = 1.
    action = xt.Action(lambda: pytest.fail('Unexpected optimizer evaluation'))
    tune_module = importlib.import_module('tune_corr')
    opt = object.__new__(tune_module.TuneCorrection)
    xd.Optimize.__init__(opt,
        vary=[xd.Vary('tune', correction.line.vars, step=1e-6)],
        targets=[action.target(0, value=0., tol=1e-12)],
        action_results={action: [2.]}, show_call_counter=False)
    # The actual response is twice this value, so full steps oscillate with
    # unchanged penalty. Only the tune iteration limit should reject them.
    opt.set_response(np.array([[1.]]))
    opt.settings = tune_module.TuneSettings()
    opt.configure_goals(final_refinement=False, error_fraction=0.)
    correction.optimizers.tunes = opt
    trials = []
    def evaluate(tw):
        value = 2 * float(correction.line['tune'])
        trials.append(value)
        return [value]
    monkeypatch.setattr(correction, '_full_twiss', evaluate)
    before = correction.snapshot()
    with pytest.raises(RuntimeError, match='Tune correction iteration limit'):
        opt.correct([2.], full_twiss=correction._full_twiss)
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
        orbit_x=stub_optimizer(run_lstsq=apply, correctable_error=lambda tw: np.array([1.])),
        orbit_y=stub_optimizer(run_lstsq=lambda **kwargs: None,
                               correctable_error=lambda tw: np.array([2.])),
        coupling=stub_optimizer(), tunes=stub_optimizer())
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

    monkeypatch.setattr(correction, 'orbit_twiss', evaluate('orbit'))
    monkeypatch.setattr(correction, '_full_twiss', evaluate('full'))
    def ensure_full(tw):
        if outcome == 'fails':
            assert correction.line['orbit'] == before.knob_values['orbit']
            assert correction.line.twiss_default['co_guess'] == before.co_guess
        return ('full', tw)
    monkeypatch.setattr(correction, 'ensure_full_twiss', ensure_full)
    correction.optimizers.tunes.correct = lambda *a, **kw: pytest.fail('Nested tune correction')
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
    for opt in (correction.optimizers.orbit_x, correction.optimizers.orbit_y):
        opt.correctable_error = lambda tw: np.array([0.])
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
    monitor_indices = np.array([0, 2])
    dispersion_indices = np.array([0, 2, 3])
    ip_indices = np.array([3])
    alignment = lambda: (np.array([.1e-6, -.2e-6]), np.array([-.3e-6, .4e-6]),
                         np.array([.1, -.2]))
    optics = diagnostic_optimizer('optics_corr', 'OpticsCorrection', 'OpticsSettings', reference)
    optics.dispersion_indices, optics.ip_indices = dispersion_indices, ip_indices
    coupling = diagnostic_optimizer('coupling_corr', 'CouplingCorrection', 'CouplingSettings', reference)
    coupling.coupling_indices, coupling.ip_indices = dispersion_indices, ip_indices
    tunes = diagnostic_optimizer('tune_corr', 'TuneCorrection', 'TuneSettings', reference)
    orbit_opts = []
    for plane in 'xy':
        opt = diagnostic_optimizer('orbit_corr', 'OrbitCorrection', 'OrbitSettings', reference)
        opt.plane, opt.monitor_indices = plane, monitor_indices
        opt.monitor_alignment = alignment
        orbit_opts.append(opt)
    correction.optimizers = SimpleNamespace(optics=optics, coupling=coupling, tunes=tunes,
                                           orbit_x=orbit_opts[0], orbit_y=orbit_opts[1])
    del correction.compute_metrics
    monkeypatch.setattr(correction, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    expected = obs.measure(tw, tw_ref=reference, dispersion_indices=dispersion_indices,
        ip_indices=ip_indices, orbit_errors=lambda table: obs.orbit_errors(
            table, monitor_indices=monitor_indices, monitor_alignment=alignment))
    workflow = importlib.import_module('correction_workflow')
    actual = workflow._legacy_metrics(correction.compute_metrics(tw))
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14, atol=0)
    orbit_only = SimpleNamespace(x=tw.x, y=tw.y)
    for i, opt in enumerate(orbit_opts):
        assert opt.compute_metrics(orbit_only)[f'orbit_{opt.plane}_rms'] == expected['orbit_rms'][i]
    printed = []
    correction.log = lambda *args: printed.append(args)
    correction.configure_goals = lambda: None
    for opt in orbit_opts:
        opt.orbit_tol = 1e-5
    correction.log_iteration(1, tw)
    assert len(printed) == 6  # Iteration marker and each optimizer's summary.
