"""State operations can be checked without building the full FCC-ee response."""

from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import xtrack as xt
import xdeps as xd


def stub_twiss(**data):
    return xd.Table(dict(name=np.array(['ip']), particle_on_co=object(), **data),
                    col_names=['name'])


def stub_optimizer(**kwargs):
    defaults = dict(goal=lambda tw: True,
                    settings=SimpleNamespace(),
                    print_metrics=lambda tw, **kw: None,
                    run_lstsq=lambda **kw: None, step_scale=.5)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def correction(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / 'examples' / 'fccee_tuning'
    monkeypatch.syspath_prepend(str(examples))
    module = importlib.import_module('correction_workflow')
    # The full-lattice comparison exercises construction and physics. Here we
    # isolate accepted-state updates and snapshots from the expensive FCC-specific preparation.
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
    state.intermediate_max_iterations = 5
    state.final_max_iterations = 40
    state.delta_chrom = 2e-4
    state.log = lambda *args, **kwargs: None
    state.failure_reason = None
    state.target_error_fraction = None
    state.optimizers = SimpleNamespace(**{name: stub_optimizer() for name in
        ('optics', 'orbit_x', 'orbit_y', 'coupling', 'tunes')})
    state.optimizers.optics.settings = copy(importlib.import_module('corr_optics').DEFAULT_OPTICS_CORR_SETTINGS)
    state.optimizers.coupling.settings = copy(importlib.import_module('corr_coupling').DEFAULT_COUPLING_CORR_SETTINGS)
    state.last_accepted_twiss = SimpleNamespace(
        particle_on_co=object(), x=[1.0], px=[2.0], y=[3.0], py=[4.0], qx=1.2, qy=2.3, delta=[0.0], c_minus=1e-4
    )
    state.line.twiss_default['co_guess'] = dict(x=1.0, px=2.0, y=3.0, py=4.0, zeta=0.0, delta=0.0)
    metrics = dict(
        orbit_x_rms=1e-6, orbit_y_rms=2e-6,
        beta_err_rms=np.array([0.01, 0.02]),
        beta_err_max=np.array([0.1, 0.2]),
        tunes=np.array([1.2, 2.3]),
        tune_error=np.zeros(2),
        dx_err_rms=0.001,
        dx_err_ip=0.00001,
        dy_rms=0.0001,
        dy_ip=0.00001,
        coupling_rms=np.array([0.001, 0.002]),
        coupling_ip=0.001,
        c_minus=0.0001,
    )
    state.compute_metrics = lambda tw, **kwargs: metrics
    yield state
    state.__dict__.clear()


def test_increase_errors_restores_rejected_trial_and_retries(correction, monkeypatch):
    baseline = correction.last_accepted_twiss
    attempts, quad_values = [], []

    def twiss(co_guess):
        assert co_guess is correction.last_accepted_twiss.particle_on_co
        tw = deepcopy(baseline)
        tw.x = [correction.error_fraction]
        return tw

    def correct(tw, *, max_iterations, chrom=False):
        attempts.append(correction.error_fraction)
        quad_values.append(float(correction.line['quad']))
        if len(attempts) == 1:
            correction.line['quad'] = 999.
            correction.last_accepted_twiss = object()
            raise RuntimeError('Unstable candidate')
        return tw, 4 if len(attempts) == 2 else 3

    monkeypatch.setattr(correction, 'twiss', twiss)
    monkeypatch.setattr(correction, 'correct_at_fixed_strengths', correct)
    assert correction.iterative_error_increase_and_correction(
        target_error_fraction=.4, initial_increment=.1,
        min_increment=.0125, max_increment=.1)
    np.testing.assert_allclose(attempts, [.3, .25, .3, .3625, .4])
    np.testing.assert_allclose(quad_values[:2], [4.5, 3.75])
    assert correction.last_accepted_twiss.x == [.4]
    assert not correction.correction_failed and correction.failure_reason is None
    assert correction.target_error_fraction == .4


def test_error_increment_growth_is_capped_and_target_is_exact(correction, monkeypatch):
    attempts = []
    monkeypatch.setattr(correction, 'twiss', lambda co_guess: correction.last_accepted_twiss)

    def correct(tw, *, max_iterations, chrom=False):
        attempts.append(correction.error_fraction)
        return tw, 3

    monkeypatch.setattr(correction, 'correct_at_fixed_strengths', correct)
    assert correction.iterative_error_increase_and_correction(
        target_error_fraction=.4, initial_increment=.05, max_increment=.06)
    np.testing.assert_allclose(attempts, [.25, .31, .37, .4])
    assert correction.error_fraction == .4


def test_run_stops_after_minimum_increment_and_can_save_partial(correction, monkeypatch, tmp_path):
    attempts = []
    monkeypatch.setattr(correction, 'twiss', lambda co_guess: correction.last_accepted_twiss)
    monkeypatch.setattr(correction, 'refine_correction', lambda: pytest.fail('Unexpected refinement'))

    def correct(tw, *, max_iterations, chrom=False):
        attempts.append(correction.error_fraction)
        if len(attempts) > 1:
            correction.line['quad'] = 999.
            raise ValueError('Invalid optics')
        return tw, 4

    monkeypatch.setattr(correction, 'correct_at_fixed_strengths', correct)
    assert not correction.run(target_error_fraction=.5, initial_increment=.1,
                              min_increment=.03, max_increment=.1)
    np.testing.assert_allclose(attempts, [.3, .4, .35])
    assert correction.error_fraction == pytest.approx(.3)
    assert correction.line['quad'] == pytest.approx(4.5)
    assert correction.correction_failed
    assert 'Minimum error increment' in correction.failure_reason
    assert 'Invalid optics' in correction.failure_reason

    # Saving belongs to the caller, including after a failed correction.
    output = tmp_path / 'partial.json'
    correction.line.to_json(output)
    reloaded = xt.load(output)
    for name, value in correction.snapshot().knob_values.items():
        assert reloaded[name] == value
    assert reloaded.twiss_default['co_guess'] == correction.line.twiss_default['co_guess']
    assert not (tmp_path / 'history.csv').exists()


@pytest.mark.parametrize('kwargs', [
    dict(target_error_fraction=0.), dict(target_error_fraction=1.1),
    dict(target_error_fraction=np.nan), dict(target_error_fraction=.1),
    dict(initial_increment=0.), dict(min_increment=-1.),
    dict(min_increment=.02), dict(max_increment=.001), dict(max_increment=np.inf),
])
def test_invalid_execution_settings_do_not_change_state(correction, kwargs):
    before = correction.snapshot()
    with pytest.raises(ValueError):
        correction.run(**kwargs)
    assert correction.snapshot().knob_values == before.knob_values
    assert correction.target_error_fraction is None
    assert not correction.correction_failed and correction.failure_reason is None


@pytest.mark.parametrize('target', [.5, 1.])
def test_run_refines_at_requested_fraction(correction, monkeypatch, target):
    calls = []
    refined = deepcopy(correction.last_accepted_twiss)
    monkeypatch.setattr(correction, 'twiss', lambda co_guess: correction.last_accepted_twiss)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: tw is refined)

    def correct(tw, *, max_iterations, chrom=False):
        assert chrom == (max_iterations == correction.final_max_iterations)
        calls.append((correction.error_fraction, max_iterations))
        assert tw is correction.last_accepted_twiss
        return (refined if max_iterations == correction.final_max_iterations else tw), 1

    monkeypatch.setattr(correction, 'correct_at_fixed_strengths', correct)
    assert correction.run(target_error_fraction=target, initial_increment=.3, max_increment=.3)
    assert correction.error_fraction == target
    assert calls[-1] == (target, correction.final_max_iterations)
    assert all(limit == correction.intermediate_max_iterations for _, limit in calls[:-1])
    assert correction.last_accepted_twiss is refined


def test_refine_failure_restores_accepted_state(correction, monkeypatch):
    before = correction.snapshot()
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: False)
    monkeypatch.setattr(correction, 'twiss', lambda: pytest.fail('Unexpected Twiss'))

    def fail(tw, *, max_iterations, chrom=False):
        assert chrom
        assert max_iterations == correction.final_max_iterations
        assert tw is before.last_accepted_twiss
        correction.line['quad'] = 999.
        correction.last_accepted_twiss = object()
        raise RuntimeError('Iteration budget exhausted')

    monkeypatch.setattr(correction, 'correct_at_fixed_strengths', fail)
    assert not correction.refine_correction()
    assert correction.snapshot().knob_values == before.knob_values
    assert correction.last_accepted_twiss is before.last_accepted_twiss
    assert correction.optimizers.optics.dx_err_ip_tol == correction.optimizers.optics.settings.horizontal_dispersion_ip_tol
    assert correction.correction_failed
    assert correction.failure_reason == 'Refinement failed: Iteration budget exhausted'


def test_refine_already_satisfied_does_no_work(correction, monkeypatch):
    monkeypatch.setattr(correction, 'twiss', lambda: pytest.fail('Unexpected Twiss'))
    monkeypatch.setattr(correction, 'correct_at_fixed_strengths',
                        lambda tw, **kwargs: pytest.fail('Unexpected correction'))
    assert correction.refine_correction()
    assert correction.optimizers.optics.dx_err_ip_tol == correction.optimizers.optics.settings.horizontal_dispersion_ip_tol
    assert not correction.correction_failed
    assert correction.failure_reason is None


def test_scaling_is_separate_from_setting_strength(correction):
    correction.scale_corrections(0.4)
    assert correction.error_fraction == 0.2
    assert correction.line['on_nonlinear'] == 0.2
    assert correction.line.get(correction.correction_knobs) == [4.0, 6.0, 8.0, 14.0]
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
    assert correction.line.get(correction.correction_knobs) == [4.0, 6.0, 8.0, 14.0]


def test_snapshot_restores_all_trial_state_and_preserves_expressions(correction):
    before = correction.snapshot()
    correction.scale_corrections(0.5)
    correction.set_error_fraction(0.5)
    correction.last_accepted_twiss = object()
    correction.restore(before)
    assert correction.snapshot().knob_values == before.knob_values
    assert correction.last_accepted_twiss is before.last_accepted_twiss
    correction.set_error_fraction(0.3)
    assert correction.line['bpm_offset'] == 0.0003


def test_refine_keeps_returned_twiss_without_extra_evaluation_or_history(correction, monkeypatch):
    before = correction.snapshot()
    defaults = deepcopy(correction.line.twiss_default)
    tw = deepcopy(correction.last_accepted_twiss)
    tw.x = [0.123]
    monkeypatch.setattr(correction, 'compute_metrics',
                        lambda tw: pytest.fail('Unexpected metrics evaluation'))
    monkeypatch.setattr(correction, 'twiss', lambda: pytest.fail('Unexpected Twiss'))
    messages = []
    monkeypatch.setattr(correction, 'log', messages.append)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: False)
    monkeypatch.setattr(correction, 'correct_at_fixed_strengths',
                        lambda seed, **kwargs: (tw, 1))
    assert correction.refine_correction()
    assert messages == ['Final refinement', 'Accepted']
    assert correction.last_accepted_twiss is tw
    assert correction.line.twiss_default == defaults
    assert correction.snapshot().knob_values == before.knob_values
    assert not hasattr(correction, 'history')


def test_twiss_evaluation_is_explicit(correction, monkeypatch):
    defaults = deepcopy(correction.line.twiss_default)
    correction.steps_r_matrix = dict(dx=1e-8)
    correction.stats = dict(calls=0, seconds=0., orbit_calls=0, orbit_seconds=0.)
    calls = []
    seed = SimpleNamespace(particle_on_co=object())

    def compute(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(particle_on_co=object())

    monkeypatch.setattr(correction.line, 'twiss', compute)
    first = correction.twiss(co_guess=seed.particle_on_co)
    second = correction.twiss(co_guess=seed.particle_on_co)
    assert second is not first
    assert len(calls) == 2
    assert calls[0]['co_guess'] is seed.particle_on_co
    assert not calls[0]['only_orbit']
    correction.twiss(co_guess=second.particle_on_co, only_orbit=True)
    assert calls[-1]['co_guess'] is second.particle_on_co
    assert calls[-1]['only_orbit']
    correction.twiss()
    assert calls[-1]['co_guess'] == dict(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    assert all(call['method'] == '4d' and call['delta0'] == 0. for call in calls)
    assert correction.line.twiss_default == defaults
    assert correction.stats['calls'] == 4
    assert correction.stats['orbit_calls'] == 1

    # A supplied point satisfying the goals requires no further Twiss.
    tw, iterations = correction.correct_at_fixed_strengths(second, max_iterations=5)
    assert tw is second and iterations == 0
    assert len(calls) == 4


@pytest.mark.parametrize('defaults', [{}, dict(method='6d', delta0=.01,
                                             co_guess=dict(x=.1, y=.2))])
def test_twiss_particle_guess_recomputes_closed_orbit(correction, defaults):
    line = xt.Line(elements=[
        xt.LineSegmentMap(qx=.31, qy=.32, betx=2., bety=3.),
        xt.Multipole(knl=[1e-6]),
    ])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.twiss_default.update(deepcopy(defaults))
    line.build_tracker()
    seed = line.twiss(method='4d', delta0=0., co_guess={})
    seed_x = seed.particle_on_co.x.copy()
    line[1].knl[0] = 2e-6
    stats = dict(calls=0, seconds=0., orbit_calls=0, orbit_seconds=0.)
    expected = line.twiss(method='4d', delta0=0., co_guess={})
    correction.line = line
    correction.stats = stats
    correction.steps_r_matrix = {}
    for only_orbit in (False, True):
        actual = correction.twiss(co_guess=seed.particle_on_co, only_orbit=only_orbit)
        np.testing.assert_allclose(actual.x, expected.x, rtol=1e-12, atol=1e-15)
        assert not np.allclose(actual.x, seed.x, rtol=1e-6, atol=1e-15)
    np.testing.assert_array_equal(seed.particle_on_co.x, seed_x)
    assert stats['calls'] == 2 and stats['orbit_calls'] == 1
    actual = correction.twiss()
    np.testing.assert_allclose(actual.x, expected.x, rtol=1e-12, atol=1e-15)
    assert line.twiss_default == defaults


def diagnostic_optimizer(module_name, settings_name, reference):
    # Test metric/goal methods independently of the expensive response setup.
    module = importlib.import_module(module_name)
    from functools import partial
    base = importlib.import_module('correction_optimizer').CorrectionOptimizer
    opt = object.__new__(base)
    for name in ('compute_metrics', 'goal', 'print_metrics'):
        setattr(opt, name, partial(getattr(module, name), opt))
    opt.settings = copy(getattr(module, settings_name))
    opt.reference = reference
    # Only diagnostic data is needed; response construction is tested separately.
    vars(opt).update(vars(opt.settings))
    if module_name == 'corr_optics':
        opt.dx_err_rms_tol, opt.dx_err_ip_tol = .005, .0005
    elif module_name == 'corr_coupling':
        opt.rdt_rms_tol, opt.rdt_ip_tol, opt.c_minus_tol = .03, .03, .003
        opt.dy_rms_tol, opt.dy_ip_tol = .005, .0001
    return opt


def test_execution_methods_set_limits_and_snapshots_leave_them_unchanged(correction, monkeypatch):
    reference = correction.last_accepted_twiss
    optics = diagnostic_optimizer('corr_optics', 'DEFAULT_OPTICS_CORR_SETTINGS', reference)
    coupling = diagnostic_optimizer('corr_coupling', 'DEFAULT_COUPLING_CORR_SETTINGS', reference)
    orbit = diagnostic_optimizer('corr_orbit', 'DEFAULT_ORBIT_CORR_SETTINGS', reference)
    correction.optimizers = SimpleNamespace(optics=optics, coupling=coupling, orbit=orbit)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: True)
    monkeypatch.setattr(correction, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    orbit.plane = 'x'
    orbit.compute_metrics = lambda tw: {'orbit_x_rms': orbit.orbit_tol}
    assert orbit.goal(None)
    orbit.compute_metrics = lambda tw: {'orbit_x_rms': np.nextafter(orbit.orbit_tol, np.inf)}
    assert not orbit.goal(None)

    limits = [
        (optics, 'dx_err_rms_tol', 'horizontal_dispersion_rms_tol'),
        (optics, 'dx_err_ip_tol', 'horizontal_dispersion_ip_tol'),
        (coupling, 'rdt_rms_tol', 'coupling_rms_tol'),
        (coupling, 'rdt_ip_tol', 'coupling_ip_tol'),
        (coupling, 'c_minus_tol', 'c_minus_tol'),
        (coupling, 'dy_rms_tol', 'dispersion_rms_tol'),
        (coupling, 'dy_ip_tol', 'dispersion_ip_tol'),
    ]
    before = correction.snapshot()
    for prefix in ('', 'intermediate_', ''):
        for opt, attr, _ in limits:
            setattr(opt, attr, -1.)
        if prefix:
            assert correction.iterative_error_increase_and_correction(target_error_fraction=correction.error_fraction)
        else:
            assert correction.refine_correction()
        for opt, attr, setting in limits:
            assert getattr(opt, attr) == getattr(opt.settings, prefix + setting)
        assert orbit.orbit_tol == 1e-5
        correction.set_error_fraction(.8)
        correction.restore(before)
        # Rollback restores the machine, without restoring optimizer configuration.
        for opt, attr, setting in limits:
            assert getattr(opt, attr) == getattr(opt.settings, prefix + setting)
        assert correction.error_fraction == .2
    coupling.dy_ip_tol = 6e-5
    correction.restore(before)
    assert coupling.dy_ip_tol == 6e-5


@pytest.mark.parametrize('key, tolerance', [
    ('coupling_rms', 'rdt_rms_tol'), ('coupling_ip', 'rdt_ip_tol'),
    ('c_minus', 'c_minus_tol'), ('dy_rms', 'dy_rms_tol'), ('dy_ip', 'dy_ip_tol')])
def test_coupling_goal_limits_are_strict(correction, key, tolerance):
    opt = diagnostic_optimizer('corr_coupling', 'DEFAULT_COUPLING_CORR_SETTINGS', correction.last_accepted_twiss)
    metrics = dict(coupling_rms=np.zeros(2), coupling_ip=0., c_minus=0.,
                   dy_rms=0., dy_ip=0.)
    opt.compute_metrics = lambda tw: metrics
    correction.optimizers.coupling = opt
    setting = {'rdt_rms_tol': 'coupling_rms_tol', 'rdt_ip_tol': 'coupling_ip_tol',
               'c_minus_tol': 'c_minus_tol', 'dy_rms_tol': 'dispersion_rms_tol',
               'dy_ip_tol': 'dispersion_ip_tol'}[tolerance]
    for prefix in ('intermediate_', ''):
        limit = getattr(opt.settings, prefix + setting)
        setattr(opt, tolerance, limit)
        metrics[key] = -limit if key == 'c_minus' else limit
        assert not opt.goal(None)
        metrics[key] = np.nextafter(metrics[key], 0.)
        assert opt.goal(None)
        metrics[key] = 0.


def test_additional_correction_participates_in_workflow(correction):
    base = importlib.import_module('correction_optimizer').CorrectionOptimizer

    def compute_metrics(optimizer, tw):
        return {'chromaticity': tw['chromaticity']}

    def goal(optimizer, tw):
        return abs(tw['chromaticity']) < optimizer.tolerance

    def build_chromaticity_optimizer(line):
        action = xt.Action(lambda: pytest.fail('Unexpected action execution'))
        optimizer = base(
            line, {'chromaticity': 0.}, settings=SimpleNamespace(tolerance=1.),
            name='chromaticity', compute_metrics=compute_metrics, goal=goal,
            nominal_knob_values={'sext': 1.},
            vary=[xt.Vary('sext', step=1e-6)],
            targets=[action.target('chromaticity', value=0., tol=1e-9)])
        optimizer.set_response([[1.]])
        optimizer.tolerance = optimizer.settings.tolerance
        return optimizer

    correction.line['sext'] = 3.
    opt = build_chromaticity_optimizer(correction.line)
    correction.optimizers = SimpleNamespace(chromaticity=opt)
    correction._identify_correction_knobs()
    assert correction.correction_knobs == ['sext']
    assert correction.nominal_knob_values == {'sext': 1.}
    before = correction.snapshot()
    correction.scale_corrections(.4)
    assert correction.line['sext'] == 5.  # Scale the deviation from nominal once.
    correction.set_error_fraction(.4)
    assert not correction.goals_met({'chromaticity': 2.})
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
    correction.optimizers.other = build_chromaticity_optimizer(correction.line)
    other = correction.optimizers.other
    assert type(opt) is type(other) is base
    # Callbacks and editable settings belong to the instance that was built.
    other.settings.tolerance = .2
    other.tolerance = other.settings.tolerance
    assert opt.settings.tolerance == 1.
    assert opt.goal({'chromaticity': .5})
    assert not other.goal({'chromaticity': .5})
    correction._identify_correction_knobs()
    assert correction.correction_knobs == ['sext']
    correction.optimizers.other.nominal_knob_values['sext'] = 2.
    with pytest.raises(ValueError, match='Conflicting nominal'):
        correction._identify_correction_knobs()


def test_optics_goal_computes_errors_without_aggregate_metrics(correction, monkeypatch):
    reference = SimpleNamespace(
        betx=np.ones(1000), bety=np.ones(1000), dx=np.zeros(1000),
        alfx=np.zeros(1000), alfy=np.zeros(1000), qx=0., qy=0.)
    opt = diagnostic_optimizer('corr_optics', 'DEFAULT_OPTICS_CORR_SETTINGS', reference)
    opt.dispersion_indices = np.array([0, 1])
    opt.ip_indices = np.array([1])
    correction.optimizers.optics = opt
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
    opt.dx_err_ip_tol = opt.settings.horizontal_dispersion_ip_tol
    assert not opt.goal(tw)
    tw.dx[1] = np.nextafter(tw.dx[1], 0.)
    assert opt.goal(tw)
    opt.dx_err_ip_tol *= .5
    assert not opt.goal(tw)  # Active tolerances are directly editable.


def test_optics_targets_use_relative_beta_and_physical_alpha(correction, monkeypatch):
    import xdeps as xd

    module = importlib.import_module('corr_optics')
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
    settings = copy(module.DEFAULT_OPTICS_CORR_SETTINGS)
    weights = np.array([settings.ir_beta_weight, settings.ir_beta_weight,
                        settings.alpha_ip_weight, settings.alpha_ip_weight,
                        (settings.beta_err_rms_tol / settings.horizontal_dispersion_rms_tol)
                        * (settings.ir_beta_weight * settings.horizontal_dispersion_rms_tol
                           / settings.horizontal_dispersion_ip_tol), 1000., 1000.])
    monkeypatch.setattr(module, 'build_normal_quad_optics_response_matrix',
                        lambda *args: physical_response.copy())

    def no_tracking():
        raise AssertionError('Unexpected Twiss')

    monkeypatch.setattr(correction.line, 'twiss', no_tracking)
    opt = module.build_opt_corr_optimizer(
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
    assert all(vary.weight == 1. for vary in opt.vary)
    expected = np.linalg.lstsq(
        weights[:, None] * physical_response,
        -weights * relative_residuals, rcond=1e-12)[0]
    np.testing.assert_allclose(opt.get_correction(action_results=supplied, rcond=1e-12), expected)
    correction.optimizers = SimpleNamespace(optics=opt)
    correction.optics_weights = target_weights
    assert status.penalty == pytest.approx(
        np.linalg.norm(weights * relative_residuals))


def test_correction_optimizer_tags_outcome_only_when_requested(correction):
    base = importlib.import_module('correction_optimizer').CorrectionOptimizer
    correction.line['quad'] = 1.
    calls = []

    def evaluate():
        value = correction.line['quad']
        calls.append(value)
        return {'error': value}

    action = xt.Action(evaluate)
    opt = base(correction.line, {'error': 1.}, settings=SimpleNamespace(),
               compute_metrics=lambda opt, tw: {}, goal=lambda opt, tw: False,
               vary=[xt.Vary('quad', step=1e-6)],
               targets=[action.target('error', value=0., tol=1e-12)])
    opt.set_response(np.array([[1.]]))
    assert calls == []
    initial_log_length = len(opt.log())
    assert opt.run_lstsq([{'error': 1.}], step_scale=.5) is opt
    assert correction.line['quad'] == pytest.approx(.5)
    assert calls == []
    assert len(opt.log()) == initial_log_length

    assert opt.run_lstsq(action_results=[{'error': .5}], tag_outcome=True) is opt
    assert correction.line['quad'] == pytest.approx(0.)
    assert calls == [pytest.approx(0.)]
    assert len(opt.log()) == initial_log_length + 1


@pytest.mark.parametrize('response, accepted', [(0.1, False), (1., True)])
def test_optics_keeps_worsening_steps_until_goal_or_budget(correction, monkeypatch, response, accepted):
    import xdeps as xd

    correction.line['quad'] = 1.
    def no_tracking():
        raise AssertionError('Optimizer must use supplied results')

    action = xt.Action(no_tracking)
    baseline = stub_twiss(error=0.75, inactive=0.)
    base = importlib.import_module('correction_optimizer').CorrectionOptimizer
    opt = base(correction.line, baseline, settings=SimpleNamespace(),
        compute_metrics=lambda opt, tw: {}, goal=lambda opt, tw: False,
        vary=[xd.Vary('quad', correction.line.vars, step=1e-6)],
        targets=[action.target('error', value=0., tol=1e-12, weight=2.),
                 action.target('inactive', value=0., tol=1e-12)])
    opt.targets[1].active = False
    opt.set_response(np.array([[response], [0.]]))
    opt.goal = lambda tw: False
    correction.optimizers.optics = opt
    trials = []

    def evaluate_trial(co_guess, *, chrom=False):
        knob = float(correction.line['quad'])
        trials.append(knob)
        return stub_twiss(error=knob**2 - 0.25, inactive=1000.)

    monkeypatch.setattr(correction, 'twiss', evaluate_trial)
    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: abs(tw['error']) < 0.2)
    before = correction.snapshot()
    if accepted:
        result, iterations = correction.correct_at_fixed_strengths(baseline, max_iterations=5)
        assert iterations == 1
        np.testing.assert_allclose(trials, [0.25])
        assert correction.line['quad'] == pytest.approx(0.25)
        assert result['error'] == pytest.approx(-0.1875)
    else:
        with pytest.raises(RuntimeError, match='goals not met after 5 iterations'):
            correction.correct_at_fixed_strengths(baseline, max_iterations=5)
        assert len(trials) == 5
        assert trials[0] == pytest.approx(-6.5)
        assert all(abs(b) > abs(a) for a, b in zip(trials, trials[1:]))
        assert correction.line['quad'] == trials[-1]
        # The error-increase/refinement caller owns rollback after a stage fails.
        correction.restore(before)
        assert correction.line['quad'] == 1.
    assert len(opt.log()) == 1


def test_explicit_stage_order_passes_latest_twiss(correction, monkeypatch):
    calls = []
    def optimizer(name):
        def apply(*, action_results, **kwargs):
            calls.append((name, action_results[0].stage))
        return stub_optimizer(run_lstsq=apply, goal=lambda tw: False)

    correction.optimizers = SimpleNamespace(**{
        name: optimizer(name) for name in ('orbit_x', 'orbit_y', 'optics', 'coupling')})
    def stage(name, advance=1):
        def run(co_guess):
            calls.append((name, co_guess))
            return SimpleNamespace(stage=co_guess + advance, particle_on_co=co_guess + advance)
        return run

    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: tw.stage == 6)
    monkeypatch.setattr(correction, 'twiss',
                        lambda co_guess, only_orbit=False, chrom=False: stage('orbit' if only_orbit else 'full')(co_guess))
    correction.optimizers.tunes = optimizer('tune')
    monkeypatch.setattr(correction, 'snapshot', lambda: pytest.fail('Unexpected stage snapshot'))
    result, iterations = correction.correct_at_fixed_strengths(
        SimpleNamespace(stage=0, particle_on_co=0), max_iterations=5)
    assert (result.stage, iterations) == (6, 1)
    assert calls == [('orbit_x', 0), ('orbit_y', 0), ('full', 0),
                     ('optics', 1), ('full', 1),
                     ('orbit_x', 2), ('orbit_y', 2), ('full', 2),
                     ('coupling', 3), ('full', 3),
                     ('orbit_x', 4), ('orbit_y', 4), ('full', 4),
                     ('tune', 5), ('full', 5)]
    assert correction.correct_at_fixed_strengths(result, max_iterations=5) == (result, 0)
    assert len(calls) == 15  # No work at an already accepted point.


@pytest.mark.parametrize('limit', [5, 40])
@pytest.mark.parametrize('converges', [False, True])
def test_cycle_budget_checks_goals_after_last_cycle(correction, monkeypatch,
                                                   limit, converges):
    logged = []
    tune_calls = []
    monkeypatch.setattr(correction, 'log_iteration', lambda iteration, tw: logged.append(iteration))
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: converges and logged[-1] == limit)
    correction.optimizers.tunes.goal = lambda tw: False
    correction.optimizers.tunes.run_lstsq = lambda **kw: tune_calls.append(kw['action_results'][0])
    tw = stub_twiss()
    monkeypatch.setattr(correction, 'twiss', lambda co_guess, **kwargs: tw)
    if converges:
        assert correction.correct_at_fixed_strengths(tw, max_iterations=limit) == (tw, limit)
    else:
        with pytest.raises(RuntimeError, match=f'goals not met after {limit} iterations'):
            correction.correct_at_fixed_strengths(tw, max_iterations=limit)
    assert logged == list(range(limit + 1))
    assert len(tune_calls) == limit


@pytest.mark.parametrize('twiss_fails', [False, True])
def test_tune_step_in_main_cycle(correction, monkeypatch, twiss_fails):
    import xdeps as xd

    action = xt.Action(lambda: pytest.fail('Unexpected optimizer evaluation'))
    baseline = stub_twiss(tunes=np.array([12., 3.]))
    tune_module = importlib.import_module('corr_tune')
    opt = diagnostic_optimizer('corr_tune', 'DEFAULT_TUNE_CORR_SETTINGS', None)
    xd.Optimize.__init__(opt,
        vary=[xd.Vary(name, correction.line.vars, step=1e-6) for name in ('tune', 'quad')],
        targets=[action.target(lambda tw, i=i: tw.tunes[i], value=value, tol=1e-15)
                 for i, value in enumerate([1., 2.])],
        action_results={action: baseline}, show_call_counter=False)
    opt.set_response(np.eye(2))
    opt.settings = copy(tune_module.DEFAULT_TUNE_CORR_SETTINGS)
    for target in opt.targets:
        target.tol = opt.settings.tune_tol
    opt.goal = lambda tw: np.all(np.abs(tw.tunes - [1., 2.]) < opt.settings.tune_tol)
    correction.optimizers.tunes = opt
    trials = []
    def evaluate(co_guess, *, chrom=False):
        values = np.array([float(correction.line[name]) for name in ('tune', 'quad')])
        trials.append(values)
        if twiss_fails:
            raise ValueError('Invalid trial optics')
        return stub_twiss(tunes=values)

    monkeypatch.setattr(correction, 'twiss', evaluate)
    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: np.allclose(tw.tunes, [1., 2.]))
    if twiss_fails:
        before = correction.snapshot()
        with pytest.raises(ValueError, match='Invalid trial optics'):
            correction.correct_at_fixed_strengths(baseline, max_iterations=1)
        assert len(trials) == 1
        correction.restore(before)
        np.testing.assert_allclose([correction.line[name] for name in ('tune', 'quad')], baseline.tunes)
    else:
        result, iterations = correction.correct_at_fixed_strengths(baseline, max_iterations=1)
        assert iterations == 1
        np.testing.assert_allclose(result.tunes, [1., 2.])
        assert len(trials) == 1
        assert correction.correct_at_fixed_strengths(result, max_iterations=1)[0] is result
        assert len(trials) == 1  # Goal check does not run Twiss again.
    assert len(opt.log()) == 1


def test_tunes_allow_non_improving_iterations_until_limit(correction, monkeypatch):
    import xdeps as xd

    correction.line['tune'] = 1.
    action = xt.Action(lambda: pytest.fail('Unexpected optimizer evaluation'))
    tune_module = importlib.import_module('corr_tune')
    opt = diagnostic_optimizer('corr_tune', 'DEFAULT_TUNE_CORR_SETTINGS', None)
    xd.Optimize.__init__(opt,
        vary=[xd.Vary('tune', correction.line.vars, step=1e-6)],
        targets=[action.target('qx', value=0., tol=1e-12)],
        action_results={action: stub_twiss(qx=2.)}, show_call_counter=False)
    # The actual response is twice this value, so full steps oscillate with
    # unchanged penalty. Only the main iteration budget should reject them.
    opt.set_response(np.array([[1.]]))
    opt.settings = copy(tune_module.DEFAULT_TUNE_CORR_SETTINGS)
    for target in opt.targets:
        target.tol = opt.settings.tune_tol
    opt.goal = lambda tw: abs(tw.qx) < opt.settings.tune_tol
    correction.optimizers.tunes = opt
    trials = []
    def evaluate(co_guess, *, chrom=False):
        value = 2 * float(correction.line['tune'])
        trials.append(value)
        return stub_twiss(qx=value)
    monkeypatch.setattr(correction, 'twiss', evaluate)
    before = correction.snapshot()
    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: False)
    with pytest.raises(RuntimeError, match='goals not met after 5 iterations'):
        correction.correct_at_fixed_strengths(stub_twiss(qx=2.), max_iterations=5)
    assert trials == [-2., 2., -2., 2., -2.]
    correction.restore(before)
    assert correction.line['tune'] == 1.


@pytest.mark.parametrize('fail_at', [None, 1, 3])
def test_coupling_orbit_correction_is_unconditional_and_failures_propagate(
        correction, monkeypatch, fail_at):
    before = correction.snapshot()
    orbit_calls = []
    def apply(*, action_results, **kwargs):
        orbit_calls.append(action_results[0].stage)
        correction.line['orbit'] += 1.
    correction.optimizers.orbit_x = stub_optimizer(run_lstsq=apply)
    correction.optimizers.orbit_y = stub_optimizer(run_lstsq=apply)
    correction.optimizers.coupling = stub_optimizer(
        goal=lambda tw: False, run_lstsq=lambda **kwargs: None)
    monkeypatch.setattr(correction, 'log_iteration', lambda *args: None)
    monkeypatch.setattr(correction, 'goals_met', lambda tw, **kwargs: tw.stage == 3)
    monkeypatch.setattr(correction, 'snapshot', lambda: pytest.fail('Unexpected local rollback'))
    def evaluate(co_guess, *, only_orbit=False, chrom=False):
        assert not only_orbit
        if co_guess + 1 == fail_at:
            raise ValueError('Invalid orbit')
        return SimpleNamespace(stage=co_guess + 1, particle_on_co=co_guess + 1)
    monkeypatch.setattr(correction, 'twiss', evaluate)
    initial = SimpleNamespace(stage=0, particle_on_co=0)
    if fail_at is None:
        result, iterations = correction.correct_at_fixed_strengths(initial, max_iterations=1)
        assert (result.stage, iterations) == (3, 1)
        assert orbit_calls == [0, 0, 2, 2]  # Even though ordinary orbit goals are met.
    else:
        with pytest.raises(ValueError, match='Invalid orbit'):
            correction.correct_at_fixed_strengths(initial, max_iterations=1)
        assert correction.line['orbit'] > before.knob_values['orbit']
        correction.restore(before)  # The outer caller owns rollback.
        assert correction.line['orbit'] == before.knob_values['orbit']


def test_module_metrics_match_expected_errors(correction, monkeypatch):
    import xdeps as xd

    orbit = importlib.import_module('corr_orbit')
    columns = ['name', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'f1001', 'f1010', 'x', 'y']
    reference = xd.Table(dict(
        name=np.array(['bpm1', 'other', 'bpm2', 'ip']),
        betx=np.array([10., 20., 30., 40.]), bety=np.array([15., 25., 35., 45.]),
        alfx=np.zeros(4), alfy=np.zeros(4),
        dx=np.array([.1, .2, .3, .4]), dy=np.array([.01, .02, .03, .04]),
        f1001=np.array([.01j, .02, -.03j, .04]),
        f1010=np.array([-.01, .02j, .03, -.04j]),
        x=np.zeros(4), y=np.zeros(4), qx=.31, qy=.32, c_minus=0.), col_names=columns)
    tw = xd.Table({key: deepcopy(reference[key])
                   for key in columns + ['qx', 'qy', 'c_minus']}, col_names=columns)
    tw.betx[:] *= [1.01, 1.2, .98, 1.03]
    tw.bety[:] *= [.99, .8, 1.02, .97]
    tw.alfx[:] = [0., 1., 0., .0003]
    tw.alfy[:] = [0., 2., 0., -.0002]
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
    optics = diagnostic_optimizer('corr_optics', 'DEFAULT_OPTICS_CORR_SETTINGS', reference)
    optics.dispersion_indices, optics.ip_indices = dispersion_indices, ip_indices
    coupling = diagnostic_optimizer('corr_coupling', 'DEFAULT_COUPLING_CORR_SETTINGS', reference)
    coupling.coupling_indices, coupling.ip_indices = dispersion_indices, ip_indices
    tunes = diagnostic_optimizer('corr_tune', 'DEFAULT_TUNE_CORR_SETTINGS', reference)
    monkeypatch.setattr(orbit, 'monitor_alignment', lambda optimizer: alignment())
    orbit_opts = []
    for plane in 'xy':
        opt = diagnostic_optimizer('corr_orbit', 'DEFAULT_ORBIT_CORR_SETTINGS', reference)
        opt.plane, opt.monitor_indices = plane, monitor_indices
        orbit_opts.append(opt)
    correction.optimizers = SimpleNamespace(optics=optics, coupling=coupling, tunes=tunes,
                                           orbit_x=orbit_opts[0], orbit_y=orbit_opts[1])
    del correction.compute_metrics
    monkeypatch.setattr(correction, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    # Known errors at all elements (beta) or only the selected BPMs/IP (dispersion
    # and coupling); large errors at the unobserved element must be excluded.
    x, y = np.array([.9e-6, -1.8e-6]), np.array([-2.7e-6, 3.6e-6])
    roll = np.array([.1, -.2])
    expected = dict(
        orbit_x_rms=np.sqrt(np.mean((x * np.cos(roll) + y * np.sin(roll))**2)),
        orbit_y_rms=np.sqrt(np.mean((-x * np.sin(roll) + y * np.cos(roll))**2)),
        beta_err_rms=np.full(2, np.sqrt((.01**2 + .2**2 + .02**2 + .03**2) / 4)),
        beta_err_max=np.array([.2, .2]),
        alpha_err_ip=np.array([.0003, .0002]),
        tunes=np.array([.310001, .319998]),
        tune_error=np.array([.310001 - .31, .319998 - .32]),
        dx_err_rms=np.sqrt((.001**2 + .002**2 + .00004**2) / 3),
        dx_err_ip=.00004,
        dy_rms=np.sqrt((.0001**2 + .0002**2 + .00001**2) / 3),
        dy_ip=.00001,
        coupling_rms=np.sqrt(np.array([
            .001**2 + .002**2 + .002**2 + .0004**2,
            .002**2 + .001**2 + .0002**2]) / 3),
        coupling_ip=.0004,
        c_minus=-.0002,
    )
    actual = correction.compute_metrics(tw)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=0)
    orbit_only = SimpleNamespace(x=tw.x, y=tw.y)
    for opt in orbit_opts:
        key = f'orbit_{opt.plane}_rms'
        assert opt.compute_metrics(orbit_only)[key] == expected[key]
    printed = []
    correction.log = lambda *args: printed.append(args)
    for opt in orbit_opts:
        opt.orbit_tol = 1e-5
    correction.log_iteration(1, tw)
    assert len(printed) == 6  # Iteration marker and each optimizer's summary.


def test_chromaticity_is_skipped_during_error_increase(correction, monkeypatch):
    correction.optimizers.chromaticity = stub_optimizer(
        goal=lambda tw: pytest.fail('Chromaticity goal during error increase'),
        run_lstsq=lambda **kwargs: pytest.fail('Chromaticity step during error increase'))
    correction.optimizers.second_order_chrom = stub_optimizer(
        goal=lambda tw: pytest.fail('Second-order goal during error increase'),
        run_lstsq=lambda **kwargs: pytest.fail('Second-order step during error increase'))
    correction.optimizers.tunes.goal = lambda tw: tw['qx'] == .3
    calls = []
    def evaluate(co_guess, *, chrom):
        assert not chrom
        calls.append(co_guess)
        return stub_twiss(qx=.3)
    monkeypatch.setattr(correction, 'twiss', evaluate)
    initial = stub_twiss(qx=.4)
    tw, iterations = correction.correct_at_fixed_strengths(initial, max_iterations=5)
    assert tw.qx == .3 and iterations == 1
    assert calls == [initial.particle_on_co]


def test_chromaticity_stage_rechecks_other_goals(correction, monkeypatch):
    # The sextupole change spoils the tune; success requires another cycle.
    state = dict(qx=0., dqx=2., dqy=3.)
    calls = []
    def evaluate(co_guess, *, chrom=False):
        assert chrom
        calls.append('twiss')
        return stub_twiss(**state)
    def chrom_step(*, action_results):
        assert all(action_results[0][key] == value for key, value in state.items())
        calls.append('chromaticity')
        state.update(qx=1., dqx=0.)
    def tune_step(*, action_results):
        assert all(action_results[0][key] == value for key, value in state.items())
        calls.append('tunes')
        state['qx'] = 0.
    correction.optimizers.chromaticity = stub_optimizer(
        goal=lambda tw: tw._data.get('dqx') == 0., run_lstsq=chrom_step)
    correction.optimizers.tunes = stub_optimizer(
        goal=lambda tw: tw['qx'] == 0., run_lstsq=tune_step)
    monkeypatch.setattr(correction, 'twiss', evaluate)
    tw, iterations = correction.correct_at_fixed_strengths(
        stub_twiss(qx=0.), max_iterations=5, chrom=True)
    assert iterations == 2
    assert calls == ['twiss', 'chromaticity', 'twiss', 'tunes', 'twiss']
    assert correction.goals_met(tw)


def test_second_order_stage_rechecks_first_order_and_tunes(correction, monkeypatch):
    state = dict(qx=0., dqx=0., dqy=0., ddqx=4., ddqy=0.)
    calls = []
    def evaluate(co_guess, *, chrom=False):
        assert chrom
        calls.append('twiss')
        return stub_twiss(**state)
    def second_order_step(*, action_results):
        assert all(action_results[0][key] == value for key, value in state.items())
        calls.append('second_order')
        state.update(qx=1., dqx=2., ddqx=0.)
    def first_order_step(*, action_results):
        calls.append('first_order')
        state['dqx'] = 0.
    def tune_step(*, action_results):
        calls.append('tunes')
        state['qx'] = 0.
    correction.optimizers.chromaticity = stub_optimizer(
        goal=lambda tw: tw._data.get('dqx') == 0., run_lstsq=first_order_step)
    correction.optimizers.second_order_chrom = stub_optimizer(
        goal=lambda tw: tw._data.get('ddqx') == 0., run_lstsq=second_order_step)
    correction.optimizers.tunes = stub_optimizer(
        goal=lambda tw: tw['qx'] == 0., run_lstsq=tune_step)
    monkeypatch.setattr(correction, 'twiss', evaluate)
    tw, iterations = correction.correct_at_fixed_strengths(stub_twiss(**state), max_iterations=5, chrom=True)
    assert iterations == 2
    assert calls == ['second_order', 'twiss', 'tunes', 'twiss', 'first_order', 'twiss']
    assert correction.goals_met(tw)


def test_second_order_knobs_are_scaled_once_and_rolled_back(correction, monkeypatch):
    correction.line['dko_arc_f'] = .2
    correction.optimizers.second_order_chrom = SimpleNamespace(
        vary=[SimpleNamespace(name='dko_arc_f')], nominal_knob_values={'dko_arc_f': 0.})
    # Isolate the new optimizer, while exercising the normal control discovery.
    correction.optimizers = SimpleNamespace(second_order_chrom=correction.optimizers.second_order_chrom)
    correction._identify_correction_knobs()
    before = correction.snapshot()
    correction.scale_corrections(.4)
    assert correction.line['dko_arc_f'] == pytest.approx(.4)
    correction.restore(before)
    assert correction.line['dko_arc_f'] == .2


def test_chromaticity_family_trims_scale_only_through_nonlinear_strength(correction):
    module = importlib.import_module('corr_chromaticity')
    # Installation needs only the family variables, even with no sextupole elements.
    for family in module.SEXTUPOLE_FAMILY_KNOBS:
        correction.line[family] = .2
    knobs = module.install_chromaticity_knobs(correction.line)
    correction.line[knobs[0]] = .4
    correction.line['chromaticity_strength'] = (
        correction.line.ref['on_nonlinear'] * correction.line.ref[knobs[0]])
    correction.optimizers = SimpleNamespace(
        chromaticity=SimpleNamespace(
            vary=[SimpleNamespace(name=name) for name in knobs],
            nominal_knob_values=dict.fromkeys(knobs, 0.)),
        tunes=SimpleNamespace(vary=[SimpleNamespace(name='tune')],
                              nominal_knob_values={'tune': 10.}))
    correction._identify_correction_knobs()
    before = correction.snapshot()
    assert correction.line['chromaticity_strength'] == pytest.approx(.08)
    correction.scale_corrections(.4)
    assert correction.line[knobs[0]] == .4
    assert correction.line['tune'] == 14.
    correction.set_error_fraction(.4)
    assert correction.line['chromaticity_strength'] == pytest.approx(.16)
    correction.line[knobs[0]] = .8
    correction.restore(before)
    assert correction.line[knobs[0]] == .4
    assert correction.line['chromaticity_strength'] == pytest.approx(.08)
