"""Analytical response assembly from actual targets and compound knobs."""
from pathlib import Path
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import xtrack as xt


@pytest.fixture
def optics_problem(monkeypatch, temp_context_default_func):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples' / 'fccee_tuning'))
    module = importlib.import_module('optics_corr')
    env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))
    env['family'] = 0.001
    env['trim'] = 0.0
    env['unused'] = 0.0
    env['intermediate'] = 2 * env.ref['family']
    names = []
    for i in range(8):
        # Resolve the distributed knl perturbation assumed by the analytic model.
        env.new(f'qf{i}', xt.Quadrupole, length=0.4, num_multipole_kicks=32,
                k1='0.5 + intermediate + 0.1*trim', knl=[0., '0.3*trim'],
                knl_rel=[0., '0.2*trim'])
        env.new(f'qd{i}', xt.Quadrupole, length=0.4, num_multipole_kicks=32,
                k1='-0.5 - family', knl=[0., '-0.2*trim'])
        for kind in ('f', 'd'):
            env.new(f'drift{kind}{i}', xt.Drift, length=1.)
            env.new(f'bend{kind}{i}', xt.Bend, length=1., angle=0.02)
            names.extend([f'q{kind}{i}', f'drift{kind}{i}', f'bend{kind}{i}'])
    line = env.new_line(components=names)
    reference = line.twiss(method='4d')

    def no_tracking():
        raise AssertionError('Response builder must not run actions or Twiss')

    action = xt.Action(no_tracking)
    # Deliberately mixed order, distinct alpha/beta locations, and a duplicate.
    keys = [('alfy', 'qd2'), 'qx', ('dx', 'bendf3'), ('betx', 'qf1'),
            ('alfx', 'qf4'), 'qy', ('bety', 'qd6'), ('alfy', 'qd2')]
    targets = [xt.Target(key[0], at=key[1], value=reference[key], action=action)
               if isinstance(key, tuple) else xt.Target(key, value=reference[key], action=action)
               for key in keys]
    opt = line.match(solve=False, targets=targets,
                     vary=[xt.Vary(name, step=1e-6, weight=3.)
                           for name in ('trim', 'family', 'unused')],
                     action_results={action: reference}, show_call_counter=False)
    opt.targets[0].weight = 7.
    opt.targets[0].active = False
    opt.vary[1].active = False
    return module, line, reference, opt, keys


def test_optics_response_matches_tracking_and_restores(optics_problem, monkeypatch):
    module, line, reference, opt, keys = optics_problem
    original_twiss = line.twiss
    monkeypatch.setattr(line, 'twiss', lambda *a, **k: pytest.fail('Unexpected Twiss'))
    baseline = [vary.get_value() for vary in opt.vary]
    original_knl = line.get('qf0').get_total_knl_ksl()[0].copy()
    response = module.build_normal_quad_optics_response_matrix(opt, reference)
    np.testing.assert_array_equal([vary.get_value() for vary in opt.vary], baseline)
    np.testing.assert_array_equal(line.get('qf0').get_total_knl_ksl()[0], original_knl)
    assert not opt.targets[0].active and not opt.vary[1].active
    assert opt.targets[0].weight == 7. and opt.vary[1].weight == 3.
    np.testing.assert_array_equal(response[:, 2], 0.)
    np.testing.assert_array_equal(response[0], response[-1])
    for j, vary in enumerate(opt.vary[:2]):
        try:
            vary.container[vary.name] = baseline[j] + vary.step
            plus = original_twiss(method='4d')
            vary.container[vary.name] = baseline[j] - vary.step
            minus = original_twiss(method='4d')
        finally:
            vary.container[vary.name] = baseline[j]
        numerical = np.array([(plus[key] - minus[key]) / (2 * vary.step) for key in keys])
        np.testing.assert_allclose(response[:, j], numerical, rtol=3e-3, atol=1e-5)


def test_optics_probe_failure_restores_knob(optics_problem, monkeypatch):
    module, line, reference, opt, keys = optics_problem
    baseline = [vary.get_value() for vary in opt.vary]
    original = xt.Quadrupole.get_total_knl_ksl
    calls = 0

    def fail_during_probe(element):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('Failed strength read')
        return original(element)

    monkeypatch.setattr(xt.Quadrupole, 'get_total_knl_ksl', fail_during_probe)
    with pytest.raises(RuntimeError, match='Failed strength read'):
        module.build_normal_quad_optics_response_matrix(opt, reference)
    np.testing.assert_array_equal([vary.get_value() for vary in opt.vary], baseline)
    assert line.ref['qf0'].knl[1].xdeps.expr is not None
    assert line.ref['intermediate'].xdeps.expr is not None


def test_optics_response_rejects_unsupported_targets(optics_problem):
    module, line, reference, opt, keys = optics_problem
    opt.targets[0].tar = ('x', 'qd2')
    with pytest.raises(ValueError, match='Unsupported optics target'):
        module.build_normal_quad_optics_response_matrix(opt, reference)


def test_optics_response_with_only_tunes(optics_problem):
    module, line, reference, opt, keys = optics_problem
    response = module.build_normal_quad_optics_response_matrix(opt, reference)
    tunes = line.match(solve=False, targets=[opt.targets[1], opt.targets[5]],
                       vary=list(opt.vary),
                       action_results={opt.targets[0].action: reference})
    np.testing.assert_allclose(
        module.build_normal_quad_optics_response_matrix(tunes, reference),
        response[[1, 5]], rtol=1e-13)


def test_optics_targets_are_built_per_location(optics_problem):
    module, line, reference, opt, keys = optics_problem
    settings = module.OpticsSettings(ir_half_length=0.1)
    built = module.OpticsCorrection(
        line, reference, knobs=['trim', 'family'],
        observation_names=['qf1', 'qd6'], interaction_points=['qf1'],
        ip_names=['qd6'], settings=settings)
    expected_keys = [
        ('betx', 'qf1'), ('bety', 'qf1'), ('alfx', 'qf1'), ('alfy', 'qf1'), ('dx', 'qf1'),
        ('betx', 'qd6'), ('bety', 'qd6'), ('alfx', 'qd6'), ('alfy', 'qd6'), ('dx', 'qd6'),
        'qx', 'qy',
    ]
    assert [target.tar for target in built.targets] == expected_keys
    expected_weights = np.r_[np.array([5., 5., .1, .1, 30., 1., 1., .1, .1, 1500.])
                             / np.sqrt(2), 1000., 1000.]
    for target, key, weight in zip(built.targets, expected_keys, expected_weights):
        assert target.value == reference[key]
        if isinstance(key, tuple) and key[0] in ('betx', 'bety'):
            assert target.weight == pytest.approx(weight / reference[key])
            assert target.tol == pytest.approx(1e-15 * reference[key], rel=1e-14, abs=0)
        else:
            assert target.weight == pytest.approx(weight)
            assert target.tol == 1e-15
