"""Arc quad offsets, Q'' conventions, and restoration of response probes."""
import importlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import xdeps as xd
import xtrack as xt


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples/fccee_tuning'))
    return importlib.import_module('corr_second_order_chrom')


def quad_line():
    elements = {name: xt.Quadrupole(length=length, order=3)
                for name, length in [('qf2a.0', 1.), ('qf2a.1', 3.), ('qf3a.0', 2.),
                                     ('qd1a.0', 2.), ('qd1am.0', 1.5), ('qf2a.99', 4.)]}
    elements['bend'] = xt.Bend(length=.5, order=3)
    line = xt.Line(elements=elements)
    line['kqf2'], line['kqd1'] = .5, -.4
    line['derived_qf'] = 2 * line.ref['kqf2']
    line['optics_trim'] = .1
    line.ref['qf2a.0'].k1 = line.ref['kqf2'] + line.ref['optics_trim']
    line.ref['qf2a.1'].k1 = line.ref['kqf2']
    line.ref['qf3a.0'].k1 = line.ref['derived_qf']
    line.ref['qd1a.0'].k1 = line.ref['kqd1']
    line.ref['qd1am.0'].k1 = 3 * line.ref['kqd1']
    # A similarly named but independent quadrupole and a controlled bend are excluded.
    line.ref['bend'].k1 = line.ref['kqf2']
    line['on_nonlinear'] = .25
    line['existing_octupole'] = .2
    line.ref['qf2a.0'].knl[3] = line.ref['existing_octupole']
    return line


def test_k3_trims_follow_controlled_quadrupoles_and_reload(module, tmp_path):
    line = quad_line()
    names = list(line.element_names)
    line['on_arc_octupole_corr'] = 0.
    knobs = module.install_second_order_chrom_knobs(line)
    assert list(line.element_names) == names
    assert line['on_arc_octupole_corr'] == 1.
    np.testing.assert_array_equal(line.get(knobs), [0., 0.])
    circuit = line.metadata['correction_knobs']['second_order_chrom']
    assert circuit['units'] == 'm^-4'
    assert circuit['family_knobs'] == ['kqf2', 'kqd1']
    assert [set(group) for group in circuit['quadrupole_groups']] == [
        {'qf2a.0', 'qf2a.1', 'qf3a.0'}, {'qd1a.0', 'qd1am.0'}]
    line.set(knobs, [4., -2.])
    line['existing_octupole'] = .4
    np.testing.assert_allclose([line[n].knl[3] for n in names], [4.4, 12., 8., -4., -3., 0., 0.])
    # Physical trims must not get scaled again by on_nonlinear during the ramp.
    line['on_nonlinear'] = .5
    line['optics_trim'] = .3
    line['kqf2'] = line['kqd1'] = 0.  # No division by the family or dependence on k1.
    np.testing.assert_allclose([line[n].knl[3] for n in names], [4.4, 12., 8., -4., -3., 0., 0.])
    line.to_json(tmp_path / 'line.json')
    loaded = xt.load(tmp_path / 'line.json')
    before = str(loaded.ref['qf2a.0'].knl[3].xdeps.expr)
    with pytest.raises(AssertionError, match='knobs already exist'):
        module.install_second_order_chrom_knobs(loaded)
    assert str(loaded.ref['qf2a.0'].knl[3].xdeps.expr) == before
    np.testing.assert_array_equal(loaded.get(knobs), [4., -2.])
    np.testing.assert_allclose([loaded[n].knl[3] for n in names], [4.4, 12., 8., -4., -3., 0., 0.])
    loaded['on_arc_octupole_corr'] = 0.
    loaded['existing_octupole'] = .7
    np.testing.assert_allclose([loaded[n].knl[3] for n in names], [.7, 0., 0., 0., 0., 0., 0.])


def test_installer_rejects_existing_knobs_before_changes(module):
    line = quad_line()
    line['on_arc_octupole_corr'] = .5
    line['dko_arc_d'] = 2 * line.ref['existing_octupole']
    with pytest.raises(AssertionError, match='knobs already exist'):
        module.install_second_order_chrom_knobs(line)
    assert line['on_arc_octupole_corr'] == .5
    assert 'dko_arc_f' not in line.vars
    assert 'correction_knobs' not in line.metadata
    line['existing_octupole'] = .4
    assert line['dko_arc_d'] == pytest.approx(.8)
    assert line['qf2a.0'].knl[3] == pytest.approx(.4)
    assert line['qd1a.0'].knl[3] == 0.


def test_response_targets_and_metrics_use_consistent_derivative_conventions(module, monkeypatch):
    line = quad_line()
    line['on_nonlinear'] = 1.
    knobs = module.install_second_order_chrom_knobs(line)
    matrix = np.array([[2000., 400.], [-1000., -1100.]])  # d(Q'')/dk3
    calls = []
    def evaluate(co_guess=None, *, chrom=True, **kwargs):
        assert chrom
        if kwargs:
            assert kwargs['method'] == '4d' and kwargs['delta0'] == 0.
            assert kwargs['delta_chrom'] == 2e-4
            assert kwargs['radiation_integrals'] is False
        calls.append(line['on_nonlinear'])
        values = np.array([40., -150.])*line['on_nonlinear'] + matrix @ np.array(line.get(knobs))
        return xd.Table(dict(name=np.array(['ip']), particle_on_co=object(),
                             completed_init='periodic', periodic=True, ddqx=values[0], ddqy=values[1]), col_names=['name'])
    monkeypatch.setattr(line, 'twiss', evaluate)
    reference = evaluate()
    opt = module.build_second_order_chrom_optimizer(line, reference)
    assert opt.reference is reference
    assert opt.action_twiss.line is line
    assert all(target.action is opt.action_twiss for target in opt.targets)
    assert opt.action_twiss.kwargs['method'] == '4d'
    assert opt.action_twiss.kwargs['chrom'] is True
    assert not opt.action_twiss.allow_twiss_failure
    assert opt.action_twiss.kwargs['delta_chrom'] == 2e-4
    np.testing.assert_allclose(opt.response.matrix, matrix)
    assert calls[1:] == [1.]*5  # Includes default-action setup.
    assert line['on_nonlinear'] == 1.
    np.testing.assert_allclose(line.get(knobs), 0.)
    np.testing.assert_allclose(opt.ideal_second_order_chrom, [20., -75.])
    line['on_nonlinear'] = 1.
    tw = evaluate()
    assert opt.goal(tw)
    tw['ddqx'] += 4.
    tw['ddqy'] -= 6.
    assert not opt.goal(tw)
    np.testing.assert_allclose(opt.compute_metrics(tw)['second_order_chrom_error'], [2., -3.])
    count = len(calls)
    np.testing.assert_allclose(matrix @ opt.get_correction(action_results=[tw]), [-4., 6.])
    assert len(calls) == count
    np.testing.assert_allclose([t.tol for t in opt.targets], 2.)
    missing = xd.Table(dict(name=np.array(['ip'])), col_names=['name'])
    assert not opt.goal(missing)
    opt.print_metrics(missing, log=lambda *a: pytest.fail('Missing data should not print'))


def test_response_failure_restores_every_probe(module, monkeypatch):
    line = quad_line()
    knobs = module.install_second_order_chrom_knobs(line)
    line.set(knobs, [.3, -.4])
    count = 0
    def evaluate(co_guess, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError('failed negative probe')
        return xd.Table(dict(name=np.array(['ip']), particle_on_co=object(), ddqx=1., ddqy=2.), col_names=['name'])
    monkeypatch.setattr(line, 'twiss', evaluate)
    with pytest.raises(ValueError, match='negative probe'):
        module.build_second_order_chrom_response(line, SimpleNamespace(particle_on_co=object()), knobs=knobs, step=module.DEFAULT_SECOND_ORDER_CHROM_SETTINGS.response_step)
    np.testing.assert_allclose(line.get(knobs), [.3, -.4])


def test_builder_leaves_nonlinear_strength_unchanged_on_failure(module, monkeypatch):
    line = quad_line()
    module.install_second_order_chrom_knobs(line)
    def fail(co_guess, *, chrom, **kwargs):
        assert chrom and line['on_nonlinear'] == .25
        raise ValueError('failed nominal Twiss')
    monkeypatch.setattr(line, 'twiss', fail)
    with pytest.raises(ValueError, match='nominal Twiss'):
        module.build_second_order_chrom_optimizer(line, xd.Table(dict(name=np.array(['ip']), ddqx=40., ddqy=-150.), col_names=['name']))
    assert line['on_nonlinear'] == .25
