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
    line = xt.Line(elements={name: xt.Quadrupole(length=length, order=3)
                            for name, length in [('qf2a.0', 1.), ('qf2a.1', 3.),
                                                 ('qd1a.0', 2.)]})
    line['on_nonlinear'] = .25
    line['existing_octupole'] = .2
    line.ref['qf2a.0'].knl[3] = line.ref['existing_octupole']
    line.metadata = dict(tuning_errors=dict(magnets={
        'qf': dict(category='arc_quad', names=['qf2a.0', 'qf2a.1']),
        'qd': dict(category='arc_quad', names=['qd1a.0'])}),
        correction_knobs=dict(switches={}))
    return line


def test_trims_preserve_expressions_share_strength_and_reload(module, tmp_path):
    line = quad_line()
    names = list(line.element_names)
    knobs = module.install_second_order_chrom_knobs(line)
    assert list(line.element_names) == names
    line.set(knobs, [4., -2.])
    line['existing_octupole'] = .4
    np.testing.assert_allclose([line[n].knl[3] for n in names], [1.4, 3., -2.])
    # Physical trims must not get scaled again by on_nonlinear during the ramp.
    line['on_nonlinear'] = .5
    np.testing.assert_allclose([line[n].knl[3] for n in names], [1.4, 3., -2.])
    line.to_json(tmp_path / 'line.json')
    loaded = xt.load(tmp_path / 'line.json')
    before = str(loaded.ref['qf2a.0'].knl[3].xdeps.expr)
    assert module.install_second_order_chrom_knobs(loaded) == knobs
    assert str(loaded.ref['qf2a.0'].knl[3].xdeps.expr) == before
    loaded['on_arc_octupole_corr'] = 0.
    loaded['existing_octupole'] = .7
    np.testing.assert_allclose([loaded[n].knl[3] for n in names], [.7, 0., 0.])


def test_conflicts_are_rejected_before_mutation(module):
    line = quad_line()
    line['dko_arc_d'] = 1.
    with pytest.raises(ValueError, match='without circuit metadata'):
        module.install_second_order_chrom_knobs(line)
    assert 'on_arc_octupole_corr' not in line.vars
    assert 'dko_arc_f' not in line.vars
    np.testing.assert_allclose(line['qf2a.0'].knl[3], .2)


def test_response_targets_and_metrics_use_consistent_derivative_conventions(module):
    line = quad_line()
    knobs = module.install_second_order_chrom_knobs(line)
    matrix = np.array([[2000., 400.], [-1000., -1100.]])  # d(Q'')/dK3L
    calls = []
    def evaluate(co_guess=None, *, chrom=True):
        assert chrom
        calls.append(line['on_nonlinear'])
        values = np.array([40., -150.])*line['on_nonlinear'] + matrix @ np.array(line.get(knobs))
        return xd.Table(dict(name=np.array(['ip']), particle_on_co=object(), ddqx=values[0], ddqy=values[1]), col_names=['name'])
    opt = module.build_second_order_chrom_optimizer(line, evaluate(), full_twiss=evaluate)
    np.testing.assert_allclose(opt.response.matrix, matrix)
    assert calls[1:] == [1.]*5
    assert line['on_nonlinear'] == .25
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


def test_response_failure_restores_every_probe(module):
    line = quad_line()
    knobs = module.install_second_order_chrom_knobs(line)
    line.set(knobs, [.3, -.4])
    count = 0
    def evaluate(co_guess):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError('failed negative probe')
        return xd.Table(dict(name=np.array(['ip']), particle_on_co=object(), ddqx=1., ddqy=2.), col_names=['name'])
    with pytest.raises(ValueError, match='negative probe'):
        module.build_second_order_chrom_response(line, SimpleNamespace(particle_on_co=object()), knobs=knobs, full_twiss=evaluate, step=.02)
    np.testing.assert_allclose(line.get(knobs), [.3, -.4])


def test_builder_restores_nonlinear_strength_on_failure(module):
    line = quad_line()
    def fail(co_guess, *, chrom):
        assert chrom and line['on_nonlinear'] == 1.
        raise ValueError('failed nominal Twiss')
    with pytest.raises(ValueError, match='nominal Twiss'):
        module.build_second_order_chrom_optimizer(line, SimpleNamespace(particle_on_co=object()), full_twiss=fail)
    assert line['on_nonlinear'] == .25
