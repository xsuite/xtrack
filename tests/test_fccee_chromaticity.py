"""Family offsets, chromaticity targets, and restoration of response probes."""

from pathlib import Path
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import xdeps as xd
import xtrack as xt


@pytest.fixture
def chromaticity_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples' / 'fccee_tuning'))
    return importlib.import_module('corr_chromaticity')


def sextupole_line():
    elements = {f'{family}.0': xt.Sextupole(length=.3)
                for family in ('sf1a', 'sf2a', 'sd1a', 'sd2a', 'sf1fl')}
    elements['sf1a.1'] = xt.Sextupole(length=.3)
    line = xt.Line(elements=elements, element_names=list(elements))
    line['on_nonlinear'] = .25
    for family, strength in zip(('sf1', 'sf2', 'sd1', 'sd2'), (.2, .3, -.4, -.5)):
        line['base_' + family] = strength
        line.ref['k' + family] = line.ref['base_' + family]
    line['ksf1fl'] = line.ref['ksf1'] + .1
    for name in elements:
        family = name.split('.')[0] if name.startswith('sf1fl') else name[:3]
        line.ref[name].k2 = line.ref['k' + family] * line.ref['on_nonlinear']
    return line


def test_chromaticity_offsets_follow_family_expressions_and_reload(
        chromaticity_module, tmp_path):
    line = sextupole_line()
    element_expressions = {name: str(line.ref[name].k2.xdeps.expr)
                           for name in line.element_names}
    install = chromaticity_module.install_chromaticity_knobs
    knobs = install(line)
    assert knobs == ['dksf1_chromaticity', 'dksf2_chromaticity',
                     'dksd1_chromaticity', 'dksd2_chromaticity']
    line[knobs[0]] = .02
    assert line['ksf1'] == pytest.approx(.22)
    assert line['base_sf1'] == .2
    assert line['sf1a.0'].k2 == pytest.approx(.055)
    assert line['sf1a.1'].k2 == pytest.approx(.055)
    assert line['sf1fl.0'].k2 == pytest.approx(.08)  # Derived families follow too.
    line['on_nonlinear'] = .5
    assert line['sf1a.0'].k2 == pytest.approx(.11)
    line['base_sf1'] = .4
    assert line['sf1a.0'].k2 == pytest.approx(.21)
    assert {name: str(line.ref[name].k2.xdeps.expr)
            for name in line.element_names} == element_expressions
    line.to_json(tmp_path / 'line.json')
    reloaded = xt.load(tmp_path / 'line.json')
    expression = str(reloaded.ref['ksf1'].xdeps.expr)
    with pytest.raises(AssertionError, match='knobs already exist'):
        install(reloaded)
    assert str(reloaded.ref['ksf1'].xdeps.expr) == expression
    assert reloaded[knobs[0]] == .02
    assert reloaded['sf1a.0'].k2 == pytest.approx(.21)
    reloaded['on_chromaticity_corr'] = 0.
    assert reloaded['sf1a.0'].k2 == pytest.approx(.2)
    assert reloaded['sf1fl.0'].k2 == pytest.approx(.25)


def test_chromaticity_installer_rejects_existing_knob(chromaticity_module):
    line = sextupole_line()
    line['dksd2_chromaticity'] = 2 * line.ref['ksd2']
    before = str(line.ref['ksd2'].xdeps.expr)
    with pytest.raises(AssertionError, match='knobs already exist'):
        chromaticity_module.install_chromaticity_knobs(line)
    assert str(line.ref['ksd2'].xdeps.expr) == before
    assert line['dksd2_chromaticity'] == pytest.approx(-1.)
    assert 'dksf1_chromaticity' not in line.vars
    assert 'on_chromaticity_corr' not in line.vars


def test_chromaticity_response_and_ideal_targets(chromaticity_module, monkeypatch):
    line = sextupole_line()
    line['on_nonlinear'] = 1.
    knobs = chromaticity_module.install_chromaticity_knobs(line)
    natural, ideal = np.array([-30., -40.]), np.array([2., 3.])
    matrix = np.array([[10., 20., 3., 4.], [-1., -2., -30., -40.]])
    calls = []

    def evaluate(co_guess=None, *, chrom=True, **kwargs):
        assert chrom
        if kwargs:
            assert kwargs['method'] == '4d' and kwargs['delta0'] == 0.
            assert kwargs['delta_chrom'] == 2e-4
            assert kwargs['radiation_integrals'] is False
        calls.append(line['on_nonlinear'])
        values = natural + line['on_nonlinear'] * (ideal - natural)
        values += matrix @ np.array(line.get(knobs))
        return xd.Table(dict(name=np.array(['ip']), particle_on_co=object(),
                             completed_init='periodic', periodic=True, dqx=values[0], dqy=values[1]),
                        col_names=['name'])

    reference = evaluate()
    monkeypatch.setattr(line, 'twiss', evaluate)
    opt = chromaticity_module.build_chromaticity_corr_optimizer(
        line, reference)
    assert opt.action_twiss.line is line
    assert all(target.action is opt.action_twiss for target in opt.targets)
    assert opt.action_twiss.kwargs['method'] == '4d'
    assert opt.action_twiss.kwargs['chrom'] is True
    assert not opt.action_twiss.allow_twiss_failure
    assert opt.action_twiss.kwargs['delta_chrom'] == 2e-4
    np.testing.assert_allclose(opt.response.matrix, matrix)
    assert line['on_nonlinear'] == 1.
    assert opt.reference is reference
    np.testing.assert_array_equal(line.get(knobs), np.zeros(4))
    np.testing.assert_array_equal(opt.ideal_chromaticities, ideal)
    assert calls[1:] == [1.] * 9  # Default-action setup and eight probes.
    line['on_nonlinear'] = 1.
    tw = evaluate()
    assert opt.goal(tw)
    tw['dqx'] += .2
    tw['dqy'] -= .3
    assert not opt.goal(tw)
    count = len(calls)
    status = opt.target_status(ret=True, action_results=[tw])
    np.testing.assert_allclose(status.residue, [.2, -.3])
    step = opt.get_correction(action_results=[tw])
    np.testing.assert_allclose(matrix @ step, [-.2, .3])
    assert len(calls) == count
    missing = xd.Table(dict(name=np.array(['ip'])), col_names=['name'])
    assert not opt.goal(missing)
    assert opt.compute_metrics(missing)['chromaticity_error'] is None
    opt.print_metrics(missing, log=lambda *args: pytest.fail('No chromaticity data'))


def test_chromaticity_probes_restore_on_failure(chromaticity_module, monkeypatch):
    line = sextupole_line()
    knobs = chromaticity_module.install_chromaticity_knobs(line)
    line.set(knobs, [.01, .02, .03, .04])
    before = line.get(knobs)
    def fail(co_guess, **kwargs):
        raise ValueError('Failed chromatic Twiss')
    monkeypatch.setattr(line, 'twiss', fail)
    with pytest.raises(ValueError, match='Failed chromatic Twiss'):
        chromaticity_module.build_chromaticity_response(
            line, SimpleNamespace(particle_on_co=object()), knobs=knobs, step=1e-4)
    assert line.get(knobs) == before



def test_chromaticity_builder_leaves_nonlinear_strength_unchanged_on_failure(chromaticity_module, monkeypatch):
    line = sextupole_line()
    chromaticity_module.install_chromaticity_knobs(line)
    def fail(co_guess, *, chrom, **kwargs):
        assert chrom and line['on_nonlinear'] == .25
        raise ValueError('Failed nominal chromatic Twiss')
    monkeypatch.setattr(line, 'twiss', fail)
    with pytest.raises(ValueError, match='Failed nominal chromatic Twiss'):
        chromaticity_module.build_chromaticity_corr_optimizer(
            line, xd.Table(dict(name=np.array(['ip']), dqx=2., dqy=3.), col_names=['name']))
    assert line['on_nonlinear'] == .25
