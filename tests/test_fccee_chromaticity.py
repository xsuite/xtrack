"""Family offsets, chromaticity targets, and restoration of response probes."""

from pathlib import Path
import importlib

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
        line['k' + family] = strength
    for name in elements:
        family = name[:3]
        line.ref[name].k2 = line.ref['k' + family] * line.ref['on_nonlinear']
    return line


def test_chromaticity_offsets_are_grouped_and_preserve_original_families(
        chromaticity_module, tmp_path):
    line = sextupole_line()
    install = chromaticity_module.install_chromaticity_knobs
    knobs = install(line)
    assert knobs == ['dksf1_chromaticity', 'dksf2_chromaticity',
                     'dksd1_chromaticity', 'dksd2_chromaticity']
    line[knobs[0]] = .02
    assert line['ksf1'] == .2
    assert line['sf1a.0'].k2 == pytest.approx(.07)
    assert line['sf1a.1'].k2 == pytest.approx(.07)
    assert line['sf1fl.0'].k2 == pytest.approx(.05)
    line['on_nonlinear'] = .5
    assert line['sf1a.0'].k2 == pytest.approx(.12)  # Trim is not scaled twice.
    line['ksf1'] = .4
    assert line['sf1a.0'].k2 == pytest.approx(.22)
    line.to_json(tmp_path / 'line.json')
    reloaded = xt.load(tmp_path / 'line.json')
    expression = str(reloaded.ref['sf1a.0'].k2.xdeps.expr)
    assert install(reloaded) == knobs
    assert str(reloaded.ref['sf1a.0'].k2.xdeps.expr) == expression
    assert reloaded[knobs[0]] == .02
    reloaded['on_chromaticity_corr'] = 0.
    assert reloaded['sf1a.0'].k2 == pytest.approx(.2)


def test_chromaticity_installer_rejects_conflicts_before_mutation(chromaticity_module):
    line = sextupole_line()
    line['dksd2_chromaticity'] = .03
    with pytest.raises(ValueError, match='Conflicting chromaticity knob'):
        chromaticity_module.install_chromaticity_knobs(line)
    assert 'on_chromaticity_corr' not in line.vars
    assert 'dksf1_chromaticity' not in line.vars


def test_chromaticity_response_and_ideal_targets(chromaticity_module):
    line = sextupole_line()
    line['on_nonlinear'] = 0.
    knobs = chromaticity_module.install_chromaticity_knobs(line)
    natural, ideal = np.array([-30., -40.]), np.array([2., 3.])
    matrix = np.array([[10., 20., 3., 4.], [-1., -2., -30., -40.]])
    calls = []

    def evaluate(seed=None, *, chrom=True):
        assert chrom
        calls.append(line['on_nonlinear'])
        values = natural + line['on_nonlinear'] * (ideal - natural)
        values += matrix @ np.array(line.get(knobs))
        return xd.Table(dict(name=np.array(['ip']), dqx=values[0], dqy=values[1]),
                        col_names=['name'])

    reference = evaluate()
    opt = chromaticity_module.build_chromaticity_corr_optimizer(
        line, reference, full_twiss=evaluate)
    np.testing.assert_allclose(opt.response.matrix, matrix)
    assert line['on_nonlinear'] == 0.
    np.testing.assert_array_equal(line.get(knobs), np.zeros(4))
    np.testing.assert_array_equal(opt.ideal_chromaticities, ideal)
    assert calls[1:] == [1.] * 9  # Ideal optics and eight response probes.
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


def test_chromaticity_probes_restore_on_failure(chromaticity_module):
    line = sextupole_line()
    knobs = chromaticity_module.install_chromaticity_knobs(line)
    line.set(knobs, [.01, .02, .03, .04])
    before = line.get(knobs)
    def fail(seed):
        raise ValueError('Failed chromatic Twiss')
    with pytest.raises(ValueError, match='Failed chromatic Twiss'):
        chromaticity_module.build_chromaticity_response(
            line, None, knobs=knobs, full_twiss=fail, step=1e-4)
    assert line.get(knobs) == before



def test_chromaticity_builder_restores_nonlinear_strength_on_failure(chromaticity_module):
    line = sextupole_line()
    def fail(seed, *, chrom):
        assert chrom and line['on_nonlinear'] == 1.
        raise ValueError('Failed nominal chromatic Twiss')
    with pytest.raises(ValueError, match='Failed nominal chromatic Twiss'):
        chromaticity_module.build_chromaticity_corr_optimizer(
            line, None, full_twiss=fail)
    assert line['on_nonlinear'] == .25
