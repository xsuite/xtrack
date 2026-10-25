"""Tune trims follow existing family expressions and survive saving and loading."""

from pathlib import Path
import importlib

import numpy as np
import pytest
import xtrack as xt


@pytest.fixture
def tune_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples' / 'fccee_tuning'))
    return importlib.import_module('corr_tune')


@pytest.mark.parametrize('family_expressions', [False, True])
def test_tune_offsets_preserve_expressions_and_reload(tune_module, tmp_path, family_expressions):
    line = xt.Line(elements={name: xt.Quadrupole(length=.4) for name in ('qf', 'qd', 'qf2')},
                   element_names=['qf', 'qd', 'qf2'])
    line['base_qf'], line['base_qd'] = .5, -.4
    line['kqf2'] = line.ref['base_qf'] if family_expressions else .5
    line['kqd1'] = line.ref['base_qd'] if family_expressions else -.4
    line['derived'] = 2 * line.ref['kqf2'] + .1
    line.ref['qf'].k1 = line.ref['kqf2']
    line.ref['qf2'].k1 = line.ref['derived']
    line.ref['qd'].k1 = line.ref['kqd1']
    element_expressions = {name: str(line.ref[name].k1.xdeps.expr) for name in line.element_names}
    line['on_tune_corr'] = 0.
    knobs = tune_module.install_tune_knobs(line)
    assert knobs == ['dkqf2_tune', 'dkqd1_tune']
    assert line['on_tune_corr'] == 1.
    np.testing.assert_array_equal(line.get(knobs), [0., 0.])
    line.set(knobs, [.03, -.02])
    np.testing.assert_allclose(line.get(['kqf2', 'kqd1']), [.53, -.42])
    np.testing.assert_allclose([line[name].k1 for name in ('qf', 'qd', 'qf2')],
                               [.53, -.42, 1.16])
    assert {name: str(line.ref[name].k1.xdeps.expr) for name in line.element_names} == element_expressions
    line.to_json(tmp_path / 'tunes.json')
    reloaded = xt.load(tmp_path / 'tunes.json')
    expressions = {family: str(reloaded.ref[family].xdeps.expr) for family in ('kqf2', 'kqd1')}
    with pytest.raises(AssertionError, match='knobs already exist'):
        tune_module.install_tune_knobs(reloaded)
    np.testing.assert_allclose(reloaded.get(knobs), [.03, -.02])
    # Reference preparation and clean reconstruction zero only the offsets.
    reloaded.set(reloaded.metadata['correction_knobs']['tunes']['nominal_values'])
    np.testing.assert_allclose([reloaded[name].k1 for name in ('qf', 'qd', 'qf2')],
                               [.5, -.4, 1.1])
    reloaded.set(knobs, [.03, -.02])
    if family_expressions:
        reloaded['base_qf'], reloaded['base_qd'] = .6, -.3
        np.testing.assert_allclose([reloaded[name].k1 for name in ('qf', 'qd', 'qf2')],
                                   [.63, -.32, 1.36])
    reloaded['on_tune_corr'] = 0.
    np.testing.assert_allclose(reloaded.get(['kqf2', 'kqd1']),
                               [.6, -.3] if family_expressions else [.5, -.4])
    assert {family: str(reloaded.ref[family].xdeps.expr) for family in ('kqf2', 'kqd1')} == expressions


@pytest.mark.parametrize('existing', ['dkqf2_tune', 'dkqd1_tune'])
def test_tune_installer_rejects_existing_knobs_before_changes(tune_module, existing):
    line = xt.Line(elements=[])
    line['kqf2'], line['kqd1'] = .5, -.4
    line[existing] = .02
    with pytest.raises(AssertionError, match='knobs already exist'):
        tune_module.install_tune_knobs(line)
    assert line.get(['kqf2', 'kqd1']) == [.5, -.4]
    assert line[existing] == .02
    assert 'on_tune_corr' not in line.vars
    assert 'correction_knobs' not in line.metadata
    other = 'dkqd1_tune' if existing == 'dkqf2_tune' else 'dkqf2_tune'
    assert other not in line.vars


def test_tune_builder_uses_installed_family_trims(tune_module):
    import xdeps as xd

    line = xt.Line(elements=[])
    line['kqf2'], line['kqd1'] = .5, -.4
    knobs = tune_module.install_tune_knobs(line)
    orbit = object()
    calls = []

    def evaluate(co_guess=None):
        assert co_guess is None or co_guess is orbit
        calls.append(co_guess)
        return xd.Table(dict(name=np.array(['ip']), particle_on_co=orbit,
                             qx=line['kqf2'], qy=2 * line['kqd1']), col_names=['name'])

    reference = evaluate()
    optimizer = tune_module.build_tune_corr_optimizer(line, reference, full_twiss=evaluate)
    np.testing.assert_allclose(optimizer.response.matrix, [[1., 0.], [0., 2.]], atol=1e-8)
    np.testing.assert_array_equal(line.get(knobs), [0., 0.])
    assert len(calls) == 3
    assert optimizer.goal(reference)
