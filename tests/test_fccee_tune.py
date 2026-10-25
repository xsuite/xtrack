"""Independent tune offsets preserve nominal family variables."""

from pathlib import Path
import importlib

import numpy as np
import pytest
import xtrack as xt


def test_tune_offsets_preserve_base_knobs_and_reuse_after_reload(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples' / 'fccee_tuning'))
    module = importlib.import_module('corr_tune')
    line = xt.Line(elements={name: xt.Quadrupole(length=.4) for name in ('qf', 'qd', 'qf2')},
                   element_names=['qf', 'qd', 'qf2'])
    line['kqf2'], line['kqd1'] = .5, -.4
    line.ref['qf'].k1 = line.ref['kqf2']
    line.ref['qf2'].k1 = 2 * line.ref['kqf2'] + .1
    line.ref['qd'].k1 = line.ref['kqd1']
    line.metadata['correction_knobs'] = dict(tunes=dict(
        knobs=['kqf2', 'kqd1'], nominal_values={'kqf2': .5, 'kqd1': -.4}))
    line['dkqf2_tune'] = 2 * line.ref['kqf2']
    knobs = module.install_tune_knobs(line)
    assert knobs == ['dkqf2_tune', 'dkqd1_tune']
    np.testing.assert_array_equal(line.get(knobs), [0., 0.])
    line.set(knobs, [.03, -.02])
    assert line.get(['kqf2', 'kqd1']) == [.5, -.4]
    np.testing.assert_allclose([line[name].k1 for name in ('qf', 'qd', 'qf2')],
                               [.53, -.42, 1.16])
    line.to_json(tmp_path / 'tunes.json')
    reloaded = xt.load(tmp_path / 'tunes.json')
    expression = str(reloaded.ref['qf'].k1.xdeps.expr)
    reloaded[knobs[1]] = 3 * reloaded.ref['kqd1']
    assert module.install_tune_knobs(reloaded) == knobs
    np.testing.assert_array_equal(reloaded.get(knobs), [0., 0.])
    assert str(reloaded.ref['qf'].k1.xdeps.expr) == expression
    reloaded['kqf2'] = .6
    assert reloaded['qf'].k1 == pytest.approx(.6)
    assert reloaded['qf2'].k1 == pytest.approx(1.3)
    reloaded.set(knobs, [.03, -.02])
    assert reloaded['qf'].k1 == pytest.approx(.63)
    assert reloaded['qf2'].k1 == pytest.approx(1.36)
    reloaded['kqd1'] = -.3
    assert reloaded[knobs[1]] == -.02
    assert reloaded['qd'].k1 == pytest.approx(-.32)
