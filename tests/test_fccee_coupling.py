"""Coupling trims act once per physical sextupole and preserve existing fields."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import xtrack as xt


@pytest.fixture
def coupling_problem(monkeypatch, temp_context_default_func):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples' / 'fccee_tuning'))
    install = importlib.import_module('corr_coupling').install_coupling_knobs
    line = xt.Line(elements={
        'sf.0': xt.Sextupole(length=1.),
        'sf.1': xt.Sextupole(length=3.),
        'sd': xt.Sextupole(length=2.),
        'q': xt.Quadrupole(length=1.)})
    line['family'] = .2
    line['existing_skew'] = .1
    line['on_nonlinear'] = .5
    line['on_coupling_corr'] = 0.
    for name in ('sf.0', 'sf.1', 'sd'):
        line.ref[name].k2 = line.ref['family'] * line.ref['on_nonlinear']
    line.ref['sf.0'].ksl[1] = line.ref['existing_skew']
    line.metadata['tuning_errors'] = dict(magnets={
        'sf': dict(names=['sf.0', 'sf.1']),
        'sd': dict(names=['sd']),
        'q': dict(names=['q'])})
    return install, line


def test_coupling_trims_preserve_fields_and_share_integrated_strength(coupling_problem):
    install, line = coupling_problem
    knobs = install(line)
    assert knobs == ['dk1sl_coupling_sf.0', 'dk1sl_coupling_sd']
    assert line.metadata['correction_knobs']['coupling']['magnet_groups'] == [
        ['sf.0', 'sf.1'], ['sd']]
    assert line['on_coupling_corr'] == 1.
    np.testing.assert_array_equal(line.get(knobs), [0., 0.])
    line.set(knobs, [.4, -.2])
    np.testing.assert_allclose([line[name].ksl[1] for name in ('sf.0', 'sf.1', 'sd')],
                               [.2, .3, -.2])
    line['on_nonlinear'] = 0.
    assert line['sf.0'].k2 == 0.
    assert line['sf.0'].ksl[1] == pytest.approx(.2)
    line['family'] = .4
    line['on_nonlinear'] = .5
    assert line['sf.0'].k2 == pytest.approx(.2)
    line['existing_skew'] = .3
    assert line['sf.0'].ksl[1] == pytest.approx(.4)
    line['on_coupling_corr'] = 0.
    np.testing.assert_allclose([line[name].ksl[1] for name in ('sf.0', 'sf.1', 'sd')],
                               [.3, 0., 0.])


@pytest.mark.parametrize('round_trip', [False, True])
def test_coupling_installer_rejects_existing_knobs(coupling_problem, tmp_path, round_trip):
    install, line = coupling_problem
    if round_trip:
        knobs = install(line)
        line.set(knobs, [.4, -.2])
        line['on_coupling_corr'] = 0.
        line.to_json(tmp_path / 'line.json')
        line = xt.load(tmp_path / 'line.json')
    else:
        # A collision on the last knob must not partially install the first one.
        line['dk1sl_coupling_sd'] = -.2
    before = [line[name].ksl[1] for name in ('sf.0', 'sf.1', 'sd')]
    with pytest.raises(AssertionError, match='knobs already exist'):
        install(line)
    assert line['on_coupling_corr'] == 0.
    assert line['dk1sl_coupling_sd'] == -.2
    np.testing.assert_array_equal(
        [line[name].ksl[1] for name in ('sf.0', 'sf.1', 'sd')], before)
    if not round_trip:
        assert 'dk1sl_coupling_sf.0' not in line.vars
        assert 'correction_knobs' not in line.metadata
