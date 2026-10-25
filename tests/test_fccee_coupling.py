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


def test_coupling_targets_match_response_in_observation_order(coupling_problem, monkeypatch):
    import xdeps as xd

    install, line = coupling_problem
    module = importlib.import_module('corr_coupling')
    knobs = install(line)
    # The response has quantity blocks, while observations deliberately reverse table order.
    matrix = np.random.default_rng(42).normal(size=(10, 2))
    monkeypatch.setattr(module, 'skew_quadrupole_response',
                        lambda *args: (matrix.copy(), np.array([.5, 2.])))

    def evaluate(co_guess=None, **kwargs):
        changes = matrix @ np.array(line.get(knobs))
        tw = xd.Table(dict(
            name=np.array(['sf.0', 'unused', 'sd']), particle_on_co=object(),
            completed_init='periodic', periodic=True,
            f1001=np.array([.1 + .2j, 10j, .3 + .4j]),
            f1010=np.array([.5 + .6j, 20j, .7 + .8j]),
            dy=np.array([.01, 30., .02])), col_names=['name', 'f1001', 'f1010', 'dy'])
        tw.f1001[[2, 0]] += changes[:2] + 1j * changes[2:4]
        tw.f1010[[2, 0]] += changes[4:6] + 1j * changes[6:8]
        tw.dy[[2, 0]] += changes[8:]
        return tw

    reference = evaluate()
    monkeypatch.setattr(line, 'twiss', evaluate)
    opt = module.build_coupling_corr_optimizer(
        line, reference, observation_names=['sd', 'sf.0'], ip_names=['sd'],
        log=lambda *args: None)
    assert opt.action_twiss.line is line
    assert all(target.action is opt.action_twiss for target in opt.targets)
    assert opt.action_twiss.kwargs['method'] == '4d'
    assert opt.action_twiss.kwargs['chrom'] is False
    assert not opt.action_twiss.allow_twiss_failure
    expected = matrix[[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]]
    np.testing.assert_array_equal(opt.response.matrix, expected)
    settings = opt.settings
    np.testing.assert_allclose([target.weight for target in opt.targets],
        np.array([settings.coupling_ip_weight] * 4
                 + [settings.dispersion_weight * settings.coupling_ip_weight
                    * settings.dispersion_rms_tol / settings.dispersion_ip_tol]
                 + [1.] * 4 + [settings.dispersion_weight]) / np.sqrt(2))
    errors = np.array([.002, -.001])
    line.set(knobs, errors)
    tw = evaluate()
    status = opt.target_status(ret=True, action_results=[tw])
    np.testing.assert_allclose(status.residue, expected @ errors)
    np.testing.assert_allclose(opt.get_correction(action_results=[tw]), -errors)
