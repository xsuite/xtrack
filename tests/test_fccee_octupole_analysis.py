"""Check action normalization and independent detuning fits in the arc study."""
import importlib.util
from pathlib import Path

import numpy as np
import xtrack as xt

spec = importlib.util.spec_from_file_location('arc_octupoles',
    Path(__file__).resolve().parents[1] / 'examples/fccee_tuning/007_arc_octupoles.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_separate_action_fits_do_not_alias_horizontal_curvature_into_vertical_slope():
    actions = analysis.AMPLITUDES**2 * analysis.EMITTANCES / 2
    detuning = np.array([[1.7e5, 2.2e5], [2.2e5, -3.3e5]])
    tunes = np.array([.157, .2]) + actions @ detuning.T
    # The emittances differ by a factor ~500. A joint linear fit to both
    # actions would misinterpret this horizontal curvature as vertical slope.
    tunes += actions[:, :1]**2 * np.array([2e12, -3e12])
    for maximum in (1., 2., 3.):
        slopes = analysis.detuning_slopes(tunes, maximum)
        np.testing.assert_allclose(slopes[:, 1], detuning[:, 1], rtol=1e-7)
    # With no curvature, recover all four coefficients and the geometric units.
    linear = np.array([.157, .2]) + actions @ detuning.T
    np.testing.assert_allclose(analysis.detuning_slopes(linear, 3.), detuning, rtol=1e-7)


def test_quad_trims_preserve_expressions_and_share_integrated_strength(tmp_path):
    line = xt.Line(elements={name: xt.Quadrupole(length=length, order=3)
                            for name, length in [('qf2a.0', 1.), ('qf2a.1', 3.),
                                                 ('qd1a.0', 2.)]})
    line['on_nonlinear'] = 1.
    line['existing_octupole'] = .2
    line.ref['qf2a.0'].knl[3] = line.ref['existing_octupole']
    line.metadata = dict(tuning_errors=dict(magnets={
        'qf': dict(category='arc_quad', names=['qf2a.0', 'qf2a.1']),
        'qd': dict(category='arc_quad', names=['qd1a.0'])}),
        correction_knobs=dict(switches={}))
    names_before = list(line.element_names)
    analysis.install_octupole_trims(line)
    assert list(line.element_names) == names_before
    line['dko_arc_f'] = 4.
    line['dko_arc_d'] = -2.
    line['existing_octupole'] = .4
    np.testing.assert_allclose([line[n].knl[3] for n in names_before], [1.4, 3., -2.])
    line.to_json(tmp_path / 'line.json')
    loaded = xt.load(tmp_path / 'line.json')
    loaded['on_arc_octupole_corr'] = 0.
    loaded['existing_octupole'] = .7
    np.testing.assert_allclose([loaded[n].knl[3] for n in names_before], [.7, 0., 0.])
    loaded['on_arc_octupole_corr'] = 1.
    loaded['on_nonlinear'] = 0.
    np.testing.assert_allclose([loaded[n].knl[3] for n in names_before], [.7, 0., 0.])
