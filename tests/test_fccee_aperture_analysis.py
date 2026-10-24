"""Launch conventions and loss bookkeeping for the standalone 6D analysis."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xtrack as xt

spec = importlib.util.spec_from_file_location('fccee_aperture',
    Path(__file__).resolve().parents[1] / 'examples/fccee_tuning/006_da_ma_6d.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_ma_actions_and_eight_phase_directions():
    _, grid = analysis.launch_grids()
    shape = (len(analysis.MA_AMPLITUDES), len(analysis.MA_DELTAS), len(analysis.MA_PHASES))
    x, px, y, py = [grid[k].reshape(shape) for k in ('x_norm', 'px_norm', 'y_norm', 'py_norm')]
    expected = np.broadcast_to(analysis.MA_AMPLITUDES[:, None, None]**2, shape)
    np.testing.assert_allclose(x*x+px*px, expected, atol=1e-12)
    np.testing.assert_allclose(y*y+py*py, expected, atol=1e-12)
    np.testing.assert_allclose(x, y)
    np.testing.assert_allclose(px, py)
    directions = np.mod(np.arctan2(px[[0, -1], 0], x[[0, -1], 0]), 2*np.pi)
    np.testing.assert_allclose(np.sort(directions.ravel()), np.arange(8)*np.pi/4)


def test_loss_bookkeeping_after_particle_reordering():
    p = SimpleNamespace(particle_id=np.array([2, 0, 3, 1]),
        state=np.array([1, -1, 1, -1]), at_turn=np.array([300, 7, 300, 0]),
        at_element=np.array([0, 12, 0, 91]))
    result = analysis.ordered_result(p, 300)
    np.testing.assert_array_equal(result['completed_turns'], [7, 0, 300, 300])
    np.testing.assert_array_equal(result['survived'], [False, False, True, True])
    np.testing.assert_array_equal(result['at_element'], [12, 91, 0, 0])


def test_launch_fixes_physical_longitudinal_coordinates_and_includes_dispersion():
    line = xt.Line(elements=[xt.Drift(length=1)])
    line.particle_ref = xt.Particles(p0c=45.6e9, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    co = line.particle_ref.copy()
    co.x = 2e-5
    co.zeta = .001
    co.delta = -.0003
    # A symplectic dispersion transformation, with x += D*pzeta and zeta += D*px.
    W = np.eye(6)
    W[0, 5] = 2.
    W[4, 1] = 2.
    tw = SimpleNamespace(particle_on_co=co, W_matrix=W[None, :, :],
                         zeta=np.array(co.zeta), delta=np.array(co.delta))
    grid = dict(x_norm=np.array([0., 3.]), px_norm=np.array([0., 1.]),
                y_norm=np.array([0., 2.]), py_norm=np.zeros(2),
                delta_offset=np.array([-.01, .02]))
    p = analysis.make_particles(line, tw, grid)
    np.testing.assert_allclose(p.zeta, co.zeta[0], atol=1e-15)
    np.testing.assert_allclose(p.delta, co.delta[0]+grid['delta_offset'], atol=1e-14)
    expected_x = co.x[0]+3*np.sqrt(analysis.GEMITT_X)+2*(p.pzeta[1]-co.pzeta[0])
    np.testing.assert_allclose(p.x[1], expected_x, atol=1e-14)
