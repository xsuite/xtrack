import numpy as np
import pytest
import xobjects as xo
from _symplectic import assert_symplectic, fd_jacobian
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']


def _bend_full(**kw):
    return xt.Bend(
        length=1.5,
        angle=0.1,
        k1=0.05,
        knl=[0, 0, 0.2],
        edge_entry_model='full',
        edge_exit_model='full',
        edge_entry_fint=0.5,
        edge_exit_fint=0.5,
        edge_entry_hgap=0.04,
        edge_exit_hgap=0.04,
        edge_entry_angle=0.03,
        edge_exit_angle=-0.02,
        **kw,
    )


def _quad_qsad(**kw):
    return xt.Quadrupole(
        length=0.8,
        k1=1.2,
        k1s=0.3,
        knl=[0, 0, 0.5],
        edge_entry_model='full',
        edge_exit_model='full',
        edge_entry_qint1=0.2,
        edge_entry_qint2=0.1,
        edge_exit_qint1=0.2,
        edge_exit_qint2=-0.05,
        **kw,
    )


ELEMENTS = {'bend': _bend_full, 'quad': _quad_qsad}


def _line(element, context=None):
    line = xt.Line(elements=[element])
    line.set_particle_ref('proton', p0c=1e9)
    line.build_tracker(_context=context)
    return line


def _particles(line, **coords):
    defaults = {'x': 2e-3, 'px': -1e-4, 'y': -1.5e-3, 'py': 2e-4, 'zeta': 1e-3, 'delta': 5e-4}
    defaults.update(coords)
    num = max(np.size(vv) for vv in defaults.values())
    return line.build_particles(
        **{kk: np.broadcast_to(vv, num).copy() for kk, vv in defaults.items()}
    )


def _coords(p):
    p.move(_context=xo.ContextCpu())
    return np.array([getattr(p, cc) for cc in COORDS])


@for_all_test_contexts
@pytest.mark.parametrize('name', ELEMENTS)
def test_new_fringe_maps_round_trip(test_context, name):
    line = _line(ELEMENTS[name](), test_context)
    x = np.linspace(-3e-3, 3e-3, 5)
    p0 = _particles(
        line,
        x=x,
        px=np.linspace(2e-4, -3e-4, 5),
        y=-x,
        py=np.linspace(-1e-4, 2e-4, 5),
        delta=np.linspace(-2e-3, 1e-3, 5),
    )
    p = p0.copy()
    line.track(p)
    assert np.max(np.abs(_coords(p.copy()) - _coords(p0.copy()))) > 1e-5
    line.track(p, backtrack=True)
    assert np.all(p.state == 1)
    np.testing.assert_allclose(_coords(p), _coords(p0), rtol=0, atol=1e-14)


@pytest.mark.parametrize('name', ELEMENTS)
def test_new_fringe_maps_are_symplectic(name):
    line = _line(ELEMENTS[name]())
    centre = _coords(_particles(line))[:, 0]
    assert_symplectic(fd_jacobian(line, centre))


def _track_once(element):
    line = _line(element)
    p = _particles(line)
    line.track(p)
    return _coords(p)


def _quad(model, active=True, **kw):
    return xt.Quadrupole(
        length=0.8,
        k1=1.2,
        edge_entry_model=model,
        edge_exit_model=model,
        edge_entry_active=active,
        edge_exit_active=active,
        **kw,
    )


QINT = {
    'edge_entry_qint1': 0.2, 'edge_entry_qint2': 0.1, 'edge_exit_qint1': 0.2, 'edge_exit_qint2': 0.1
}


def test_quadrupole_soft_fringe_models():
    hard = _track_once(_quad('full'))
    none = _track_once(_quad('full', active=False))
    assert np.max(np.abs(hard - none)) > 1e-10

    # qint = 0: the soft fringe does nothing
    np.testing.assert_array_equal(_track_once(_quad('no-mult')), none)
    np.testing.assert_array_equal(_track_once(_quad('suppressed', **QINT)), none)

    soft = _track_once(_quad('no-mult', **QINT))
    full = _track_once(_quad('full', **QINT))
    assert np.max(np.abs(soft - none)) > 1e-10
    assert np.max(np.abs(full - hard)) > 1e-10
    assert np.max(np.abs(full - soft)) > 1e-10


@pytest.mark.parametrize('fringe', [False, True])
def test_mult_fringe_includes_knl_rel(fringe):
    # knl_rel is relative to k1 * length: both magnets have the same field
    common = {'length': 0.8, 'k1': 1.2, 'edge_entry_active': fringe, 'edge_exit_active': fringe}
    rel = _track_once(
        xt.Quadrupole(knl_rel=[0, 0, 0.5], ksl_rel=[0, 0, -0.2], **common)
    )
    absolute = _track_once(
        xt.Quadrupole(knl=[0, 0, 0.5 * 0.96], ksl=[0, 0, -0.2 * 0.96], **common)
    )
    np.testing.assert_allclose(rel, absolute, rtol=1e-14, atol=1e-16)
