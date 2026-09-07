import warnings

import numpy as np
import pytest
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack import Magnet


def make_particles(context):
    return xt.Particles(
        kinetic_energy0=50e6,
        x=[1e-3, -1e-3],
        y=2e-3,
        zeta=1e-2,
        px=10e-3,
        py=20e-3,
        delta=1e-2,
        _context=context,
    )


def test_yoshida_integrator_names_and_warning():
    expected_indices = {
        'adaptive': 0,
        'yoshida-6': 2,  # preserve the historical sixth-order behaviour
        'yoshida4': 2,  # preserve the historical sixth-order behaviour
        'uniform': 3,
        'yoshida-4': 4,
        'yoshida-8': 5,
    }
    for name, index in expected_indices.items():
        if name == 'yoshida4':
            with pytest.warns(FutureWarning, match="use 'yoshida-6'"):
                magnet = Magnet(integrator=name)
            exp_name = 'yoshida-6'
            exp_index = expected_indices[exp_name]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                magnet = Magnet(integrator=name)
            exp_name = name
            exp_index = index
        assert magnet.integrator == exp_name
        assert magnet._integrator == exp_index


@for_all_test_contexts
def test_adaptive_integrator_keeps_historical_yoshida6(test_context):
    kwargs = {
        'length': 1.3,
        'angle': 0.07,
        'k1': 0.21,
        'k2': -0.13,
        'num_multipole_kicks': 7,
        'model': 'bend-kick-bend',
        '_context': test_context,
    }
    adaptive = Magnet(integrator='adaptive', **kwargs)
    yoshida6 = Magnet(integrator='yoshida-6', **kwargs)
    p_adaptive = make_particles(test_context)
    p_yoshida6 = p_adaptive.copy()
    adaptive.track(p_adaptive)
    yoshida6.track(p_yoshida6)
    for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
        xo.assert_allclose(
            getattr(p_adaptive, coordinate),
            getattr(p_yoshida6, coordinate),
            rtol=0,
            atol=0,
        )


@for_all_test_contexts
def test_check_uniform_integrator(test_context):
    mm1 = Magnet(angle=0.1 * 2.0, k1=0.3, k0=0.2, length=2.0, _context=test_context)
    mm2 = mm1.copy()

    mm1.edge_entry_active = False
    mm1.edge_exit_active = False
    mm2.edge_entry_active = False
    mm2.edge_exit_active = False

    mm1.integrator = 'uniform'
    mm2.integrator = 'teapot'
    mm1.num_multipole_kicks = 1
    mm2.num_multipole_kicks = 1

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm1.track(p_test)
    mm2.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_ref_cpu.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[mm1])
    line.build_tracker(compile=False, _context=test_context)
    line.track(p_test, backtrack=True)
    p_test.move(_context=xo.ContextCpu())
    xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
    xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

    # more kicks (needs loser thresholds)
    mm1.num_multipole_kicks = 10
    mm2.num_multipole_kicks = 10

    p_test = p0.copy()
    p_ref = p0.copy()

    mm1.track(p_test)
    mm2.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_ref_cpu.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=0, rtol=5e-3)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=0, rtol=5e-3)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=0, rtol=1e-2)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=0, rtol=5e-3)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=0, rtol=5e-3)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=0, rtol=5e-3)

    # Test backtracking
    line = xt.Line(elements=[mm1])
    line.build_tracker(compile=False, _context=test_context)
    line.track(p_test, backtrack=True)
    p_test.move(_context=xo.ContextCpu())
    xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
    xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)


_N_KICKS_PER_SLICE = {
    'yoshida-4': 3,
    'yoshida-6': 7,
    'yoshida-8': 15,
}

_CONVERGENCE_SLICES = {
    'yoshida-4': [16, 64, 256, 512, 1024],
    'yoshida-6': [4, 8, 16, 32, 64],
    'yoshida-8': [2, 4, 8, 16],
}

_COORDS = ('x', 'px', 'y', 'py', 'delta')
_FINAL_POINT_RTOL = {
    'yoshida-4': 2e-12,
    'yoshida-6': 3e-13,
    'yoshida-8': 5e-13,
}

def _max_rel_error(particle, exact_coords):
    # Relative errors, as all tracked coordinates are non-zero.
    return max(
        abs(getattr(particle, coord)[0] - exact_coords[coord])
        / abs(exact_coords[coord])
        for coord in _COORDS
    )


@for_all_test_contexts
@pytest.mark.parametrize(
    'integrator, expected_order',
    [
        ('yoshida-4', 4),
        ('yoshida-6', 6),
        ('yoshida-8', 8),
    ],
)
def test_integrator_convergence_order_large_angle_bend(test_context, integrator, expected_order):
    stages = _N_KICKS_PER_SLICE[integrator]

    bend_kwargs = {
        'length': 1.0,
        'angle': np.pi / 10,
        'edge_entry_active': False,
        'edge_exit_active': False,
    }
    exact = Magnet(model='bend-kick-bend', **bend_kwargs)
    p = xt.Particles(
        kinetic_energy0=1e9, x=3e-3, px=1e-3, y=2e-3, py=-1e-3, delta=2e-3,
        _context=test_context
    )
    p_exact = p.copy()
    exact.track(p_exact)
    bend_exact_coords = {coord: getattr(p_exact, coord)[0] for coord in _COORDS}
    num_slices = _CONVERGENCE_SLICES[integrator]
    errors = []
    for n_slices in num_slices:
        mm = Magnet(
            model='rot-kick-rot-low-order',
            integrator=integrator,
            num_multipole_kicks=stages * n_slices,
            **bend_kwargs,
        )
        p_slice = p.copy()
        mm.track(p_slice)
        errors.append(_max_rel_error(p_slice, bend_exact_coords))

    slope, _ = np.polyfit(np.log(num_slices), np.log(errors), 1)
    xo.assert_allclose(-slope, expected_order, atol=0.2, rtol=0)

    # Takes the final n_slices[-1] point
    rtol = _FINAL_POINT_RTOL[integrator]
    for coord in _COORDS:
        xo.assert_allclose(
            getattr(p_slice, coord)[0], bend_exact_coords[coord], atol=0, rtol=rtol,
        )
