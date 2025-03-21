import numpy as np
import pytest

import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

from xtrack.beam_elements.magnets import Magnet, MagnetEdge


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


@for_all_test_contexts
def test_magnet_expanded_drift(test_context):
    magnet = Magnet(
        length=2.0,
        k0=0.0,
        k1=0.0,
        h=0.0,
        integrator='teapot',
        model='drift-kick-drift-expanded',
        _context=test_context,
    )

    assert magnet.model == 'drift-kick-drift-expanded'
    assert magnet.integrator == 'teapot'

    drift = xt.Drift(length=2.0, _context=test_context)

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    drift.track(p_ref)

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


@for_all_test_contexts
def test_magnet_exact_drift(test_context):
    magnet = Magnet(
        length=2.0,
        k0=0.0,
        k1=0.0,
        h=0.0,
        integrator='teapot',
        model='drift-kick-drift-exact',
        _context=test_context,
    )

    assert magnet.model == 'drift-kick-drift-exact'
    assert magnet.integrator == 'teapot'

    exact_drift = xt.Solenoid(length=2.0, _context=test_context)  # Solenoid is exact drift when off

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    exact_drift.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_sextupole(test_context):
    magnet = Magnet(
        length=2.0,
        k2=3,
        k2s=5,
        integrator='teapot',
        model='drift-kick-drift-expanded',
        _context=test_context,
    )

    sextupole = xt.Sextupole(length=2.0, k2=3., k2s=5., _context=test_context)
    assert magnet.integrator == 'teapot'
    assert magnet.model == 'drift-kick-drift-expanded'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    sextupole.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_sextupole_with_kicks_that_do_nothing(test_context):
    magnet = Magnet(
        length=2.0,
        k2=3,
        k2s=5,
        num_multipole_kicks=5,
        integrator='uniform',
        model='drift-kick-drift-expanded',
        _context=test_context,
    )

    sextupole = xt.Sextupole(
        length=2.0,
        k2=3.,
        k2s=5.,
        num_multipole_kicks=5,
        _context=test_context,
    )

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    sextupole.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_sextupole_with_kicks(test_context):
    magnet = Magnet(
        length=2.0,
        k2s=5,
        knl=[0., 0., 3. * 2., 0., 0., 0.],
        num_multipole_kicks=5,
        integrator='teapot',
        model='drift-kick-drift-expanded',
        _context=test_context,
    )

    sextupole = xt.Sextupole(
        length=2.0,
        k2=3.,
        k2s=5.,
        num_multipole_kicks=5,
        integrator='teapot',
        _context=test_context,
    )
    assert sextupole.integrator == 'teapot'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    sextupole.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_sextupole_with_skew_kick(test_context):
    magnet = Magnet(
        length=2.0,
        integrator='uniform',
        model='drift-kick-drift-expanded',
        k2=0,
        k2s=5,
        knl=[0, 0, 6, 0, 0, 0],
        ksl=[0, 0, -2, 0, 0, 0],
        num_multipole_kicks=5,
        _context=test_context,
    )

    sextupole = xt.Sextupole(
        length=2.0,
        k2=3,
        k2s=5,
        num_multipole_kicks=5,
        _context=test_context,
    )
    sextupole.ksl[2] = -2.

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    sextupole.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
@pytest.mark.parametrize('integrator', ['adaptive', 'teapot'])
def test_magnet_quadrupole(test_context, integrator):
    magnet = Magnet(
        length=2.0,
        k1=3,
        num_multipole_kicks=1,
        integrator='teapot',
        model='mat-kick-mat',
        _context=test_context,
    )

    quadrupole = xt.Quadrupole(
        length=2,
        k1=3,
        num_multipole_kicks=1,
        integrator=integrator,
        _context=test_context,
    )
    assert quadrupole.integrator == integrator

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    quadrupole.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_curved_quad(test_context):
    magnet = Magnet(
        length=2.0,
        h=0.05,
        k1=-0.3,
        num_multipole_kicks=15,
        integrator='yoshida4',
        model='rot-kick-rot',
        _context=test_context,
    )

    bend = xt.Bend(
        length=2.0,
        k1=-0.3,
        h=0.05,
        num_multipole_kicks=15,
        _context=test_context,
    )

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    bend.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_bend_auto_no_kicks(test_context):
    magnet = Magnet(
        length=2.0,
        h=0.05,
        k1=0,
        model='bend-kick-bend',
        integrator='yoshida4',
        num_multipole_kicks=0,
        _context=test_context,
    )

    eref = xt.Bend(
        length=2.0,
        h=0.05,
        num_multipole_kicks=0,
        _context=test_context,
    )

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    eref.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
def test_magnet_bend_auto_quad_kick(test_context):
    magnet = Magnet(
        length=2.0,
        model='bend-kick-bend',
        integrator='yoshida4',
        num_multipole_kicks=1,
        h=0.05,
        k1=0.3,
        _context=test_context,
    )

    bend = xt.Bend(
        length=2.0,
        h=0.05,
        k1=0.3,
        model='bend-kick-bend',
        num_multipole_kicks=1,
        edge_entry_active=False,
        edge_exit_active=False,
        _context=test_context,
    )

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    bend.track(p_ref)

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
    line = xt.Line(elements=[magnet])
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


@for_all_test_contexts
@pytest.mark.parametrize('model', ['bend-kick-bend', 'rot-kick-rot', 'expanded'])
def test_magnet_bend_dip_quad_kick(model, test_context):
    magnet = Magnet(
        length=2.0,
        k0=0.2,
        k1=0.3,
        h=0.1,
        integrator='yoshida4',
        num_multipole_kicks=10,
        _context=test_context,
    )

    bend = xt.Bend(length=2.0, h=0.1, k1=0.3, k0=0.2, num_multipole_kicks=10)
    bend.edge_entry_active = False
    bend.edge_exit_active = False

    magnet.model = model
    bend.model = model

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    bend.track(p_ref)

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
    line = xt.Line(elements=[magnet])
    line.build_tracker(compile=False, _context=test_context)
    line.track(p_test, backtrack=True)
    p_test.move(_context=xo.ContextCpu())
    xo.assert_allclose(p_test.s, 0.0, rtol=0, atol=1e-7)
    xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
@pytest.mark.parametrize('model', ['bend-kick-bend', 'rot-kick-rot', 'expanded'])
def test_magnet_bend_dip_quad_kick_with_multipoles(model, test_context):
    magnet = Magnet(
        length=2.0,
        h=0.1,
        k0=0.2,
        k1=0.3,
        k2=0.1,
        k3=0.15,
        k0s=0.02,
        k1s=0.03,
        k2s=0.01,
        k3s=0.02,
        knl=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ksl=[0.6, 0.5, 0.4, 0.15, 0.2, 0.1],
        num_multipole_kicks=10,
        integrator='yoshida4',
        _context=test_context,
    )

    bend = xt.Bend(
        length=2.0,
        h=0.1,
        k1=0.3,
        k0=0.2,
        num_multipole_kicks=10,
        _context=test_context,
    )
    bend.edge_entry_active = False
    bend.edge_exit_active = False
    bend.knl = [0.1, 0.2, 0.3 + 0.1 * 2, 0.4 + 0.15 * 2, 0.5, 0.6]
    bend.ksl = [0.6 + 0.02 * 2, 0.5 + 0.03 * 2, 0.4 + 0.01 * 2, 0.15 + 0.02 * 2,
        0.2, 0.1]

    magnet.model = model
    bend.model = model

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    magnet.track(p_test)
    bend.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_ref_cpu.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-14, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[magnet])
    line.build_tracker(compile=False, _context=test_context)
    line.track(p_test, backtrack=True)
    p_test.move(_context=xo.ContextCpu())
    xo.assert_allclose(p_test.s, 0.0, rtol=0, atol=1e-7)
    xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_check_uniform_integrator(test_context):
    mm1 = Magnet(h=0.1, k1=0.3, k0=0.2, length=2.0, _context=test_context)
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


@for_all_test_contexts
def test_edge_suppressed_edge(test_context):
    e_test = MagnetEdge(model='suppressed', kn=[0], ks=[0], _context=test_context)
    e_ref = xt.DipoleEdge(model='suppressed', k=0, _context=test_context)

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_linear_edge_does_nothing(test_context):
    e_test = MagnetEdge(model='linear', kn=[0], ks=[0], _context=test_context)
    e_ref = xt.DipoleEdge(model='linear', k=0, _context=test_context)

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_full_edge_does_nothing(test_context):
    e_test = MagnetEdge(model='full', kn=[0], ks=[0], _context=test_context)
    e_ref = xt.DipoleEdge(model='full', k=0, _context=test_context)

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_only_linear_edge(test_context):
    e_test = MagnetEdge(
        model='linear', kn=[3], face_angle=0.1, face_angle_feed_down=0.2,
        fringe_integral=0.3,
        half_gap=0.4, _context=test_context
    )
    e_ref = xt.DipoleEdge(
        model='linear', k=3, e1=0.1, e1_fd=0.2, fint=0.3, hgap=0.4,
        _context=test_context
    )

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())
    
    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_full_edge_with_dipole_component(test_context):
    e_test = MagnetEdge(
        model='full', kn=[3], face_angle=0.1, face_angle_feed_down=0.2,
        fringe_integral=0.3,
        half_gap=0.4, _context=test_context
    )
    e_ref = xt.DipoleEdge(
        model='full', k=3, e1=0.1, e1_fd=0.2, fint=0.3, hgap=0.4,
        _context=test_context
    )

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_multipole_fringe_without_dipole_component(test_context):
    e_test = MagnetEdge(
        model='full', kn=[0, 2, 3], k_order=2, _context=test_context
    )
    e_ref = xt.MultipoleEdge(kn=[0, 2, 3], order=2, _context=test_context)

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())
    
    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_full_model_with_dipole_component_no_angle(test_context):
    e_test = MagnetEdge(
        model='full', kn=[3, 4, 5], fringe_integral=0.3, half_gap=0.4, k_order=2,
        _context=test_context
    )
    e_ref = [
        xt.DipoleEdge(model='full', k=3, fint=0.3, hgap=0.4),
        xt.MultipoleEdge(kn=[0, 4, 5], order=2),
    ]

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)

    mini_line = xt.Line(elements=e_ref)
    mini_line.build_tracker(_context=test_context)
    mini_line.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_full_model_with_dipole_component_and_angle(test_context):
    e_test = MagnetEdge(
        model='full', kn=[3, 4, 5], face_angle=0.2, face_angle_feed_down=0.0,
        fringe_integral=0.3,
        half_gap=0.4, k_order=2, _context=test_context
    )
    e_ref = [
        xt.YRotation(angle=np.rad2deg(-0.2)),
        # The rotation is also the other way than in the underlying map :'(
        xt.DipoleEdge(model='full', k=3, fint=0.3, hgap=0.4),
        xt.MultipoleEdge(kn=[0, 4, 5], order=2),
        xt.Wedge(angle=-0.2, k=3),
    ]

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)

    mini_line = xt.Line(elements=e_ref)
    mini_line.build_tracker(_context=test_context)
    mini_line.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_full_model_with_dipole_component_and_angle_exit(test_context):
    e_test = MagnetEdge(
        model='full', kn=[3, 4, 5], is_exit=True, face_angle=0.2,
        face_angle_feed_down=0.0,
        fringe_integral=0.3, half_gap=0.4, k_order=2, _context=test_context
    )
    e_ref = [
        xt.Wedge(angle=-0.2, k=3),
        xt.MultipoleEdge(kn=[0, 4, 5], is_exit=True, order=2),
        xt.DipoleEdge(model='full', k=-3, fint=0.3, hgap=0.4),
        xt.YRotation(angle=np.rad2deg(-0.2)),
    ]

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)

    mini_line = xt.Line(elements=e_ref)
    mini_line.build_tracker(_context=test_context)
    mini_line.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_edge_linear_edge_exit(test_context):
    e_test = MagnetEdge(
        model='linear', is_exit=True, kn=[3], face_angle=0.1,
        face_angle_feed_down=0.2,
        fringe_integral=0.3, half_gap=0.4, _context=test_context
    )
    e_ref = xt.DipoleEdge(
        model='linear', k=3, e1=0.1, e1_fd=0.2, fint=0.3, hgap=0.4,
        _context=test_context,
    )

    p0 = xt.Particles(
        kinetic_energy0=50e6,
        x=1e-2, y=2e-2, zeta=1e-2, px=10e-2, py=20e-2, delta=1e-2,
        _context=test_context,
    )

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)
    e_ref.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_magnet_and_edge_only_linear(test_context):
    bb = xt.Bend(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    bb.edge_entry_active = 1
    bb.edge_exit_active = 0
    bb.model = 'rot-kick-rot'
    bb.num_multipole_kicks = 10

    mm = Magnet(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )
    mm.edge_entry_active = 1
    mm.edge_exit_active = 0
    mm.model = 'rot-kick-rot'
    mm.num_multipole_kicks = 10

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    bb.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[mm])
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


@for_all_test_contexts
def test_magnet_and_edge_exit_alone(test_context):
    bb = xt.Bend(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    bb.edge_entry_active = 0
    bb.edge_exit_active = 1
    bb.model = 'rot-kick-rot'
    bb.num_multipole_kicks = 10

    mm = Magnet(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    mm.edge_entry_active = 0
    mm.edge_exit_active = 1
    mm.model = 'rot-kick-rot'
    mm.num_multipole_kicks = 10

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    bb.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[mm])
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


@for_all_test_contexts
def test_magnet_and_edge_full_bend_linear_edges(test_context):
    bb = xt.Bend(
        h=0.1, k0=0.11, length=10,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    bb.edge_entry_active = 1
    bb.edge_exit_active = 1
    bb.model = 'rot-kick-rot'
    bb.num_multipole_kicks = 10
    bb.edge_entry_model = 'linear'
    bb.edge_exit_model = 'linear'

    mm = Magnet(
        h=0.1, k0=0.11, length=10,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    mm.edge_entry_active = 1
    mm.edge_exit_active = 1
    mm.model = 'rot-kick-rot'
    mm.num_multipole_kicks = 10
    mm.edge_entry_model = 'linear'
    mm.edge_exit_model = 'linear'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    bb.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-13, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[mm])
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


@for_all_test_contexts
def test_magnet_and_edge_nonlinear_entry_alone(test_context):
    bb = xt.Bend(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    bb.edge_entry_active = 1
    bb.edge_exit_active = 0
    bb.model = 'rot-kick-rot'
    bb.num_multipole_kicks = 10
    bb.edge_entry_model = 'full'

    mm = Magnet(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    mm.edge_entry_active = 1
    mm.edge_exit_active = 0
    mm.model = 'rot-kick-rot'
    mm.num_multipole_kicks = 10
    mm.edge_entry_model = 'full'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    bb.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_magnet_and_edge_nonlinear_exit_alone(test_context):
    bb = xt.Bend(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    bb.edge_entry_active = 0
    bb.edge_exit_active = 1
    bb.model = 'rot-kick-rot'
    bb.num_multipole_kicks = 10
    bb.edge_exit_model = 'full'

    mm = Magnet(
        h=0.1, k0=0.11, length=0,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2,
        _context=test_context,
    )

    mm.edge_entry_active = 0
    mm.edge_exit_active = 1
    mm.model = 'rot-kick-rot'
    mm.num_multipole_kicks = 10
    mm.edge_exit_model = 'full'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    bb.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_magnet_and_edge_nonlinear_both_edges(test_context):
    bb = xt.Bend(
        h=0.1, k0=0.11, length=10,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2
    )

    bb.edge_entry_active = 1
    bb.edge_exit_active = 1
    bb.model = 'rot-kick-rot'
    bb.num_multipole_kicks = 10
    bb.edge_entry_model = 'full'
    bb.edge_exit_model = 'full'

    mm = Magnet(
        h=0.1, k0=0.11, length=10,
        edge_entry_angle=0.02, edge_exit_angle=0.03,
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        edge_entry_fint=0.1, edge_exit_fint=0.2
    )

    mm.edge_entry_active = 1
    mm.edge_exit_active = 1
    mm.model = 'rot-kick-rot'
    mm.num_multipole_kicks = 10
    mm.edge_entry_model = 'full'
    mm.edge_exit_model = 'full'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    bb.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=3e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_magnet_and_edge_quadrupole_nonlinear_fringes(test_context):
    qq = xt.Quadrupole(k1=0.11, length=3, _context=test_context)
    qq.edge_entry_active = 1
    qq.edge_exit_active = 1

    mm = Magnet(
        k0=0, k1=0.11, length=3,
        edge_entry_model='full', edge_exit_model='full',
        edge_entry_fint=0.1, edge_exit_fint=0.2,  # should be ignored
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        _context=test_context,
    )  # should be ignored
    mm.edge_entry_active = 1
    mm.edge_exit_active = 1
    mm.model = 'mat-kick-mat'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    qq.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_magnet_and_edge_sextupole_nonlinear_fringes(test_context):
    ss = xt.Sextupole(k2=0.11 + 0.03 / 3., k2s=0.12 - 0.04 / 3, length=3, _context=test_context)
    ss.edge_entry_active = 1
    ss.edge_exit_active = 1

    mm = Magnet(
        k0=0, k2=0.11, k2s=0.12, length=3,
        knl=[0, 0, 0.03], ksl=[0, 0, -0.04],
        edge_entry_model='full', edge_exit_model='full',
        edge_entry_fint=0.1, edge_exit_fint=0.2,  # should be ignored
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        _context=test_context,
    )  # should be ignored
    mm.edge_entry_active = 1
    mm.edge_exit_active = 1
    mm.model = 'drift-kick-drift-expanded'
    mm.num_multipole_kicks = 1
    mm.integrator = 'uniform'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    ss.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_magnet_and_edge_octupole_nonlinear_fringes(test_context):
    oo = xt.Octupole(k3=0.3 + 0.02 / 3., k3s=0.4 - 0.07 / 3, length=3, _context=test_context)
    oo.edge_entry_active = 1
    oo.edge_exit_active = 1
    oo.edge_entry_active = 1
    oo.edge_exit_active = 1

    mm = Magnet(
        k0=0, k3=0.3, k3s=0.4, length=3,
        knl=[0, 0, 0, 0.02], ksl=[0, 0, 0, -0.07],
        edge_entry_model='full', edge_exit_model='full',
        edge_entry_fint=0.1, edge_exit_fint=0.2,  # should be ignored
        edge_entry_hgap=0.04, edge_exit_hgap=0.05,
        _context=test_context,
    )  # should be ignored
    mm.edge_entry_active = 1
    mm.edge_exit_active = 1
    mm.model = 'drift-kick-drift-expanded'
    mm.num_multipole_kicks = 1
    mm.integrator = 'uniform'

    p0 = make_particles(test_context)
    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    oo.track(p_ref)

    p_test_cpu = p_test.copy(_context=xo.ContextCpu())
    p_ref_cpu = p_ref.copy(_context=xo.ContextCpu())

    xo.assert_allclose(p_test_cpu.x, p_ref_cpu.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.y, p_ref_cpu.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.zeta, p_ref_cpu.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.px, p_ref_cpu.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.py, p_ref_cpu.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test_cpu.delta, p_ref_cpu.delta, atol=1e-15, rtol=0)

    line = xt.Line(elements=[mm])
    line.build_tracker(compile=False, _context=test_context)
    line.track(p_test, backtrack=True)
    p_test.move(_context=xo.ContextCpu())
    assert np.all(p_test.state == -32)
