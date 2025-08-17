import xtrack as xt
import numpy as np
import pytest

from xobjects.test_helpers import for_all_test_contexts
import xobjects as xo
from cpymad.madx import Madx

assert_allclose = np.testing.assert_allclose

@for_all_test_contexts
@pytest.mark.parametrize(
    'slice_mode',
    [None, 'thin', 'thick'],
    ids=['no_slice', 'thin_slice', 'thick_slice'])
def test_test_tilt_shifts_vs_sandwtch(test_context, slice_mode):

    ele_test = [
        xt.Bend(k0=0.04, h=0.03, length=1,
                k1=0.1,
                knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4],
                edge_entry_angle=0.05, edge_exit_angle=0.06,
                edge_entry_hgap=0.06, edge_exit_hgap=0.07,
                edge_entry_fint=0.08, edge_exit_fint=0.09,
                ),
        xt.Quadrupole(k1=2., k1s=-3., length=3.),
        xt.Sextupole(k2=0.1, k2s=0.2, length=0.3),
        xt.Octupole(k3=0.1, k3s=0.2, length=0.4),
        xt.Multipole(knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4],
                            length=0.4, hxl=0.1)
    ]

    for elem in ele_test:
        print('ele type:', elem.__class__.__name__)

        shift_x = 1e-3
        shift_y = 2e-3
        shift_s = 10e-3
        rot_s_rad = -0.4

        line_test = xt.Line(elements=[elem.copy()])

        line_ref = xt.Line(elements=[
            xt.Drift(length=shift_s),
            xt.XYShift(dx=shift_x, dy=shift_y),
            xt.SRotation(angle=np.rad2deg(rot_s_rad)),
            elem.copy(),
            xt.SRotation(angle=np.rad2deg(-rot_s_rad)),
            xt.XYShift(dx=-shift_x, dy=-shift_y),
            xt.Drift(length=-shift_s)
        ])
        line_ref.config.XTRACK_GLOBAL_XY_LIMIT = 1000

        if slice_mode is not None:
            line_test.slice_thick_elements(
                slicing_strategies=[xt.Strategy(xt.Teapot(3, mode=slice_mode))])
            line_ref.slice_thick_elements(
                slicing_strategies=[xt.Strategy(xt.Teapot(3, mode=slice_mode))])

        line_test['e0'].rot_s_rad = rot_s_rad
        line_test['e0'].shift_x = shift_x
        line_test['e0'].shift_y = shift_y
        line_test['e0'].shift_s = shift_s

        p_test = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                              _context=test_context)
        p_ref = p_test.copy()
        p0 = p_test.copy()

        line_test.build_tracker(_context=test_context)
        line_ref.build_tracker(_context=test_context)

        line_test.track(p_test)
        line_ref.track(p_ref)

        p_test.move(_context=xo.context_default)
        p_ref.move(_context=xo.context_default)

        assert_allclose = np.testing.assert_allclose
        assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
        assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
        assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
        assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)
        assert_allclose(p_test.zeta, p_ref.zeta, rtol=0, atol=5e-12)
        assert_allclose(p_test.delta, p_ref.delta, rtol=0, atol=1e-13)

        # Test backtrack
        p_test.move(_context=test_context)
        p_ref.move(_context=test_context)

        line_test.track(p_test, backtrack=True)
        line_ref.track(p_ref, backtrack=True)

        p_test.move(_context=xo.context_default)
        p_ref.move(_context=xo.context_default)

        assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-11)
        assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-11)
        assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-11)
        assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-11)
        assert_allclose(p_test.zeta, p_ref.zeta, rtol=0, atol=1e-7)
        assert_allclose(p_test.delta, p_ref.delta, rtol=0, atol=1e-11)

        p0.move(_context=xo.context_default)
        assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-11)
        assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-11)
        assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-11)
        assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-11)
        assert_allclose(p_test.zeta, p0.zeta, rtol=0, atol=1e-7)
        assert_allclose(p_test.delta, p0.delta, rtol=0, atol=1e-11)

def test_tilt_shifts_vs_madx():

    mad = Madx()

    tilt_deg = 12
    shift_x = -2e-3
    shift_y = 3e-3
    k1 = 0.2
    tilt_rad = np.deg2rad(tilt_deg)

    x_test = 1e-3
    px_test = 2e-3
    y_test = 3e-3
    py_test = 4e-3

    mad.input(f"""
    k1={k1};

    elm: quadrupole,
        k1:=k1,
        l=1,
        tilt={tilt_rad};

    seq: sequence, l=1;
    elm: elm, at=0.5;
    endsequence;

    beam, particle=proton, gamma=100;
    use, sequence=seq;

    select,flag=error,clear;
    select,flag=error,pattern=elm;
    ealign, dx={shift_x}, dy={shift_y};

    twiss, betx=1, bety=1, x={x_test}, px={px_test}, y={y_test}, py={py_test};

    """)

    elm = xt.Quadrupole(k1=k1, length=1)

    elm_tilted = xt.Quadrupole(k1=k1, length=1, rot_s_rad=tilt_rad,
                            shift_x=shift_x, shift_y=shift_y)

    lsandwitch = xt.Line(elements=[
        xt.XYShift(dx=shift_x, dy=shift_y),
        xt.SRotation(angle=tilt_deg),
        elm,
        xt.SRotation(angle=-tilt_deg),
        xt.XYShift(dx=-shift_x, dy=-shift_y)
    ])
    lsandwitch.build_tracker()

    l_tilted = xt.Line(elements=[elm_tilted])
    l_tilted.build_tracker()

    lmad = xt.Line.from_madx_sequence(mad.sequence.seq, enable_align_errors=True)
    lmad.build_tracker()

    p0 = xt.Particles(x=x_test, px=px_test, y=y_test, py=py_test, gamma0=100)

    pmad = p0.copy()
    lmad.track(pmad)

    psandwitch = p0.copy()
    lsandwitch.track(psandwitch)

    plinetilted = p0.copy()
    l_tilted.track(plinetilted)

    peletitled = p0.copy()
    elm_tilted.track(peletitled)

    assert elm.rot_s_rad == 0
    elm.rot_s_rad = tilt_rad
    elm.shift_x = shift_x
    elm.shift_y = shift_y
    pprop = p0.copy()
    elm.track(pprop)

    for pp in [psandwitch, plinetilted, pmad, peletitled, pprop]:
        assert_allclose(pp.x, mad.table.twiss.x[-1], rtol=0, atol=1e-12)
        assert_allclose(pp.px, mad.table.twiss.px[-1], rtol=0, atol=1e-12)
        assert_allclose(pp.y, mad.table.twiss.y[-1], rtol=0, atol=1e-12)
        assert_allclose(pp.py, mad.table.twiss.py[-1], rtol=0, atol=1e-12)
        assert_allclose(pp.zeta, pp.beta0[0]*mad.table.twiss.t[-1], rtol=0, atol=1e-12)


@for_all_test_contexts
def test_shift_x(test_context):

    k1 = 2.
    length = 0.1

    quad = xt.Quadrupole(k1=k1, length=length, shift_x=1e-3, _context=test_context)

    assert quad.shift_x == 1e-3
    assert quad.shift_y == 0
    assert quad._sin_rot_s == 0.0
    assert quad._cos_rot_s == 1.0

    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    quad.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.px, -k1 * length * -1e-3, rtol=5e-3, atol=0)

    # Change the shift
    quad.shift_x = 2e-3
    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    quad.track(p)
    p.move(_context=xo.context_default)
    assert_allclose(p.px, -k1 * length * -2e-3, rtol=5e-3, atol=0)

    # Make a line
    line = xt.Line(elements=[quad])

    # Slice the line:
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(3))])
    line.build_tracker(_context=test_context)

    tt = line.get_table()
    assert len(tt.rows[r'e0\.\..*']) == 5

    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    line.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.px, -k1 * length * -2e-3, rtol=5e-3, atol=0)

    # Change the shift
    quad.shift_x = 3e-3
    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    line.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.px, -k1 * length * -3e-3, rtol=5e-3, atol=0)

@for_all_test_contexts
def test_shift_y(test_context):

    k1 = 2.
    length = 0.1

    quad = xt.Quadrupole(k1=k1, length=length, shift_y=1e-3, _context=test_context)

    assert quad.shift_x == 0
    assert quad.shift_y == 1e-3
    assert quad._sin_rot_s == 0.0
    assert quad._cos_rot_s == 1.0

    p = xt.Particles(y=0, p0c=1e12, _context=test_context)
    quad.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.py, k1 * length * -1e-3, rtol=5e-3, atol=0)

    # Change the shift
    quad.shift_y = 2e-3
    p = xt.Particles(y=0, p0c=1e12, _context=test_context)
    quad.track(p)
    p.move(_context=xo.context_default)
    assert_allclose(p.py, k1 * length * -2e-3, rtol=5e-3, atol=0)

    # Make a line
    line = xt.Line(elements=[quad])

    # Slice the line:
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(3))])
    line.build_tracker(_context=test_context)

    tt = line.get_table()
    assert len(tt.rows[r'e0\.\..*']) == 5

    p = xt.Particles(y=0, p0c=1e12, _context=test_context)
    line.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.py, k1 * length * -2e-3, rtol=5e-3, atol=0)

    # Change the shift
    quad.shift_y = 3e-3
    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    line.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.py, k1 * length * -3e-3, rtol=5e-3, atol=0)

@for_all_test_contexts
def test_rot_s(test_context):

    k0 = 2.
    length = 0.1
    rot_s_rad = 0.2

    bend = xt.Bend(k0=k0, length=length, rot_s_rad=rot_s_rad, _context=test_context)

    assert bend.shift_x == 0
    assert bend.shift_y == 0
    assert_allclose(bend._sin_rot_s, np.sin(rot_s_rad), rtol=0, atol=1e-14)
    assert_allclose(bend._cos_rot_s, np.cos(rot_s_rad), rtol=0, atol=1e-14)

    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    bend.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3, atol=0)
    assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3, atol=0)

    rot_s_rad = 0.3
    bend.rot_s_rad = rot_s_rad
    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    bend.track(p)
    p.move(_context=xo.context_default)
    assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3, atol=0)
    assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3, atol=0)

    # Make a line
    line = xt.Line(elements=[bend])

    # Slice the line:
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(3))])
    line.build_tracker(_context=test_context)

    tt = line.get_table()
    assert len(tt.rows[r'e0\.\..?']) == 3

    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    line.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3,
                    atol=0)
    assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3,
                    atol=0)

    rot_s_rad = 0.4
    bend.rot_s_rad = rot_s_rad
    p = xt.Particles(x=0, p0c=1e12, _context=test_context)
    line.track(p)
    p.move(_context=xo.context_default)

    assert_allclose(p.px, -k0 * length * np.cos(rot_s_rad), rtol=5e-3,
                    atol=0)
    assert_allclose(p.py, -k0 * length * np.sin(rot_s_rad), rtol=5e-3,
                    atol=0)
