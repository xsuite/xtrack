# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts
import pytest
from cpymad.madx import Madx


@for_all_test_contexts
def test_rect_ellipse(test_context):
    ctx2np = test_context.nparray_from_context_array

    aper_rect_ellipse = xt.LimitRectEllipse(
        _context=test_context, max_x=23e-3, max_y=18e-3, a=23e-2, b=23e-2
    )
    aper_ellipse = xt.LimitRectEllipse(_context=test_context, a=23e-2, b=23e-2)
    aper_rect = xt.LimitRect(
        _context=test_context, max_x=23e-3, min_x=-23e-3, max_y=18e-3, min_y=-18e-3
    )

    XX, YY = np.meshgrid(
        np.linspace(-30e-3, 30e-3, 100), np.linspace(-30e-3, 30e-3, 100)
    )
    x_part = XX.flatten()
    y_part = XX.flatten()
    part_re = xp.Particles(_context=test_context, x=x_part, y=y_part)
    part_e = part_re.copy()
    part_r = part_re.copy()

    aper_rect_ellipse.track(part_re)
    aper_ellipse.track(part_e)
    aper_rect.track(part_r)

    flag_re = ctx2np(part_re.state)[np.argsort(ctx2np(part_re.particle_id))]
    flag_r = ctx2np(part_r.state)[np.argsort(ctx2np(part_r.particle_id))]
    flag_e = ctx2np(part_e.state)[np.argsort(ctx2np(part_e.particle_id))]

    assert np.all(flag_re == (flag_r & flag_e))


@for_all_test_contexts
def test_aperture_racetrack(test_context):
    aper = xt.LimitRacetrack(
        _context=test_context,
        min_x=-5e-2,
        max_x=10e-2,
        min_y=-2e-2,
        max_y=4e-2,
        a=2e-2,
        b=1e-2,
    )

    xy_out = np.array(
        [
            [-4.8e-2, 3.7e-2],
            [9.6e-2, 3.7e-2],
            [-4.5e-2, -1.8e-2],
            [9.8e-2, -1.8e-2],
        ]
    )

    xy_in = np.array(
        [
            [-4.2e-2, 3.3e-2],
            [9.4e-2, 3.6e-2],
            [-3.8e-2, -1.8e-2],
            [9.2e-2, -1.8e-2],
        ]
    )

    xy_all = np.concatenate([xy_out, xy_in], axis=0)

    particles = xp.Particles(
        _context=test_context, p0c=6500e9, x=xy_all[:, 0], y=xy_all[:, 1]
    )

    aper.track(particles)

    part_state = test_context.nparray_from_context_array(particles.state)
    part_id = test_context.nparray_from_context_array(particles.particle_id)

    assert np.all(part_state[part_id < 4] == 0)
    assert np.all(part_state[part_id >= 4] == 1)


@for_all_test_contexts
def test_aperture_polygon(test_context):
    np2ctx = test_context.nparray_to_context_array
    ctx2np = test_context.nparray_from_context_array

    x_vertices = np.array([1.5, 0.2, -1, -1, 1]) * 1e-2
    y_vertices = np.array([1.3, 0.5, 1, -1, -1]) * 1e-2

    aper = xt.LimitPolygon(
        _context=test_context,
        x_vertices=np2ctx(x_vertices),
        y_vertices=np2ctx(y_vertices),
    )

    # Try some particles inside
    parttest = xp.Particles(
        _context=test_context, p0c=6500e9, x=x_vertices * 0.99, y=y_vertices * 0.99
    )
    aper.track(parttest)
    xo.assert_allclose(ctx2np(parttest.state), 1)

    # Try some particles outside
    parttest = xp.Particles(
        _context=test_context, p0c=6500e9, x=x_vertices * 1.01, y=y_vertices * 1.01
    )
    aper.track(parttest)
    xo.assert_allclose(ctx2np(parttest.state), 0)


@pytest.mark.parametrize('loader', ['cpymad', 'native'])
def test_mad_import(loader):
    mad_src = """
        m_circle: marker, apertype="circle", aperture={.2};
        m_ellipse: marker, apertype="ellipse", aperture={.2, .1};
        m_rectangle: marker, apertype="rectangle", aperture={.07, .05};
        m_rectellipse: marker, apertype="rectellipse", aperture={.2, .4, .25, .45};
        m_racetrack: marker, apertype="racetrack", aperture={.6,.4,.2,.1};
        m_octagon: marker, apertype="octagon", aperture={.4, .5, 0.5, 1.};
        m_polygon: marker, apertype="circle", aper_vx= {+5.800e-2,+5.800e-2,-8.800e-2}, aper_vy= {+3.500e-2,-3.500e-2,+0.000e+0};
        beam;
        ss: sequence,l=1;
            m_circle, at=0;
            m_ellipse, at=0.01;
            m_rectangle, at=0.02;
            m_rectellipse, at=0.03;
            m_racetrack, at=0.04;
            m_octagon, at=0.05;
            m_polygon, at=0.06;
        endsequence;
    """

    if loader == 'cpymad':
        mad = Madx()
        mad.input(mad_src)
        mad.beam()
        mad.use('ss')
        line = xt.Line.from_madx_sequence(mad.sequence.ss, install_apertures=True)
    elif loader == 'native':
        env = xt.load(string=mad_src, format='madx')
        line = env.ss

    apertures = [
        ee for ee in line.elements if ee.__class__.__name__.startswith('Limit')
    ]

    circ = apertures[0]
    assert circ.__class__.__name__ == 'LimitEllipse'
    xo.assert_allclose(circ.a_squ, 0.2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(circ.b_squ, 0.2**2, atol=1e-13, rtol=0)

    ellip = apertures[1]
    assert ellip.__class__.__name__ == 'LimitEllipse'
    xo.assert_allclose(ellip.a_squ, 0.2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(ellip.b_squ, 0.1**2, atol=1e-13, rtol=0)

    rect = apertures[2]
    assert rect.__class__.__name__ == 'LimitRect'
    assert rect.min_x == -0.07
    assert rect.max_x == +0.07
    assert rect.min_y == -0.05
    assert rect.max_y == +0.05

    rectellip = apertures[3]
    assert rectellip.max_x == 0.2
    assert rectellip.max_y == 0.4
    xo.assert_allclose(rectellip.a_squ, 0.25**2, atol=1e-13, rtol=0)
    xo.assert_allclose(rectellip.b_squ, 0.45**2, atol=1e-13, rtol=0)

    racetr = apertures[4]
    assert racetr.__class__.__name__ == 'LimitRacetrack'
    assert racetr.min_x == -0.6
    assert racetr.max_x == +0.6
    assert racetr.min_y == -0.4
    assert racetr.max_y == +0.4
    assert racetr.a == 0.2
    assert racetr.b == 0.1

    octag = apertures[5]
    assert octag.__class__.__name__ == 'LimitPolygon'
    assert octag._xobject.x_vertices[0] == 0.4
    xo.assert_allclose(
        octag._xobject.y_vertices[0], 0.4 * np.tan(0.5), atol=1e-14, rtol=0
    )
    assert octag._xobject.y_vertices[1] == 0.5
    xo.assert_allclose(
        octag._xobject.x_vertices[1], 0.5 / np.tan(1.0), atol=1e-14, rtol=0
    )

    assert octag._xobject.y_vertices[2] == 0.5
    xo.assert_allclose(
        octag._xobject.x_vertices[2], -0.5 / np.tan(1.0), atol=1e-14, rtol=0
    )
    assert octag._xobject.x_vertices[3] == -0.4
    xo.assert_allclose(
        octag._xobject.y_vertices[3], 0.4 * np.tan(0.5), atol=1e-14, rtol=0
    )

    assert octag._xobject.x_vertices[4] == -0.4
    xo.assert_allclose(
        octag._xobject.y_vertices[4], -0.4 * np.tan(0.5), atol=1e-14, rtol=0
    )
    assert octag._xobject.y_vertices[5] == -0.5
    xo.assert_allclose(
        octag._xobject.x_vertices[5], -0.5 / np.tan(1.0), atol=1e-14, rtol=0
    )

    assert octag._xobject.y_vertices[6] == -0.5
    xo.assert_allclose(
        octag._xobject.x_vertices[6], 0.5 / np.tan(1.0), atol=1e-14, rtol=0
    )
    assert octag._xobject.x_vertices[7] == 0.4
    xo.assert_allclose(
        octag._xobject.y_vertices[7], -0.4 * np.tan(0.5), atol=1e-14, rtol=0
    )

    polyg = apertures[6]
    assert polyg.__class__.__name__ == 'LimitPolygon'
    assert len(polyg._xobject.x_vertices) == 3
    assert len(polyg._xobject.y_vertices) == 3

    xo.assert_allclose(
        polyg.x_vertices,
        np.array([5.8e-2, 5.8e-2, -8.8e-2]),
        atol=1e-14,
        rtol=0,
    )
    xo.assert_allclose(
        polyg.y_vertices, np.array([3.5e-2, -3.5e-2, 0]), atol=1e-14, rtol=0
    )


def test_mad_import_aper_offset_without_aperture():
    mad_src = """
        m: marker, aper_offset={0.1, 0.2};
        beam;
        ss: sequence,l=1;
            m, at=0.5;
        endsequence;
        """

    env = xt.load(string=mad_src, format='madx')
    line = env.ss

    assert line['m'].name_associated_aperture is None
    assert not any(ee.__class__.__name__.startswith('Limit') for ee in line.elements)


def test_mad_import_aper_offset_on_sequence_placement():
    mad_src = """
        m: marker, apertype="circle", aperture={0.2}, aper_offset={0.1, 0.2};
        beam;
        ss: sequence,l=1;
            m, at=0.5, aper_offset={0.3, 0.4};
        endsequence;
        """

    env = xt.load(string=mad_src, format='madx')
    line = env.ss

    assert line['m'].name_associated_aperture == 'm_aper'
    apertures = [
        ee for ee in line.elements if ee.__class__.__name__.startswith('Limit')
    ]
    assert len(apertures) == 1
    aperture = apertures[0]
    assert aperture.__class__.__name__ == 'LimitEllipse'
    xo.assert_allclose(aperture.a_squ, 0.2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(aperture.b_squ, 0.2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(aperture.shift_x, 0.1, atol=1e-13, rtol=0)
    xo.assert_allclose(aperture.shift_y, 0.2, atol=1e-13, rtol=0)


@for_all_test_contexts
def test_longitudinal_rect(test_context):
    aper_rect_longitudinal = xt.LongitudinalLimitRect(
        _context=test_context,
        min_zeta=-10e-3,
        max_zeta=20e-3,
        min_pzeta=-1e-3,
        max_pzeta=4e-3,
    )

    coords = np.array(
        [
            [-9e-4, 0.0],
            [15e-3, 0.0],
            [0.0, -8e-4],
            [0.0, 2e-3],
        ]
    )
    particles = xp.Particles(
        _context=test_context, zeta=coords[:, 0], pzeta=coords[:, 1]
    )

    aper_rect_longitudinal.track(particles)
    particles.move(_context=xo.ContextCpu())
    assert np.all(particles.state == 1)

    coords = np.array(
        [
            [-11e-3, 0.0],
            [22e-3, 0.0],
            [0.0, -2e-3],
            [0.0, 6e-3],
        ]
    )
    particles = xp.Particles(
        _context=test_context, zeta=coords[:, 0], pzeta=coords[:, 1]
    )

    aper_rect_longitudinal.track(particles)
    particles.move(_context=xo.ContextCpu())
    assert np.all(particles.state == -2)


@for_all_test_contexts
def test_aper_tilt(test_context):
    n_part = 300000

    particles = xt.Particles(
        _context=test_context,
        p0c=6500e9,
        x=np.random.uniform(-0.25, 0.25, n_part),
        px=np.zeros(n_part),
        y=np.random.uniform(0, 0.1, n_part),
        py=np.zeros(n_part),
    )

    tilt_deg = 10.0
    aper = xt.LimitRect(
        _context=test_context,
        min_x=-0.1,
        max_x=0.1,
        min_y=-0.001,
        max_y=0.001,
        shift_x=0.08,
        shift_y=0.04,
        rot_s_rad=np.deg2rad(tilt_deg),
    )

    aper.track(particles)

    part_state = test_context.nparray_from_context_array(particles.state)
    part_x = test_context.nparray_from_context_array(particles.x)
    part_y = test_context.nparray_from_context_array(particles.y)

    x_alive = part_x[part_state > 0]
    y_alive = part_y[part_state > 0]

    assert_allclose = np.testing.assert_allclose
    assert_allclose(np.mean(x_alive), 0.08, rtol=5e-2, atol=0)
    assert_allclose(np.mean(y_alive), 0.04, rtol=5e-2, atol=0)
    slope = np.polyfit(x_alive, y_alive, 1)[0]
    assert_allclose(slope, np.tan(np.deg2rad(tilt_deg)), rtol=5e-2, atol=0)


def test_aperture_svg_path():
    svg = {
        'path': """M4 8 10 1 13 0 12 3 5 9C6 10 6 11 7 10 7 11 8 12 7 12A1.42 1.42 0 016 13 5 5 0 004 10Q3.5 9.9 3.5 10.5T2 11.8 1.2 11 2.5 9.5 3 9A5 5 90 000 7 1.42 1.42 0 011 6C1 5 2 6 3 6 2 7 3 7 4 8"""
    }

    aper = xt.LimitPolygon(svg=svg)

    assert aper.copy().svg == aper.svg
    aper2 = xt.LimitPolygon.from_dict(aper.to_dict())
    assert aper2.svg == aper.svg

    # fmt:off
    x_expected = np.array([
        0.004   , 0.01    , 0.013   , 0.012   , 0.005   , 0.005272,
        0.005496, 0.005684, 0.005848, 0.006   , 0.006152, 0.006316,
        0.006504, 0.007   , 0.007027, 0.007096, 0.007189, 0.007288,
        0.007375, 0.007432, 0.007441, 0.007384, 0.007   , 0.006954,
        0.006894, 0.006819, 0.006732, 0.006633, 0.006523, 0.006404,
        0.006276, 0.006   , 0.005907, 0.005787, 0.005642, 0.005473,
        0.00528 , 0.005064, 0.004827, 0.00457 , 0.004   , 0.003905,
        0.00382 , 0.003745, 0.00368 , 0.003625, 0.00358 , 0.003545,
        0.00352 , 0.0035  , 0.003485, 0.00344 , 0.003365, 0.00326 ,
        0.003125, 0.00296 , 0.002765, 0.00254 , 0.002   , 0.001722,
        0.001488, 0.001298, 0.001152, 0.00105 , 0.000992, 0.000978,
        0.001008, 0.0012  , 0.001339, 0.001476, 0.001611, 0.001744,
        0.001875, 0.002004, 0.002131, 0.002256, 0.0025  , 0.002613,
        0.002712, 0.002797, 0.002868, 0.002925, 0.002968, 0.002997,
        0.003012, 0.003   , 0.002777, 0.002532, 0.002268, 0.001986,
        0.001687, 0.001372, 0.001045, 0.000706, 0.      , 0.000046,
        0.000106, 0.000181, 0.000268, 0.000367, 0.000477, 0.000596,
        0.000724, 0.001   , 0.001029, 0.001112, 0.001243, 0.001416,
        0.001625, 0.001864, 0.002127, 0.002408, 0.003   , 0.002758,
        0.002624, 0.002586, 0.002632, 0.00275 , 0.002928, 0.003154,
        0.003416
    ])
    y_expected = np.array([
        -0.008   , -0.001   , -0.      , -0.003   , -0.009   , -0.009298,
        -0.009584, -0.009846, -0.010072, -0.01025 , -0.010368, -0.010414,
        -0.010376, -0.01    , -0.010299, -0.010592, -0.010873, -0.011136,
        -0.011375, -0.011584, -0.011757, -0.011888, -0.012   , -0.012141,
        -0.012276, -0.012404, -0.012523, -0.012633, -0.012732, -0.012819,
        -0.012894, -0.013   , -0.012643, -0.012294, -0.011955, -0.011628,
        -0.011313, -0.011014, -0.010732, -0.010468, -0.01    , -0.009987,
        -0.009988, -0.010003, -0.010032, -0.010075, -0.010132, -0.010203,
        -0.010288, -0.0105  , -0.010621, -0.010744, -0.010869, -0.010996,
        -0.011125, -0.011256, -0.011389, -0.011524, -0.0118  , -0.011918,
        -0.011992, -0.012022, -0.012008, -0.01195 , -0.011848, -0.011702,
        -0.011512, -0.011   , -0.010715, -0.01046 , -0.010235, -0.01004 ,
        -0.009875, -0.00974 , -0.009635, -0.00956 , -0.0095  , -0.009495,
        -0.00948 , -0.009455, -0.00942 , -0.009375, -0.00932 , -0.009255,
        -0.00918 , -0.009   , -0.008707, -0.00843 , -0.008173, -0.007936,
        -0.00772 , -0.007527, -0.007358, -0.007213, -0.007   , -0.006859,
        -0.006724, -0.006596, -0.006477, -0.006367, -0.006268, -0.006181,
        -0.006106, -0.006   , -0.005757, -0.005616, -0.005559, -0.005568,
        -0.005625, -0.005712, -0.005811, -0.005904, -0.006   , -0.006272,
        -0.006496, -0.006684, -0.006848, -0.007   , -0.007152, -0.007316,
        -0.007504
    ])
    # fmt:on

    xo.assert_allclose(aper.x_vertices, x_expected, atol=1e-6, rtol=0)
    xo.assert_allclose(aper.y_vertices, y_expected, atol=1e-6, rtol=0)
    xo.assert_allclose(aper2.x_vertices, x_expected, atol=1e-6, rtol=0)
    xo.assert_allclose(aper2.y_vertices, y_expected, atol=1e-6, rtol=0)


def test_limitrect_to_dict():
    lrect = xt.LimitRect(min_x=-0.03, max_x=0.03, min_y=0.0, max_y=0.09)
    lrect2 = xt.LimitRect.from_dict(lrect.to_dict())

    assert lrect2.min_x == lrect.min_x
    assert lrect2.max_x == lrect.max_x
    assert lrect2.min_y == lrect.min_y
    assert lrect2.max_y == lrect.max_y

    lrectdef = xt.LimitRect()

    assert lrectdef.min_x == -1e10
    assert lrectdef.max_x == 1e10
    assert lrectdef.min_y == -1e10
    assert lrectdef.max_y == 1e10
