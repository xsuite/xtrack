# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_rect_ellipse(test_context):
    np2ctx = test_context.nparray_to_context_array
    ctx2np = test_context.nparray_from_context_array

    aper_rect_ellipse = xt.LimitRectEllipse(_context=test_context,
            max_x=23e-3, max_y=18e-3, a=23e-2, b=23e-2)
    aper_ellipse = xt.LimitRectEllipse(_context=test_context,
                                       a=23e-2, b=23e-2)
    aper_rect = xt.LimitRect(_context=test_context,
                             max_x=23e-3, min_x=-23e-3,
                             max_y=18e-3, min_y=-18e-3)

    XX, YY = np.meshgrid(np.linspace(-30e-3, 30e-3, 100),
                         np.linspace(-30e-3, 30e-3, 100))
    x_part = XX.flatten()
    y_part = XX.flatten()
    part_re = xp.Particles(_context=test_context,
                           x=x_part, y=y_part)
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
    part_gen_range = 0.11
    n_part=100000

    aper = xt.LimitRacetrack(_context=test_context,
                             min_x=-5e-2, max_x=10e-2,
                             min_y=-2e-2, max_y=4e-2,
                             a=2e-2, b=1e-2)

    xy_out = np.array([
        [-4.8e-2, 3.7e-2],
        [9.6e-2, 3.7e-2],
        [-4.5e-2, -1.8e-2],
        [9.8e-2, -1.8e-2],
        ])

    xy_in = np.array([
        [-4.2e-2, 3.3e-2],
        [9.4e-2, 3.6e-2],
        [-3.8e-2, -1.8e-2],
        [9.2e-2, -1.8e-2],
        ])

    xy_all = np.concatenate([xy_out, xy_in], axis=0)

    particles = xp.Particles(_context=test_context,
            p0c=6500e9,
            x=xy_all[:, 0],
            y=xy_all[:, 1])

    aper.track(particles)

    part_state = test_context.nparray_from_context_array(particles.state)
    part_id = test_context.nparray_from_context_array(particles.particle_id)

    assert np.all(part_state[part_id<4] == 0)
    assert np.all(part_state[part_id>=4] == 1)


@for_all_test_contexts
def test_aperture_polygon(test_context):
    np2ctx = test_context.nparray_to_context_array
    ctx2np = test_context.nparray_from_context_array

    x_vertices=np.array([1.5, 0.2, -1, -1,  1])*1e-2
    y_vertices=np.array([1.3, 0.5,  1, -1, -1])*1e-2

    aper = xt.LimitPolygon(
                    _context=test_context,
                    x_vertices=np2ctx(x_vertices),
                    y_vertices=np2ctx(y_vertices))

    # Try some particles inside
    parttest = xp.Particles(
                    _context=test_context,
                    p0c=6500e9,
                    x=x_vertices*0.99,
                    y=y_vertices*0.99)
    aper.track(parttest)
    assert np.allclose(ctx2np(parttest.state), 1)

    # Try some particles outside
    parttest = xp.Particles(
                    _context=test_context,
                    p0c=6500e9,
                    x=x_vertices*1.01,
                    y=y_vertices*1.01)
    aper.track(parttest)
    assert np.allclose(ctx2np(parttest.state), 0)


def test_mad_import():

    from cpymad.madx import Madx

    mad = Madx()

    mad.input("""
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

        use,sequence=ss;
        twiss,betx=1,bety=1;
        """
        )

    line = xt.Line.from_madx_sequence(mad.sequence.ss, install_apertures=True)

    apertures = [ee for ee in line.elements if ee.__class__.__name__.startswith('Limit')]

    circ = apertures[0]
    assert circ.__class__.__name__ == 'LimitEllipse'
    assert np.isclose(circ.a_squ, .2**2, atol=1e-13, rtol=0)
    assert np.isclose(circ.b_squ, .2**2, atol=1e-13, rtol=0)

    ellip = apertures[1]
    assert ellip.__class__.__name__ == 'LimitEllipse'
    assert np.isclose(ellip.a_squ, .2**2, atol=1e-13, rtol=0)
    assert np.isclose(ellip.b_squ, .1**2, atol=1e-13, rtol=0)

    rect = apertures[2]
    assert rect.__class__.__name__ == 'LimitRect'
    assert rect.min_x == -.07
    assert rect.max_x == +.07
    assert rect.min_y == -.05
    assert rect.max_y == +.05

    rectellip = apertures[3]
    assert rectellip.max_x == .2
    assert rectellip.max_y == .4
    assert np.isclose(rectellip.a_squ, .25**2, atol=1e-13, rtol=0)
    assert np.isclose(rectellip.b_squ, .45**2, atol=1e-13, rtol=0)

    racetr = apertures[4]
    assert racetr.__class__.__name__ == 'LimitRacetrack'
    assert racetr.min_x == -.6
    assert racetr.max_x == +.6
    assert racetr.min_y == -.4
    assert racetr.max_y == +.4
    assert racetr.a == .2
    assert racetr.b == .1

    octag = apertures[5]
    assert octag.__class__.__name__ == 'LimitPolygon'
    assert octag._xobject.x_vertices[0] == 0.4
    assert np.isclose(octag._xobject.y_vertices[0], 0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[1] == 0.5
    assert np.isclose(octag._xobject.x_vertices[1], 0.5/np.tan(1.), atol=1e-14, rtol=0)

    assert octag._xobject.y_vertices[2] == 0.5
    assert np.isclose(octag._xobject.x_vertices[2], -0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[3] == -0.4
    assert np.isclose(octag._xobject.y_vertices[3], 0.4*np.tan(0.5), atol=1e-14, rtol=0)


    assert octag._xobject.x_vertices[4] == -0.4
    assert np.isclose(octag._xobject.y_vertices[4], -0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[5] == -0.5
    assert np.isclose(octag._xobject.x_vertices[5], -0.5/np.tan(1.), atol=1e-14, rtol=0)


    assert octag._xobject.y_vertices[6] == -0.5
    assert np.isclose(octag._xobject.x_vertices[6], 0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[7] == 0.4
    assert np.isclose(octag._xobject.y_vertices[7], -0.4*np.tan(0.5), atol=1e-14, rtol=0)

    polyg = apertures[6]
    assert polyg.__class__.__name__ == 'LimitPolygon'
    assert len(polyg._xobject.x_vertices) == 3
    assert len(polyg._xobject.y_vertices) == 3
    assert polyg._xobject.x_vertices[0] == -8.8e-2
    assert polyg._xobject.y_vertices[0] == 0
    assert polyg._xobject.x_vertices[1] == 5.8e-2
    assert polyg._xobject.y_vertices[1] == -3.5e-2


@for_all_test_contexts
def test_longitudinal_rect(test_context):
    np2ctx = test_context.nparray_to_context_array
    ctx2np = test_context.nparray_from_context_array

    aper_rect_longitudinal = xt.LongitudinalLimitRect(_context=test_context,
            min_zeta=-10E-3,max_zeta=20E-3, min_pzeta=-1E-3, max_pzeta = 4E-3)

    coords = np.array([
        [-9E-4, 0.0],
        [15E-3, 0.0],
        [0.0, -8E-4],
        [0.0, 2E-3],
        ])
    particles = xp.Particles(_context=test_context,
                           zeta=coords[:,0], pzeta=coords[:,1])

    aper_rect_longitudinal.track(particles)
    particles.move(_context=xo.ContextCpu())
    assert np.all(particles.state == 1)

    coords = np.array([
        [-11E-3, 0.0],
        [22E-3, 0.0],
        [0.0, -2E-3],
        [0.0, 6E-3],
        ])
    particles = xp.Particles(_context=test_context,
                           zeta=coords[:,0], pzeta=coords[:,1])

    aper_rect_longitudinal.track(particles)
    particles.move(_context=xo.ContextCpu())
    assert np.all(particles.state == -2)
