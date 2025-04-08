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
    xo.assert_allclose(ctx2np(parttest.state), 1)

    # Try some particles outside
    parttest = xp.Particles(
                    _context=test_context,
                    p0c=6500e9,
                    x=x_vertices*1.01,
                    y=y_vertices*1.01)
    aper.track(parttest)
    xo.assert_allclose(ctx2np(parttest.state), 0)


def test_mad_import():

    from cpymad.madx import Madx

    mad = Madx(stdout=False)

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
    xo.assert_allclose(circ.a_squ, .2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(circ.b_squ, .2**2, atol=1e-13, rtol=0)

    ellip = apertures[1]
    assert ellip.__class__.__name__ == 'LimitEllipse'
    xo.assert_allclose(ellip.a_squ, .2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(ellip.b_squ, .1**2, atol=1e-13, rtol=0)

    rect = apertures[2]
    assert rect.__class__.__name__ == 'LimitRect'
    assert rect.min_x == -.07
    assert rect.max_x == +.07
    assert rect.min_y == -.05
    assert rect.max_y == +.05

    rectellip = apertures[3]
    assert rectellip.max_x == .2
    assert rectellip.max_y == .4
    xo.assert_allclose(rectellip.a_squ, .25**2, atol=1e-13, rtol=0)
    xo.assert_allclose(rectellip.b_squ, .45**2, atol=1e-13, rtol=0)

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
    xo.assert_allclose(octag._xobject.y_vertices[0], 0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[1] == 0.5
    xo.assert_allclose(octag._xobject.x_vertices[1], 0.5/np.tan(1.), atol=1e-14, rtol=0)

    assert octag._xobject.y_vertices[2] == 0.5
    xo.assert_allclose(octag._xobject.x_vertices[2], -0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[3] == -0.4
    xo.assert_allclose(octag._xobject.y_vertices[3], 0.4*np.tan(0.5), atol=1e-14, rtol=0)


    assert octag._xobject.x_vertices[4] == -0.4
    xo.assert_allclose(octag._xobject.y_vertices[4], -0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[5] == -0.5
    xo.assert_allclose(octag._xobject.x_vertices[5], -0.5/np.tan(1.), atol=1e-14, rtol=0)


    assert octag._xobject.y_vertices[6] == -0.5
    xo.assert_allclose(octag._xobject.x_vertices[6], 0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[7] == 0.4
    xo.assert_allclose(octag._xobject.y_vertices[7], -0.4*np.tan(0.5), atol=1e-14, rtol=0)

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

@for_all_test_contexts
def test_aper_tilt(test_context):

    n_part=300000

    particles = xt.Particles(_context=test_context,
            p0c=6500e9,
            x=np.random.uniform(-0.25, 0.25, n_part),
            px = np.zeros(n_part),
            y=np.random.uniform(0, 0.1, n_part),
            py = np.zeros(n_part))

    tilt_deg = 10.
    aper = xt.LimitRect(_context=test_context,
                        min_x=-.1,
                        max_x=.1,
                        min_y=-0.001,
                        max_y=0.001,
                        shift_x=0.08,
                        shift_y=0.04,
                        rot_s_rad=np.deg2rad(tilt_deg))

    aper.track(particles)

    part_id = test_context.nparray_from_context_array(particles.particle_id)
    part_state = test_context.nparray_from_context_array(particles.state)
    part_x = test_context.nparray_from_context_array(particles.x)
    part_y = test_context.nparray_from_context_array(particles.y)

    x_alive = part_x[part_state>0]
    y_alive = part_y[part_state>0]

    assert_allclose = np.testing.assert_allclose
    assert_allclose(np.mean(x_alive), 0.08, rtol=5e-2, atol=0)
    assert_allclose(np.mean(y_alive), 0.04, rtol=5e-2, atol=0)
    slope = np.polyfit(x_alive, y_alive, 1)[0]
    assert_allclose(slope, np.tan(np.deg2rad(tilt_deg)), rtol=5e-2, atol=0)



def test_aperture_svg_path():

    svg= {"path": """M4 8 10 1 13 0 12 3 5 9C6 10 6 11 7 10 7 11 8 12 7 12A1.42 1.42 0 016 13 5 5 0 004 10Q3.5 9.9 3.5 10.5T2 11.8 1.2 11 2.5 9.5 3 9A5 5 90 000 7 1.42 1.42 0 011 6C1 5 2 6 3 6 2 7 3 7 4 8"""}

    aper = xt.LimitPolygon(
                    svg=svg)

    assert aper.copy().svg==aper.svg
    aper2=xt.LimitPolygon.from_dict(aper.to_dict())
    assert aper2.svg==aper.svg

    x_expected = np.array([ 3.41600000e-03,  3.15400000e-03,  2.92800000e-03,  2.75000000e-03,
        2.63200000e-03,  2.58600000e-03,  2.62400000e-03,  2.75800000e-03,
        3.00000000e-03,  2.40800000e-03,  2.12700000e-03,  1.86400000e-03,
        1.62500000e-03,  1.41600000e-03,  1.24300000e-03,  1.11200000e-03,
        1.02900000e-03,  1.00000000e-03,  7.24296143e-04,  5.96239745e-04,
        4.76593043e-04,  3.66655177e-04,  2.67619865e-04,  1.80562445e-04,
        1.06428199e-04,  4.60220848e-05, -4.44089210e-19,  7.05663559e-04,
        1.04481682e-03,  1.37237342e-03,  1.68655149e-03,  1.98564190e-03,
        2.26801764e-03,  2.53214260e-03,  2.77657996e-03,  3.00000000e-03,
        3.01200000e-03,  2.99700000e-03,  2.96800000e-03,  2.92500000e-03,
        2.86800000e-03,  2.79700000e-03,  2.71200000e-03,  2.61300000e-03,
        2.50000000e-03,  2.25600000e-03,  2.13100000e-03,  2.00400000e-03,
        1.87500000e-03,  1.74400000e-03,  1.61100000e-03,  1.47600000e-03,
        1.33900000e-03,  1.20000000e-03,  1.00800000e-03,  9.78000000e-04,
        9.92000000e-04,  1.05000000e-03,  1.15200000e-03,  1.29800000e-03,
        1.48800000e-03,  1.72200000e-03,  2.00000000e-03,  2.54000000e-03,
        2.76500000e-03,  2.96000000e-03,  3.12500000e-03,  3.26000000e-03,
        3.36500000e-03,  3.44000000e-03,  3.48500000e-03,  3.50000000e-03,
        3.52000000e-03,  3.54500000e-03,  3.58000000e-03,  3.62500000e-03,
        3.68000000e-03,  3.74500000e-03,  3.82000000e-03,  3.90500000e-03,
        4.00000000e-03,  4.56952827e-03,  4.82689224e-03,  5.06408867e-03,
        5.27982723e-03,  5.47293432e-03,  5.64235946e-03,  5.78718097e-03,
        5.90661105e-03,  6.00000000e-03,  6.27570386e-03,  6.40376026e-03,
        6.52340696e-03,  6.63334482e-03,  6.73238014e-03,  6.81943755e-03,
        6.89357180e-03,  6.95397792e-03,  7.00000000e-03,  7.38400000e-03,
        7.44100000e-03,  7.43200000e-03,  7.37500000e-03,  7.28800000e-03,
        7.18900000e-03,  7.09600000e-03,  7.02700000e-03,  7.00000000e-03,
        6.50400000e-03,  6.31600000e-03,  6.15200000e-03,  6.00000000e-03,
        5.84800000e-03,  5.68400000e-03,  5.49600000e-03,  5.27200000e-03,
        5.00000000e-03,  1.20000000e-02,  1.30000000e-02,  1.00000000e-02,
        4.00000000e-03])
    y_expected = np.array([-0.007504  , -0.007316  , -0.007152  , -0.007     , -0.006848  ,
       -0.006684  , -0.006496  , -0.006272  , -0.006     , -0.005904  ,
       -0.005811  , -0.005712  , -0.005625  , -0.005568  , -0.005559  ,
       -0.005616  , -0.005757  , -0.006     , -0.00610643, -0.00618056,
       -0.00626762, -0.00636666, -0.00647659, -0.00659624, -0.0067243 ,
       -0.00685937, -0.007     , -0.00721282, -0.00735764, -0.00752707,
       -0.00772017, -0.00793591, -0.00817311, -0.00843047, -0.0087066 ,
       -0.009     , -0.00918   , -0.009255  , -0.00932   , -0.009375  ,
       -0.00942   , -0.009455  , -0.00948   , -0.009495  , -0.0095    ,
       -0.00956   , -0.009635  , -0.00974   , -0.009875  , -0.01004   ,
       -0.010235  , -0.01046   , -0.010715  , -0.011     , -0.011512  ,
       -0.011702  , -0.011848  , -0.01195   , -0.012008  , -0.012022  ,
       -0.011992  , -0.011918  , -0.0118    , -0.011524  , -0.011389  ,
       -0.011256  , -0.011125  , -0.010996  , -0.010869  , -0.010744  ,
       -0.010621  , -0.0105    , -0.010288  , -0.010203  , -0.010132  ,
       -0.010075  , -0.010032  , -0.010003  , -0.009988  , -0.009987  ,
       -0.01      , -0.01046786, -0.01073198, -0.01101436, -0.01131345,
       -0.01162763, -0.01195518, -0.01229434, -0.01264324, -0.013     ,
       -0.01289357, -0.01281944, -0.01273238, -0.01263334, -0.01252341,
       -0.01240376, -0.0122757 , -0.01214063, -0.012     , -0.011888  ,
       -0.011757  , -0.011584  , -0.011375  , -0.011136  , -0.010873  ,
       -0.010592  , -0.010299  , -0.01      , -0.010376  , -0.010414  ,
       -0.010368  , -0.01025   , -0.010072  , -0.009846  , -0.009584  ,
       -0.009298  , -0.009     , -0.003     , -0.        , -0.001     ,
       -0.008     ])

    xo.assert_allclose(aper.x_vertices, x_expected, atol=1e-6, rtol=0)
    xo.assert_allclose(aper.y_vertices, y_expected, atol=1e-6, rtol=0)
    xo.assert_allclose(aper2.x_vertices, x_expected, atol=1e-6, rtol=0)
    xo.assert_allclose(aper2.y_vertices, y_expected, atol=1e-6, rtol=0)