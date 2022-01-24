import numpy as np

from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

mad.input("""
    m_circle: marker, apertype="circle", aperture={.2};
    m_ellipse: marker, apertype="ellipse", aperture={.2, .1};
    m_rectangle: marker, apertype="rectangle", aperture={.07, .05};
    m_rectellipse: marker, apertype="rectellipse", aperture={.2, .4, .25, .45};
    m_racetrack: marker, apertype="racetrack", aperture={.6,.4,.2,.1};
    m_octagon: marker, apertype="octagon", aperture={.4, .5, 0.5, 1.};
    beam;
    ss: sequence,l=1;
        m_circle, at=0;
        m_ellipse, at=0.01;
        m_rectangle, at=0.02;
        m_rectellipse, at=0.03;
        m_racetrack, at=0.04;
        m_octagon, at=0.05;
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
assert np.isclose(octag._xobject.y_vertices[0], 0.4*np.tan(0.5), atol=1e-10, rtol=0)
assert octag._xobject.y_vertices[1] == 0.5
assert np.isclose(octag._xobject.x_vertices[1], 0.5/np.tan(1.), atol=1e-10, rtol=0)

assert octag._xobject.y_vertices[2] == 0.5
assert np.isclose(octag._xobject.x_vertices[2], -0.5/np.tan(1.), atol=1e-10, rtol=0)
assert octag._xobject.x_vertices[3] == -0.4
assert np.isclose(octag._xobject.y_vertices[3], 0.4*np.tan(0.5), atol=1e-10, rtol=0)


assert octag._xobject.x_vertices[4] == -0.4
assert np.isclose(octag._xobject.y_vertices[4], -0.4*np.tan(0.5), atol=1e-10, rtol=0)
assert octag._xobject.y_vertices[5] == -0.5
assert np.isclose(octag._xobject.x_vertices[5], -0.5/np.tan(1.), atol=1e-10, rtol=0)


assert octag._xobject.y_vertices[6] == -0.5
assert np.isclose(octag._xobject.x_vertices[6], 0.5/np.tan(1.), atol=1e-10, rtol=0)
assert octag._xobject.x_vertices[7] == 0.4
assert np.isclose(octag._xobject.y_vertices[7], -0.4*np.tan(0.5), atol=1e-10, rtol=0)
