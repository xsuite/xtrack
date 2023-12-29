# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from cpymad.madx import Madx
import xtrack as xt

import numpy as np

mad = Madx()

mad.input("""
    m_circle: marker, apertype="circle", aperture={.02};
    m_ellipse: marker, apertype="ellipse", aperture={.02, .01};
    m_rectangle: marker, apertype="rectangle", aperture={.07, .05};
    m_rectellipse: marker, apertype="rectellipse", aperture={.02, .04, .025, .045};
    m_racetrack: marker, apertype="racetrack", aperture={.06,.04,.02,.01};
    m_octagon: marker, apertype="octagon", aperture={.04, .05, 0.5, 1.1};
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


part_gen_range = 0.1
n_part=100000
part = xt.Particles(
        p0c=6500e9,
        x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
        px = np.zeros(n_part),
        y=np.random.uniform(-part_gen_range, part_gen_range, n_part),
        py = np.zeros(n_part),
        )

import matplotlib.pyplot as plt
plt.close('all')
i_plot = 1
for ee, nn in zip(line.elements, line.element_names):
    if ee.__class__.__name__.startswith('Limit'):
        part.state[:] = 1
        ee.track(part)

        plt.figure(i_plot)
        plt.plot(part.x[part.state<=0], part.y[part.state<=0], '.', color='red')
        plt.plot(part.x[part.state>0], part.y[part.state>0], '.', color='green')

        plt.title(nn)

        i_plot += 1

plt.show()