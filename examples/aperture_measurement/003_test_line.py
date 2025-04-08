import xtrack as xt

env = xt.Environment()

line = env.new_line(components=[
    env.new('lrect', xt.LimitRect, min_x=-0.05, max_x=0.05, min_y=-0.02, max_y=0.02, at=3.),
    env.new('lellipse', xt.LimitEllipse, a=0.02, b=0.01),
    env.new('m', xt.Marker, at=5.),
    env.place('lellipse')
    ]
)

line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=26e9)

tt = line.get_aperture_table()
#
# Table: 7 rows, 10 cols
# name                    s    x_aper_low   x_aper_high x_aper_low_discrete x_aper_high_discrete ...
# drift_1                 0       -0.0505        0.0495                 nan                  nan
# lrect                   3       -0.0505        0.0495             -0.0505               0.0495
# lellipse::0             3       -0.0205        0.0195             -0.0205               0.0195
# drift_2                 3       -0.0205        0.0195                 nan                  nan
# m                       5       -0.0205        0.0195                 nan                  nan
# lellipse::1             5       -0.0205        0.0195             -0.0205               0.0195
# _end_point              5       -0.0205        0.0195                 nan                  nan

import numpy as np
import xobjects as xo
assert np.all(tt.name == ['drift_1', 'lrect', 'lellipse', 'drift_2', 'm', 'lellipse',
       '_end_point'])
xo.assert_allclose(tt.s, np.array([0., 3., 3., 3., 5., 5., 5.]), rtol=0, atol=1e-6)
xo.assert_allclose(tt.x_aper_low, np.array([-0.0505, -0.0505, -0.0205, -0.0205, -0.0205,
       -0.0205, -0.0205]), rtol=0, atol=1e-6)
xo.assert_allclose(tt.x_aper_high, np.array([0.0495, 0.0495, 0.0195, 0.0195, 0.0195,
       0.0195, 0.0195]), rtol=0, atol=1e-6)
xo.assert_allclose(tt.x_aper_low_discrete, np.array([np.nan, -0.0505, -0.0205, np.nan,
       np.nan, -0.0205, np.nan]), rtol=0, atol=1e-6)
xo.assert_allclose(tt.x_aper_high_discrete, np.array([np.nan, 0.0495, 0.0195, np.nan,
       np.nan, 0.0195, np.nan]), rtol=0, atol=1e-6)
xo.assert_allclose(tt.y_aper_low, np.array(
    [-0.0205, -0.0205, -0.0105, -0.0105, -0.0105, -0.0105, -0.0105],
    dtype=float), rtol=0, atol=1e-6)
xo.assert_allclose(tt.y_aper_high, np.array(
    [0.0195, 0.0195, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095],
    dtype=float), rtol=0, atol=1e-6)
xo.assert_allclose(tt.y_aper_low_discrete, np.array(
    [np.nan, -0.0205, -0.0105, np.nan, np.nan, -0.0105, np.nan],
    dtype=float), rtol=0, atol=1e-6)
xo.assert_allclose(tt.y_aper_high_discrete, np.array(
    [np.nan, 0.0195, 0.0095, np.nan, np.nan, 0.0095, np.nan],
    dtype=float), rtol=0, atol=1e-6)