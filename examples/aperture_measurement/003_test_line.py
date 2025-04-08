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
