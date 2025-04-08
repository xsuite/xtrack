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