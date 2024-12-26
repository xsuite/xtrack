import xtrack as xt

env = xt.Environment()

env.new('q1', 'Quadrupole', length=2.0)
components=[
    env.new('q1', 'Quadrupole', length=2.0, anchor='start', at=1.),
    env.new('q2', 'q1', anchor='start', at=10., from_='end@q1'),
    env.new('s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_='start@q2'),

    env.new('q3', 'Quadrupole', length=2.0, at=20.),
    env.new('q4', 'q3', anchor='start', at='end@q3'),
    env.new('q5', 'q3'),

    # Sandwitch of markers expected [m2.0, m2, m2.1.0, m2.1]
    env.new('m2', 'Marker', at='start@q2'),
    env.new('m2_0', 'Marker', at='start@m2'),
    env.new('m2_1', 'Marker', at='end@m2'),
    env.new('m2_1_0', 'Marker', at='start@m2_1'),

    env.new('m1', 'Marker', at='start@q1'),

    env.new('m4', 'Marker', at='start@q4'),
    env.new('m3', 'Marker', at='end@q3'),
]
line = env.new_line(components=components)

line.get_table().show(cols=['name', 's_start', 's_end', 's_center'])