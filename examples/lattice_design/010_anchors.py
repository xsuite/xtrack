import xtrack as xt

env = xt.Environment()

env.new('q1', 'Quadrupole', length=2.0)

components=[
    env.new('q1', 'Quadrupole', length=2.0, anchor='start', at=1.),
    env.new('q2', 'q1', anchor='start', at=10., from_='q1', from_anchor='end'),
    env.new('s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_='q2', from_anchor='start'),

    # Sandwitch of markers expected [m2.0, m2, m2.1.0, m2.1]
    env.new('m2', 'Marker', at=0., from_='q2', from_anchor='start'),
    env.new('m2.0', 'Marker', at=0., from_='m2', from_anchor='start'),
    env.new('m2.1', 'Marker', at=0., from_='m2', from_anchor='end'),
    env.new('m2.1.0', 'Marker', at=0., from_='m2.1', from_anchor='start'),

    env.new('m1', 'Marker', at=0., from_='q1', from_anchor='start'),
]

line = env.new_line(components=components)

line.get_table().show(cols=['name', 's_start', 's_end', 's_center'])