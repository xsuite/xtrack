import xtrack as xt

# TODO:
# - Flatten places needs to know about the anchors
# - What happens if there are elements without `at` in the thin sandwich?

env = xt.Environment()

env.new('q1', 'Quadrupole', length=2.0)

components=[
    env.new('q1', 'Quadrupole', length=2.0, anchor='start', at=1.),
    env.new('q2', 'q1', anchor='start', at=10., from_='q1', from_anchor='end'),
    env.new('s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_='q2', from_anchor='start'),

    env.new('q3', 'Quadrupole', length=2.0, at=20.),
    env.new('q4', 'q3', anchor='start', at=0., from_='q3', from_anchor='end'),
    env.new('q5', 'q3'),

    # Sandwirch of markers expected [m2.0, m2, m2.1.0, m2.1]
    env.new('m2', 'Marker', at=0., from_='q2', from_anchor='start'),
    env.new('m2_0', 'Marker', at=0., from_='m2', from_anchor='start'),
    env.new('m2_1', 'Marker', at=0., from_='m2', from_anchor='end'),
    env.new('m2_1_0', 'Marker', at=0., from_='m2_1', from_anchor='start'),
    env.new('m2_1_1', 'Marker'),

    env.new('m1', 'Marker', at=0., from_='q1', from_anchor='start'),

    env.new('m4', 'Marker', at=0., from_='q4', from_anchor='start'),
    env.new('m3', 'Marker', at=0., from_='q3', from_anchor='end'),
]

line = env.new_line(components=components)

line.get_table().show(cols=['name', 's_start', 's_end', 's_center'])