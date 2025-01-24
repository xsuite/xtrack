import xtrack as xt

env = xt.Environment()

env.new('q1', 'Quadrupole', length=1)
l1 = env.new_line(components=['q1', 'q1'])
l2 = env.new_line(components=[
    env.place(l1, at=10, from_='m'),
    env.new('m', 'Marker', at=0),
])