import xtrack as xt

env = xt.Environment()
env.new('mb', 'Bend', length=0.5)
pp = env.place('mb')

line = env.new_line(components=[
    'mb',
    'mb',
    env.new('ip1', 'Marker', at=10),
    'mb',
    pp,
    (
        'mb',
        # env.new('mm', 'Quadrupole', length=0.5),
        env.new('ip2', 'Marker', at=20),
        'mb',
    ),
    pp
])


