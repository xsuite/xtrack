import xtrack as xt

env = xt.Environment()
env.new('mb', 'Bend', length=0.5)

line = env.new_line(components=[
    'mb',
    'mb',
    env.new('ip1', 'Marker', at=10),
    'mb',
    'mb',
    (
        'mb',
        # env.new('mm', 'Quadrupole', length=0.5),
        env.new('ip2', 'Marker', at=20),
        'mb',
    ),
])


