import xtrack as xt
import numpy as np

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
        env.new('ip2', 'Marker', at=20),
        'mb',
    ),
    pp
])

tt = line.get_table()
assert np.all(tt.name == np.array(['mb', 'mb', 'drift_1', 'ip1', 'mb',
                                   'mb', 'drift_2', 'mb', 'ip2', 'mb',
                                   'mb', '_end_point']))
assert np.all(tt.s == np.array([
    0. ,  0.5,  1. , 10. , 10. , 10.5, 11. , 19.5, 20. , 20. , 20.5, 21. ]))