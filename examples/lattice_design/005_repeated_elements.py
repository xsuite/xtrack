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

l1 = env.new_line(name='l1', components=[
    'mb',
    'mb',
    env.new('mid', 'Marker'),
    'mb',
    'mb',
])

env['s.l1'] = 10
l_twol1 = env.new_line(components=[
    env.new('ip', 'Marker', at=20),
    env.place('l1', at='s.l1', from_='ip'),
    env.place(l1, at=-env.ref['s.l1'], from_='ip'),
])
tt_twol1 = l_twol1.get_table()
assert np.all(tt_twol1.name == np.array(
    ['drift_3', 'mb', 'mb', 'mid', 'mb', 'mb', 'drift_4', 'ip',
       'drift_5', 'mb', 'mb', 'mid', 'mb', 'mb', '_end_point']))
assert np.all(tt_twol1.s == np.array(
    [ 0. ,  9. ,  9.5, 10. , 10. , 10.5, 11. , 20. , 20. , 29. , 29.5,
       30. , 30. , 30.5, 31. ]))

l_mult = env.new_line(name='l_from_list', components=[
    2 * l1,
    2 * ['mb']
])
tt_mult = l_mult.get_table()
assert np.all(tt_mult.name == np.array(['mb', 'mb', 'mid', 'mb', 'mb', 'mb', 'mb', 'mid', 'mb', 'mb', 'mb',
       'mb', '_end_point']))
assert np.all(tt_mult.s == np.array(
    [0. , 0.5, 1. , 1. , 1.5, 2. , 2.5, 3. , 3. , 3.5, 4. , 4.5, 5. ]))