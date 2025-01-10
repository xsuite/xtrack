import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=10.0),
        env.new('qr', 'Quadrupole', length=2.0, at=30),
        env.new('end', 'Marker', at=50.),
    ])

ln_insert = env.new_line(
    components=[
        env.new('s1', 'Sextupole', length=0.1),
        env.new('s2', 's1', anchor='start', at=0.3, from_='s1@end'),
        env.new('s3', 's1', anchor='start', at=0.3, from_='s2@end')
    ])

line.insert(ln_insert, anchor='start', at=1, from_='q0@end')

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'ql', 'drift_2', 'q0', 'drift_3..0', 's1', 'drift_5',
       's2', 'drift_6', 's3', 'drift_3..6', 'qr', 'drift_4', 'end',
       '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 4.5 , 10.  , 15.  , 20.  , 21.5 , 22.05, 22.25, 22.45, 22.65,
       22.85, 25.95, 30.  , 40.5 , 50.  , 50.  ]),
    rtol=0., atol=1e-14)