import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=10.0),
        env.new('qr', 'Quadrupole', length=2.0, at=30),
        env.new('end', 'Marker', at=50.),
    ])

env.new('s1', 'Sextupole', length=0.1)
env.new('s2', 's1')
env.new('s3', 's1')

line.insert(['s1', 's2', 's3'], anchor='start', at=1, from_='end@q0')

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'ql', 'drift_2', 'q0', 'drift_3..0', 's1', 's2', 's3',
       'drift_3..4', 'qr', 'drift_4', 'end', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 4.5 , 10.  , 15.  , 20.  , 21.5 , 22.05, 22.15, 22.25, 25.65,
       30.  , 40.5 , 50.  , 50.  ]),
    rtol=0., atol=1e-14)