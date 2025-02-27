import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
        env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
        env.new('mk1', 'Marker', at=40),
        env.new('mk2', 'Marker', at=42),
        env.new('end', 'Marker', at=50.),
    ])

s_tol = 1e-10

tt0 = line.get_table()
tt0.show(cols=['name', 's_start', 's_end', 's_center'])

env.new('ss', 'Sextupole', length='0.1')
pp_ss = env.place('ss')

line.insert([
    env.place('q0', at=5.0),
    pp_ss,
    env.place('q0', at=15.0),
    pp_ss,
    env.place('q0', at=41.0),
    pp_ss,
])

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1..0', 'q0::0', 'ss::0', 'drift_1..3', 'ql', 'drift_2..0',
       'q0::1', 'ss::1', 'drift_2..3', 'q0::2', 'drift_3', 'qr',
       'drift_4', 'mk1', 'q0::3', 'mk2', 'ss::2', 'drift_6..1', 'end',
       '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2.  ,  5.  ,  6.05,  7.55, 10.  , 12.5 , 15.  , 16.05, 17.55,
      20.  , 25.  , 30.  , 35.5 , 40.  , 41.  , 42.  , 42.05, 46.05,
      50.  , 50.  ]), rtol=0., atol=1e-14)