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

env.new('q1', 'q0')
env.new('q2', 'q0')
env.new('q3', 'q0')

line.insert('q1', at=5.0)
line.insert('q2', at=15.0)
line.insert('q3', at=41.0)

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1..0', 'q1', 'drift_1..2', 'ql', 'drift_2..0', 'q2',
       'drift_2..2', 'q0', 'drift_3', 'qr', 'drift_4', 'mk1', 'q3', 'mk2',
       'drift_6', 'end', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2. ,  5. ,  7.5, 10. , 12.5, 15. , 17.5, 20. , 25. , 30. , 35.5,
       40. , 41. , 42. , 46. , 50. , 50. ]), rtol=0., atol=1e-14)