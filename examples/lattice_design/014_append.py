import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment(
    particle_ref=xt.Particles(p0c=7000e9, x=1e-3, px=1e-3, y=1e-3, py=1e-3))

line0 = env.new_line(
    components=[
        env.new('b0', 'Bend', length=1.0, anchor='start', at=5.0),
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
        env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
        env.new('mk1', 'Marker', at=40),
        env.new('mk2', 'Marker'),
        env.new('mk3', 'Marker'),
        env.place('q0'),
        env.place('b0'),
        env.new('end', 'Marker', at=50.),
    ])

line = line0.copy()
line.env.new('qnew', 'Quadrupole', length=2.0)
line.append('qnew')

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0::0', 'drift_2', 'ql', 'drift_3', 'q0::0', 'drift_4',
       'qr', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0::1', 'b0::1', 'drift_6',
       'end', 'qnew', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
       40. , 41. , 42.5, 46.5, 50. , 51. , 52. ]),
    rtol=0., atol=1e-14)