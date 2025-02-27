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

class MyElement:
    def __init__(self, myparameter):
        self.myparameter = myparameter

    def track(self, particles):
        particles.px += self.myparameter

myelem = MyElement(0.1)

line = line0.copy()
line.append('myname', myelem)

tt = line.get_table()
tt.show(cols=['name', 'element_type', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0::0', 'drift_2', 'ql', 'drift_3', 'q0::0', 'drift_4',
       'qr', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0::1', 'b0::1', 'drift_6',
       'end', 'myname', '_end_point']))

assert np.all(tt.element_type == np.array(
    ['Drift', 'Bend', 'Drift', 'Quadrupole', 'Drift', 'Quadrupole',
       'Drift', 'Quadrupole', 'Drift', 'Marker', 'Marker', 'Marker',
       'Quadrupole', 'Bend', 'Drift', 'Marker', 'MyElement', '']))

xo.assert_allclose(tt.s_center, np.array(
    [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
       40. , 41. , 42.5, 46.5, 50. , 50. , 50. ]),
    rtol=0., atol=1e-14)

line = line0.copy()
line.env.new('qnew1', 'Quadrupole', length=2.0)
line.env.new('qnew2', 'Quadrupole', length=2.0)
line.append(['qnew1', 'qnew2'])

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0::0', 'drift_2', 'ql', 'drift_3', 'q0::0', 'drift_4',
       'qr', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0::1', 'b0::1', 'drift_6',
       'end', 'qnew1', 'qnew2', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
       40. , 41. , 42.5, 46.5, 50. , 51. , 53. , 54. ]),
    rtol=0., atol=1e-14)

line = line0.copy()
line.env.new('qnew1', 'Quadrupole', length=2.0)
line.env.new('qnew2', 'Quadrupole', length=2.0)

l2 = line.env.new_line(components=['qnew1', 'qnew2', 'ql'])
line.append(l2)

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0::0', 'drift_2', 'ql::0', 'drift_3', 'q0::0', 'drift_4',
       'qr', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0::1', 'b0::1', 'drift_6',
       'end', 'qnew1', 'qnew2', 'ql::1', '_end_point']))

xo.assert_allclose(tt.s_center, np.array(
    [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
       40. , 41. , 42.5, 46.5, 50. , 51. , 53. , 55. , 56. ]),
    rtol=0., atol=1e-14)

line = line0.copy()
line.env.new('qnew1', 'Quadrupole', length=2.0)
line.env.new('qnew2', 'Quadrupole', length=2.0)

l2 = line.env.new_line(components=['qnew1', 'qnew2', 'ql'])
line.append([l2, 'qr'])

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0::0', 'drift_2', 'ql::0', 'drift_3', 'q0::0', 'drift_4',
       'qr::0', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0::1', 'b0::1', 'drift_6',
       'end', 'qnew1', 'qnew2', 'ql::1', 'qr::1', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
       40. , 41. , 42.5, 46.5, 50. , 51. , 53. , 55. , 57. , 58. ]),
    rtol=0., atol=1e-14)
