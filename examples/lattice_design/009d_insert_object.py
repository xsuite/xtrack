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

class MyElement:
    def __init__(self, myparameter):
        self.myparameter = myparameter

    def track(self, particles):
        particles.px += self.myparameter

myelem = MyElement(0.1)

line.insert(
    env.place('myname', myelem, at='qr@end'))

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'ql', 'drift_2', 'q0', 'drift_3', 'qr', 'myname',
     'drift_4', 'end', '_end_point']))
assert np.all(tt.element_type == np.array(
    ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
       'Quadrupole', 'MyElement', 'Drift', 'Marker', '']))
xo.assert_allclose(tt.s, np.array(
    [ 0.,  9., 11., 19., 21., 29., 31., 31., 50., 50.]),
    rtol=0., atol=1e-14)