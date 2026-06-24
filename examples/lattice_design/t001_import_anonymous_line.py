import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('m1', xt.Marker, at=2.0),
    env.new('m2', xt.Marker, at='m1')
])

tt = line.get_table()
assert np.all(tt.name == np.array(['||drift_1', 'm1', 'm2', '_end_point']))
xo.assert_allclose(tt.s, np.array([0.0, 2.0, 2.0, 2.0]), rtol=0, atol=1e-15)
