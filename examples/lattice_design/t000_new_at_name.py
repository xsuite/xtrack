import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('m1', xt.Marker, at=2.0),
    env.new('m2', xt.Marker, at='m1')
])

env2 = xt.Environment()
line2 = env2.import_line(line)

tt = line.get_table()
tt2 = line2.get_table()

assert np.all(tt.name == tt2.name)
xo.assert_allclose(tt.s_center, tt2.s_center)

assert env is not env2
assert line.env is env
assert line2.env is env2
