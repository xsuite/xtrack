import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

env.new('mq', 'Quadrupole', length=1)

env.new_line(name='l1', length=20, components=[
    env.place('mq', at=10)])

env['l1'].get_table().cols['s_start s_center s_end']

tt = env['l1'].get_table()

assert np.all(tt.name == np.array(['drift_1', 'mq', 'drift_2', '_end_point']))
xo.assert_allclose(tt.s_center, np.array([ 4.75, 10.  , 15.25, 20.  ]),
                   rtol=0, atol=1e-10)