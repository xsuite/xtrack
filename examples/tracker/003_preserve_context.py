import xtrack as xt
import xobjects as xo

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
line = env.lhcb1

ctx = xo.ContextCpu(omp_num_threads=4)
line.build_tracker(_context=ctx)

line.append('a', xt.Marker())
line.twiss4d()
assert line._context is ctx
