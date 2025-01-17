import xtrack as xt
import numpy as np

env = xt.Environment()

line = env.new_line(
    name='l1',
    components=[
        env.new('b1', 'Bend', angle=np.pi/2, length=3.),
        env.new('d1', 'Drift', length=1.0),\
        env.new('b2', 'b1'),
        env.new('d2', 'd1'),
        env.new('b3', 'b1'),
        env.new('d3', 'd1'),
        env.new('b4', 'b1'),
        env.new('d4', 'd1'),
    ])

sv = line.survey()

breakpoint()
sv.plot()
