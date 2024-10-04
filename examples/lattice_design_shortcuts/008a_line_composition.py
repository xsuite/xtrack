import xtrack as xt
import numpy as np

env = xt.Environment()
env['pi'] = np.pi

env['l_bend'] = 8.
env['l_quad'] = 1.
env['l_cell'] = 20.
env['n_bends'] = 12.

env['h_bend']= 'pi / n_bends / l_bend'

env.new('mq', xt.Quadrupole, length='l_quad')
env.new('mb', xt.Bend, length='l_bend', h='h_bend', k0='h_bend')

env.new('mqf', 'mq', k1=0.1)
env.new('mqd', 'mq', k1=-0.1)

arc_cell = env.new_line(components=[
    env.place('mqf'), # At start cell
    env.place('mqd', at='l_cell/2', from_='mqf'), # At mid cell
    env.place('mb', at='l_cell/4'),
    env.place('mb', at='3*l_cell/4')])