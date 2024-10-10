import xtrack as xt
import numpy as np

env = xt.Environment()
env['a'] = 2
env['b'] = '3 * a'

env.ref['b']._value # is 6.0
env.ref[np.str_('b')]._value # is 6.0

env.ref['b']._expr # is "(3.0 * vars['a'])"
env.ref[np.str_('b')]._expr # is "None"
