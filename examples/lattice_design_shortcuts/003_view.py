import xtrack as xt

env = xt.Environment()

env['a']  = 3.
env['b']  = 3 * env['a']
env['c']  = '4 * a'

assert isinstance(env['a'], float)
assert isinstance(env['b'], float)
assert isinstance(env['c'], float)

assert env['a'] == 3
assert env['b'] == 9
assert env['c'] == 12

assert env.ref['a']._value == 3
assert env.ref['b']._value == 9
assert env.ref['c']._value == 12