import xtrack as xt

env = xt.Environment()

try:
    env['b'] = '3*a'
except KeyError:
    pass
else:
    raise ValueError('Variable a should not be present')

env.vars.default_to_zero = True
env['b'] = '3*a'

assert env['a'] == 0
assert env['b'] == 0

env['a'] = 3
assert env['b'] == 9