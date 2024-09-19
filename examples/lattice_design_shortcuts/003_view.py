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

assert env.get('a') == 3
assert env.get('b') == 9
assert env.get('c') == 12

env.new('mb', 'Bend', k1='3*a', h='4*a', knl=[0, '5*a'])
assert isinstance(env['mb'].k1, float)
assert isinstance(env['mb'].h, float)
assert isinstance(env['mb'].knl[0], float)
assert env['mb'].k1 == 9
assert env['mb'].h == 12
assert env['mb'].knl[0] == 0
assert env['mb'].knl[1] == 15