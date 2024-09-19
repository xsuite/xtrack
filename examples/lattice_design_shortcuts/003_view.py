import xtrack as xt

env = xt.Environment()

env['a']  = 3.
env['b1']  = 3 * env['a'] # done by value
env['b2']  = 3 * env.ref['a'] # done by reference
env['c']  = '4 * a'

assert isinstance(env['a'], float)
assert isinstance(env['b1'], float)
assert isinstance(env['b2'], float)
assert isinstance(env['c'], float)

assert env['a'] == 3
assert env['b1'] == 9
assert env['b2'] == 9
assert env['c'] == 12

assert env.ref['a']._value == 3
assert env.ref['b1']._value == 9
assert env.ref['b2']._value == 9
assert env.ref['c']._value == 12

assert env.get('a') == 3
assert env.get('b1') == 9
assert env.get('b2') == 9
assert env.get('c') == 12

env.new('mb', 'Bend', k1='3*a', h=4*env.ref['a'], knl=[0, '5*a', 6*env.ref['a']])
assert isinstance(env['mb'].k1, float)
assert isinstance(env['mb'].h, float)
assert isinstance(env['mb'].knl[0], float)
assert env['mb'].k1 == 9
assert env['mb'].h == 12
assert env['mb'].knl[0] == 0
assert env['mb'].knl[1] == 15
assert env['mb'].knl[2] == 18

env['a'] = 4
assert env['a'] == 4
assert env['b1'] == 9
assert env['b2'] == 12
assert env['c'] == 16
assert env['mb'].k1 == 12
assert env['mb'].h == 16
assert env['mb'].knl[0] == 0
assert env['mb'].knl[1] == 20
assert env['mb'].knl[2] == 24

env['mb'].k1 = '30*a'
env['mb'].h = 40 * env.ref['a']
env['mb'].knl[1] = '50*a'
env['mb'].knl[2] = 60 * env.ref['a']
assert env['mb'].k1 == 120
assert env['mb'].h == 160
assert env['mb'].knl[0] == 0
assert env['mb'].knl[1] == 200
assert env['mb'].knl[2] == 240