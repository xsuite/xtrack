import xtrack as xt
import xobjects as xo

env = xt.Environment()

env.vars({
    'k.1': 1.,
    'a': 2.,
    'b': '2 * a + k.1',
})

assert env.vv['b'] == 2 * 2 + 1

env.vars['a'] = env.vars['k.1']
assert env.vv['b'] == 2 * 1 + 1

env.vars(a=3.)
env.vars({'k.1': 'a'})
assert env.vv['k.1'] == 3.
assert env.vv['b'] == 2 * 3 + 3.

env.vars['k.1'] = 2 * env.vars['a'] + 5
assert env.vv['k.1'] == 2 * 3 + 5
assert env.vv['b'] == 2 * 3 + 2 * 3 + 5


env.vars({
    'a': 4.,
    'b': '2 * a + 5',
})

env.new('bb', xt.Bend, k0='2 * b', length=3+env.vars['a'] + env.vars['b'],
        h=5.)
assert env['bb'].k0 == 2 * (2 * 4 + 5)
assert env['bb'].length == 3 + 4 + 2 * 4 + 5
assert env['bb'].h == 5.