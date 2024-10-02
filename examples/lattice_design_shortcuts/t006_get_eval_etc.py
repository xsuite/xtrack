import xtrack as xt
import numpy as np
import xdeps as xd

to_test = 'line'
to_test = 'env'

env = xt.Environment()

env.vars({
    'k.1': 1.,
    'a': 2.,
    'b': '2 * a + k.1',
})

line = env.new_line([
    env.new('bb', xt.Bend, k0='2 * b', length=3+env.vars['a'] + env.vars['b'],
        h=5.)
])

# Line/Env methods (get, set, eval, get_expr, new_expr, info)
assert line.get('b') == 2 * 2 + 1
assert line.get('bb') is env.element_dict['bb']

assert str(line.get_expr('b')) == "((2.0 * vars['a']) + vars['k.1'])"

assert line.eval('3*a - sqrt(k.1)') == 5

ne = line.new_expr('sqrt(3*a + 3)')
assert xd.refs.is_ref(ne)
assert str(ne) == "f.sqrt(((3.0 * vars['a']) + 3.0))"

line.info('bb') # Check that it works
line.info('b')
line.info('a')








# Env/Line containers (env[...], env.ref[...]


# Vars methods (get, set, eval, get_expr, new_expr, info, get_table

# Vars container vars[...]


# View methods get_expr, get_value, get_info, get_table (for now)