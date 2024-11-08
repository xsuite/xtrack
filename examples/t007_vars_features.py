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

# Test compact and to_dict
tt = env.vars.get_table()
assert tt['expr', 'b'] == '(3.0 * a)'
dd = tt.to_dict()
assert dd['b'] == '(3.0 * a)'
assert dd['a'] == 3.0

ee = xt.Environment()
ee.vars.update(dd)
assert ee['a'] == 3.0
assert ee['b'] == 9.0
assert ee.vars.get_table()['expr', 'b'] == '(3.0 * a)'

tt1 = env.vars.get_table(compact=False)
assert tt1['expr', 'b'] ==  "(3.0 * vars['a'])"
dd1 = tt1.to_dict(compact=False)
assert dd1['b'] == "(3.0 * vars['a'])"
assert dd1['a'] == 3.0

ee1 = xt.Environment()
ee1.vars.update(dd1)
assert ee1['a'] == 3.0
assert ee1['b'] == 9.0
assert ee1.vars.get_table(compact=False)['expr', 'b'] == "(3.0 * vars['a'])"
