import xtrack as xt
import numpy as np

env = xt.Environment()

env.vars({
    'k.1': 1.,
    'a': 2.,
    'b': '2 * a + k.1',
})

line = env.new_line([])

ee = env # Test Environment
ee = line # Test Line

assert ee.vv['b'] == 2 * 2 + 1

ee.vars['a'] = ee.vars['k.1']
assert ee.vv['b'] == 2 * 1 + 1

ee.vars(a=3.)
ee.vars({'k.1': 'a'})
assert ee.vv['k.1'] == 3.
assert ee.vv['b'] == 2 * 3 + 3.

ee.vars['k.1'] = 2 * ee.vars['a'] + 5
assert ee.vv['k.1'] == 2 * 3 + 5
assert ee.vv['b'] == 2 * 3 + 2 * 3 + 5

ee.vars.set('a', 4.)
assert ee.vv['k.1'] == 2 * 4 + 5
assert ee.vv['b'] == 2 * 4 + 2 * 4 + 5

ee.vars.set('k.1', '2*a + 5')
assert ee.vv['k.1'] == 2 * 4 + 5
assert ee.vv['b'] == 2 * 4 + 2 * 4 + 5

ee.vars.set('k.1', 3 * ee.vars['a'] + 6)
assert ee.vv['k.1'] == 3 * 4 + 6
assert ee.vv['b'] == 2 * 4 + 3 * 4 + 6

ee.set('a', 0.)
assert ee.vv['k.1'] == 3 * 0 + 6
assert ee.vv['b'] == 2 * 0 + 3 * 0 + 6

ee.set('a', 2.)
ee.set('k.1', '2 * a + 5')
assert ee.vv['k.1'] == 2 * 2 + 5
assert ee.vv['b'] == 2 * 2 + 2 * 2 + 5

ee.set('k.1', 3 * ee.vars['a'] + 6)
assert ee.vv['k.1'] == 3 * 2 + 6
assert ee.vv['b'] == 2 * 2 + 3 * 2 + 6

assert hasattr(ee.ref['k.1'], '_value') # is a Ref

ee.ref['a'] = 0
assert ee.vv['k.1'] == 3 * 0 + 6
assert ee.vv['b'] == 2 * 0 + 3 * 0 + 6

ee.ref['a'] = 2
ee.ref['k.1'] = 2 * ee.ref['a'] + 5
assert ee.vv['k.1'] == 2 * 2 + 5
assert ee.vv['b'] == 2 * 2 + 2 * 2 + 5

ee.vars({
    'a': 4.,
    'b': '2 * a + 5',
    'k.1': '2 * a + 5',
})

#--------------------------------------------------

env.new('bb', xt.Bend, k0='2 * b', length=3+env.vars['a'] + env.vars['b'],
        h=5.)
assert env['bb'].k0 == 2 * (2 * 4 + 5)
assert env['bb'].length == 3 + 4 + 2 * 4 + 5
assert env['bb'].h == 5.

env.vars['a'] = 2.
assert env['bb'].k0 == 2 * (2 * 2 + 5)
assert env['bb'].length == 3 + 2 + 2 * 2 + 5
assert env['bb'].h == 5.

line = env.new_line([
    env.new('bb1', 'bb', length=3*env.vars['a'], at='2*a'),
    env.place('bb', at=10 * env.vars['a'], from_='bb1'),
])

assert line['bb1'] is not env['bb']
assert line['bb'] is env['bb']

a = env.vv['a']
assert line['bb1'].length == 3 * a
assert line['bb1'].k0 == 2 * (2 * a + 5)
assert line['bb1'].h == 5.

assert line['bb'].k0 == 2 * (2 * a + 5)
assert line['bb'].length == 3 + a + 2 * a + 5
assert line['bb'].h == 5.

tt = line.get_table(attr=True)
tt['s_center'] = tt['s'] + tt['length']/2

assert np.all(tt.name ==  np.array(['drift_1', 'bb1', 'drift_2', 'bb', '_end_point']))

assert tt['s_center', 'bb1'] == 2*a
assert tt['s_center', 'bb'] - tt['s_center', 'bb1'] == 10*a

old_a = a
line.vars['a'] = 3.
a = line.vv['a']
assert line['bb1'].length == 3 * a
assert line['bb1'].k0 == 2 * (2 * a + 5)
assert line['bb1'].h == 5.

assert line['bb'].k0 == 2 * (2 * a + 5)
assert line['bb'].length == 3 + a + 2 * a + 5
assert line['bb'].h == 5.

tt_new = line.get_table(attr=True)

# Drifts are not changed:
tt_new['length', 'drift_1'] == tt['length', 'drift_1']
tt_new['length', 'drift_2'] == tt['length', 'drift_2']

