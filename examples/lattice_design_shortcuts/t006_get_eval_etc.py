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
        h=5., ksl=[0, '3*b']),
])

for ee in [line, env]:

    # Line/Env methods (get, set, eval, get_expr, new_expr, info)
    assert ee.get('b') == 2 * 2 + 1
    assert ee.get('bb') is env.element_dict['bb']

    assert str(ee.get_expr('b')) == "((2.0 * vars['a']) + vars['k.1'])"

    assert ee.eval('3*a - sqrt(k.1)') == 5

    ne = ee.new_expr('sqrt(3*a + 3)')
    assert xd.refs.is_ref(ne)
    assert str(ne) == "f.sqrt(((3.0 * vars['a']) + 3.0))"

    ee.info('bb') # Check that it works
    ee.info('b')
    ee.info('a')

    ee.set('c', '6*a')
    assert ee.get('c') == 6 * 2

    # Line/Env containers (env[...], env.ref[...]
    assert ee['b'] == 2 * 2 + 1
    assert type(ee['bb']).__name__ == 'View'
    assert ee['bb'].__class__.__name__ == 'Bend'

    # Vars methods (get, set, eval, get_expr, new_expr, info, get_table)
    assert ee.vars.get('b') == 2 * 2 + 1

    assert str(ee.vars.get_expr('b')) == "((2.0 * vars['a']) + vars['k.1'])"

    assert ee.vars.eval('3*a - sqrt(k.1)') == 5

    ne = ee.vars.new_expr('sqrt(3*a + 3)')
    assert xd.refs.is_ref(ne)
    assert str(ne) == "f.sqrt(((3.0 * vars['a']) + 3.0))"

    ee.vars.info('b')
    ee.vars.info('a')

    ee.vars.set('d', '7*a')
    assert ee.vars.get('d') == 7 * 2

    assert xd.refs.is_ref(ee.vars['b'])

    # View methods get_expr, get_value, get_info, get_table (for now)
    assert xd.refs.is_ref(ee['bb'].get_expr('k0'))
    assert str(ee['bb'].get_expr('k0')) == "(2.0 * vars['b'])"
    assert ee['bb'].get_expr('k0')._value == 2 * (2 * 2 + 1)
    assert ee['bb'].get_value('k0') == 2 * (2 * 2 + 1)

    tt = ee['bb'].get_table()
    assert tt['value', 'k0'] == 2 * (2 * 2 + 1)
    assert tt['expr', 'k0'] == "(2.0 * vars['b'])"
