import xtrack as xt
import xdeps as xd

container_type = 'line'

env = xt.Environment()
line = xt.Line()

if container_type == 'env':
    ee = env
    eenv = env
elif container_type == 'line':
    ee = line
    eenv = line.env

ee['a']  = 3.
ee['b1']  = 3 * ee['a'] # done by value
ee['b2']  = 3 * ee.ref['a'] # done by reference
ee['c']  = '4 * a'

assert isinstance(ee['a'], float)
assert isinstance(ee['b1'], float)
assert isinstance(ee['b2'], float)
assert isinstance(ee['c'], float)

assert ee['a'] == 3
assert ee['b1'] == 9
assert ee['b2'] == 9
assert ee['c'] == 12

assert ee.ref['a']._value == 3
assert ee.ref['b1']._value == 9
assert ee.ref['b2']._value == 9
assert ee.ref['c']._value == 12

assert ee.get('a') == 3
assert ee.get('b1') == 9
assert ee.get('b2') == 9
assert ee.get('c') == 12

eenv.new('mb', 'Bend', extra={'description': 'Hello Riccarco'},
        k1='3*a', h=4*ee.ref['a'], knl=[0, '5*a', 6*ee.ref['a']])
assert isinstance(ee['mb'].k1, float)
assert isinstance(ee['mb'].h, float)
assert isinstance(ee['mb'].knl[0], float)
assert ee['mb'].k1 == 9
assert ee['mb'].h == 12
assert ee['mb'].knl[0] == 0
assert ee['mb'].knl[1] == 15
assert ee['mb'].knl[2] == 18

ee['a'] = 4
assert ee['a'] == 4
assert ee['b1'] == 9
assert ee['b2'] == 12
assert ee['c'] == 16
assert ee['mb'].k1 == 12
assert ee['mb'].h == 16
assert ee['mb'].knl[0] == 0
assert ee['mb'].knl[1] == 20
assert ee['mb'].knl[2] == 24

ee['mb'].k1 = '30*a'
ee['mb'].h = 40 * ee.ref['a']
ee['mb'].knl[1] = '50*a'
ee['mb'].knl[2] = 60 * ee.ref['a']
assert ee['mb'].k1 == 120
assert ee['mb'].h == 160
assert ee['mb'].knl[0] == 0
assert ee['mb'].knl[1] == 200
assert ee['mb'].knl[2] == 240

assert isinstance(ee['mb'].k1, float)
assert isinstance(ee['mb'].h, float)
assert isinstance(ee['mb'].knl[0], float)

assert ee.ref['mb'].k1._value == 120
assert ee.ref['mb'].h._value == 160
assert ee.ref['mb'].knl[0]._value == 0
assert ee.ref['mb'].knl[1]._value == 200
assert ee.ref['mb'].knl[2]._value == 240

assert ee.get('mb').k1 == 120
assert ee.get('mb').h == 160
assert ee.get('mb').knl[0] == 0
assert ee.get('mb').knl[1] == 200
assert ee.get('mb').knl[2] == 240

# Some interesting behavior
assert type(ee['mb']) is xd.madxutils.View
assert ee['mb'].__class__ is xt.Bend
assert type(ee.ref['mb']._value) is xt.Bend
assert type(ee.get('mb')) is xt.Bend
