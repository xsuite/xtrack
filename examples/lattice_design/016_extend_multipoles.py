import xtrack as xt
import xobjects as xo
import math

classes_to_check = ['Bend', 'Quadrupole', 'Sextupole', 'Octupole', 'Solenoid',
                    'Multipole']

for cc in classes_to_check:

    nn1 = 'test1_'+cc.lower()
    nn2 = 'test2_'+cc.lower()
    env = xt.Environment()
    env.new(nn1, cc, length=10, knl=[1,2,3,4,5,6,7,8,9,10,11,12], ksl=[3,2,1])
    env.new(nn2, cc, length=10, ksl=[1,2,3,4,5,6,7,8,9,10,11,12], knl=[3,2,1], order=11)

    assert env[nn1].__class__.__name__ == cc
    assert env[nn1].order == 11
    assert len(env[nn1].knl) == 12
    assert len(env[nn1].ksl) == 12
    xo.assert_allclose(env[nn1].knl, [1,2,3,4,5,6,7,8,9,10,11,12], rtol=0, atol=1e-15)
    xo.assert_allclose(env[nn1].ksl, [3,2,1,0,0,0,0,0,0,0, 0, 0 ], rtol=0, atol=1e-15)
    xo.assert_allclose(env[nn1].inv_factorial_order, 1/math.factorial(11), rtol=0, atol=1e-15)

    assert env[nn2].__class__.__name__ == cc
    assert env[nn2].order == 11
    assert len(env[nn2].ksl) == 12
    assert len(env[nn2].knl) == 12
    xo.assert_allclose(env[nn2].ksl, [1,2,3,4,5,6,7,8,9,10,11,12], rtol=0, atol=1e-15)
    xo.assert_allclose(env[nn2].knl, [3,2,1,0,0,0,0,0,0,0, 0, 0 ], rtol=0, atol=1e-15)
    xo.assert_allclose(env[nn2].inv_factorial_order, 1/math.factorial(11), rtol=0, atol=1e-15)

env.vars.default_to_zero = True
line = env.new_line(components=[
    env.new('b1', xt.Bend, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('q1', xt.Quadrupole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('s1', xt.Sextupole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('o1', xt.Octupole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('s2', xt.Solenoid, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
    env.new('m1', xt.Multipole, length=1, knl=['a', 'b', 'c'], ksl=['d', 'e', 'f']),
])

env['a'] = 3.
env['b'] = 2.
env['c'] = 1.
env['d'] = 4.
env['e'] = 5.
env['f'] = 6.

element_names = ['b1', 'q1']
order = 10

line.extend_knl_ksl(order=order, element_names=element_names)

assert line['b1'].order == order
assert line['q1'].order == order
assert line['s1'].order == 5
assert line['o1'].order == 5
assert line['s2'].order == 5
assert line['m1'].order == 2

xo.assert_allclose(line['b1'].inv_factorial_order, 1/math.factorial(order), rtol=0, atol=1e-15)
xo.assert_allclose(line['q1'].inv_factorial_order, 1/math.factorial(order), rtol=0, atol=1e-15)
xo.assert_allclose(line['s1'].inv_factorial_order, 1/math.factorial(5), rtol=0, atol=1e-15)
xo.assert_allclose(line['o1'].inv_factorial_order, 1/math.factorial(5), rtol=0, atol=1e-15)
xo.assert_allclose(line['s2'].inv_factorial_order, 1/math.factorial(5), rtol=0, atol=1e-15)
xo.assert_allclose(line['m1'].inv_factorial_order, 1/math.factorial(2), rtol=0, atol=1e-15)

xo.assert_allclose(line['b1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['b1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['q1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['q1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['s1'].knl, [3., 2., 1., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['s1'].ksl, [4., 5., 6., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['o1'].knl, [3., 2., 1., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['o1'].ksl, [4., 5., 6., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['s2'].knl, [3., 2., 1., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['s2'].ksl, [4., 5., 6., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['m1'].knl, [3., 2., 1.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['m1'].ksl, [4., 5., 6.] , rtol=0, atol=1e-15)

line.extend_knl_ksl(order=11)

assert line['b1'].order == 11
assert line['q1'].order == 11
assert line['s1'].order == 11
assert line['o1'].order == 11
assert line['s2'].order == 11
assert line['m1'].order == 11
assert line['b1'].inv_factorial_order == 1/math.factorial(11)
assert line['q1'].inv_factorial_order == 1/math.factorial(11)
assert line['s1'].inv_factorial_order == 1/math.factorial(11)
assert line['o1'].inv_factorial_order == 1/math.factorial(11)
assert line['s2'].inv_factorial_order == 1/math.factorial(11)
assert line['m1'].inv_factorial_order == 1/math.factorial(11)
xo.assert_allclose(line['b1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['b1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['q1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['q1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['s1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['s1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['o1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['o1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['s2'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['s2'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['m1'].knl, [3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)
xo.assert_allclose(line['m1'].ksl, [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.] , rtol=0, atol=1e-15)

# test an expression
line['b'] = 100
line['f'] = 200

xo.assert_allclose(line['o1'].knl, [3., 100., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
xo.assert_allclose(line['o1'].ksl, [4., 5., 200., 0., 0., 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
