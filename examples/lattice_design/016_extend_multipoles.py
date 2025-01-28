import xtrack as xt
import xobjects as xo
import math

classes_to_check = ['Bend', 'Quadrupole', 'Sextupole', 'Octupole', 'Solenoid',
                    'Multipole']

for cc in classes_to_check:

    nn1 = 'test_'+cc.lower()
    env = xt.Environment()
    env.new(nn1, cc, length=10, knl=[1,2,3,4,5,6,7,8,9,10,11,12], ksl=[3,2,1])

    assert env[nn1].__class__.__name__ == cc
    assert env[nn1].order == 11
    xo.assert_allclose(env[nn1].knl, [1,2,3,4,5,6,7,8,9,10,11,12], rtol=0, atol=1e-15)
    xo.assert_allclose(env[nn1].ksl, [3,2,1,0,0,0,0,0,0,0, 0, 0 ], rtol=0, atol=1e-15)
    xo.assert_allclose(env[nn1].inv_factorial_order, 1/math.factorial(11), rtol=0, atol=1e-15)

