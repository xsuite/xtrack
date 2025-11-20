import xtrack as xt
import xobjects as xo

mad_str = """
    sol1: solenoid, l=1.0, ks=0.5;
    sol2: sol1, lrad=0.2;
    sol3: sol2;

    sol4: solenoid, l=0, ks=1.0;
    sol5: sol4, lrad=0.1;
    sol6: sol5;
"""

env = xt.load(string=mad_str, format='madx')

assert isinstance(env['sol1'], xt.UniformSolenoid)
assert isinstance(env['sol2'], xt.UniformSolenoid)
assert isinstance(env['sol3'], xt.UniformSolenoid)
assert isinstance(env['sol4'], xt.UniformSolenoid)
assert isinstance(env['sol5'], xt.UniformSolenoid)
assert isinstance(env['sol6'], xt.UniformSolenoid)

xo.assert_allclose(env['sol1'].length, 1.0, rtol=0, atol=1e-12)
xo.assert_allclose(env['sol2'].length, 1.0, rtol=0, atol=1e-12)
xo.assert_allclose(env['sol3'].length, 1.0, rtol=0, atol=1e-12)
xo.assert_allclose(env['sol4'].length, 0.0, rtol=0, atol=1e-12)
xo.assert_allclose(env['sol5'].length, 0.0, rtol=0, atol=1e-12)
xo.assert_allclose(env['sol6'].length, 0.0, rtol=0, atol=1e-12)