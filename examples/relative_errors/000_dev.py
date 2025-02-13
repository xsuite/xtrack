import xtrack as xt

env = xt.Environment()
env.vars.default_to_zero = True
line = env.new_line(components=[
    env.new('mq', 'Quadrupole', length=0.5, k1='kq'),
    env.new('mqs', 'Quadrupole', length=0.5, k1s='kqs'),
    env.new('mb', 'Bend', length=0.5, angle='ang', k0_from_h=True),
])

env.set_multipolar_errors({
    'mq': {'rel_knl': [1e-6, 1e-5, 1e-4], 'rel_ksl': [-1e-6, -1e-5, -1e-4]},
    'mqs': {'rel_knl': [2e-6, 2e-5, 2e-4], 'rel_ksl': [3e-6, 3e-5, 3e-4], 'refer': 'k1s'},
    'mb': {'rel_knl': [2e-6, 3e-5, 4e-4], 'rel_ksl': [5e-6, 6e-5, 7e-4]},
})

env['kq'] = 0.1
env['kqs'] = 0.2
env['ang'] = 0.3


import xobjects as xo
import numpy as np

xo.assert_allclose(env.get('mq').knl[:3], 0.5 * 0.1 * np.array([1e-6, 1e-5, 1e-4]), rtol=1e-7, atol=0)
xo.assert_allclose(env.get('mq').ksl[:3], 0.5 * 0.1 * np.array([-1e-6, -1e-5, -1e-4]), rtol=1e-7, atol=0)
xo.assert_allclose(env.get('mqs').knl[:3], 0.5 * 0.2 * np.array([2e-6, 2e-5, 2e-4]), rtol=1e-7, atol=0)
xo.assert_allclose(env.get('mqs').ksl[:3], 0.5 * 0.2 * np.array([3e-6, 3e-5, 3e-4]), rtol=1e-7, atol=0)
xo.assert_allclose(env.get('mb').knl[:3], 0.3 * np.array([2e-6, 3e-5, 4e-4]), rtol=1e-7, atol=0)
xo.assert_allclose(env.get('mb').ksl[:3], 0.3 * np.array([5e-6, 6e-5, 7e-4]), rtol=1e-7, atol=0)
