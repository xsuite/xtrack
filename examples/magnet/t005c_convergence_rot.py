import xtrack as xt
import numpy as np
import xobjects as xo

magnet = xt.Magnet(k0=0.02, h=0.01, k1=0.01, length=2.,
                   k2=0.005, k3=0.03,
                   k1s=0.01, k2s=0.005, k3s=0.05,
                   knl=[0.003, 0.001, 0.01, 0.02, 4., 6e2, 7e6],
                   ksl=[-0.005, 0.002, -0.02, 0.03, -2, 700., 4e6])
magnet.integrator = 'yoshida4'
magnet.num_multipole_kicks = 50

p0 = xt.Particles(x=1e-2, y=2e-2, py=1e-3, delta=3e-2)

model_to_test = 'rot-kick-rot'

m_ref = magnet.copy()
m_ref.model = 'bend-kick-bend'
p_ref = p0.copy()
m_ref.track(p_ref)

m_uniform = magnet.copy()
m_uniform.model = model_to_test
m_uniform.integrator='uniform'
m_uniform.num_multipole_kicks = 50000

m_teapot = magnet.copy()
m_teapot.model = model_to_test
m_teapot.integrator='teapot'
m_teapot.num_multipole_kicks = 50000

m_yoshida = magnet.copy()
m_yoshida.model = model_to_test
m_yoshida.integrator='yoshida4'
m_yoshida.num_multipole_kicks = 100

p_ref = p0.copy()
p_uniform = p0.copy()
p_teapot = p0.copy()
p_yoshida = p0.copy()

m_ref.track(p_ref)
m_uniform.track(p_uniform)
m_teapot.track(p_teapot)
m_yoshida.track(p_yoshida)

xo.assert_allclose(p_ref.x, p_uniform.x, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.px, p_uniform.px, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.y, p_uniform.y, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.py, p_uniform.py, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.zeta, p_uniform.zeta, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.delta, p_uniform.delta, rtol=0, atol=5e-13)

xo.assert_allclose(p_ref.x, p_teapot.x, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.px, p_teapot.px, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.y, p_teapot.y, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.py, p_teapot.py, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.zeta, p_teapot.zeta, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.delta, p_teapot.delta, rtol=0, atol=5e-13)

xo.assert_allclose(p_ref.x, p_yoshida.x, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.px, p_yoshida.px, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.y, p_yoshida.y, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.py, p_yoshida.py, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.zeta, p_yoshida.zeta, rtol=0, atol=5e-13)
xo.assert_allclose(p_ref.delta, p_yoshida.delta, rtol=0, atol=5e-13)
