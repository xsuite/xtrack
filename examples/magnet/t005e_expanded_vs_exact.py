import xtrack as xt
import xobjects as xo

magnet = xt.Magnet(k0=0.002, h=0.002, k1=0.02, length=2)

m_exact = magnet.copy()
m_exact.model = 'bend-kick-bend'
m_exact.integrator='yoshida4'
m_exact.num_multipole_kicks = 1000

m_expanded = magnet.copy()
m_expanded.model = 'mat-kick-mat'
m_expanded.integrator='yoshida4'
m_expanded.num_multipole_kicks = 1000

p0 = xt.Particles(x=1e-3, y=2e-3, px=5e-6)

p_exact = p0.copy()
m_exact.track(p_exact)

p_expanded = p0.copy()
m_expanded.track(p_expanded)

xo.assert_allclose(p_exact.x, p_expanded.x, rtol=0, atol=1e-10)
xo.assert_allclose(p_exact.px, p_expanded.px, rtol=0, atol=1e-11)
xo.assert_allclose(p_exact.y, p_expanded.y, rtol=0, atol=2e-10)
xo.assert_allclose(p_exact.py, p_expanded.py, rtol=0, atol=1e-11)
xo.assert_allclose(p_exact.zeta, p_expanded.zeta, rtol=0, atol=1e-12)
xo.assert_allclose(p_exact.delta, p_expanded.delta, rtol=0, atol=1e-14)





