import xtrack as xt
import xobjects as xo


from xtrack.beam_elements.magnets import MagnetDrift

md = MagnetDrift(length=1.0, k0=0.0, k1=0.0, h=0.0, drift_model=0)

p0 = xt.Particles(kinetic_energy0=50e6,
                 x=1e-3, y=2e-3, zeta=1e-2, px=10e-3, py=20e-3, delta=1e-2)

# Expanded drift
p_test = p0.copy()
p_ref = p0.copy()

eref = xt.Drift(length=1.0)
md.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# Exact drift
md.drift_model = 1

p_test = p0.copy()
p_ref = p0.copy()

eref = xt.Solenoid(length=1.0) # Solenoid is exact drift when off
md.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)


