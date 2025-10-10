import xtrack as xt
import xobjects as xo

def zero_field(x, y, z):
    return (0*x, 0*y, 0*z)

integrator = xt.BorisSpatialIntegrator(fieldmap_callable=zero_field,
                                        s_start=10,
                                        s_end=20,
                                        n_steps=15000)

drift = xt.Drift(length=10, model='exact')

p0 = xt.Particles('proton',
        x=[1e-2, -1e-2], px=[15e-3, 0], delta=[0, 2], p0c=1e9, zeta=1)

p_integ = p0.copy()
integrator.track(p_integ)

p_drift = p0.copy()
drift.track(p_drift)

xo.assert_allclose(p_integ.x, p_drift.x, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.px, p_drift.px, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.y, p_drift.y, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.py, p_drift.py, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.s, p_drift.s, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.delta, p_drift.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.zeta, p_drift.zeta, rtol=0, atol=1e-10)

# Check behavior with lost particles
p0 = xt.Particles('proton',
        x=[1e-2, -1e-2], px=[15e-3, 0], delta=[0, 2], p0c=1e9, zeta=1,
        state=[1, -1])

p_integ = p0.copy()
integrator.track(p_integ)
p_drift = p0.copy()
drift.track(p_drift)

xo.assert_allclose(p_integ.x, p_drift.x, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.px, p_drift.px, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.y, p_drift.y, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.py, p_drift.py, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.s, p_drift.s, rtol=0, atol=1e-10)
xo.assert_allclose(p_integ.delta, p_drift.delta, rtol=0, atol=1e-10)