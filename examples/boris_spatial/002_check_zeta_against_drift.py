import xtrack as xt

def zero_field(x, y, z):
    return (0*x, 0*y, 0*z)

integrator = xt.BorisSpatialIntegrator(fieldmap_callable=zero_field,
                                        s_start=10,
                                        s_end=20,
                                        n_steps=15000)

drift = xt.Drift(length=10, model='exact')

p0 = xt.Particles('proton', x=1e-2, px=15e-3, p0c=1e9, zeta=1)

p_integ = p0.copy()
integrator.track(p_integ)

p_drift = p0.copy()
drift.track(p_drift)