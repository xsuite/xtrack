import xtrack as xt

def zero_field(x, y, z):
    return (0*x, 0*y, 0*z)

integrator = xt.BorisSpatialIntegrator(fieldmap_callable=zero_field,
                                        s_start=10,
                                        s_end=20,
                                        n_steps=15000)