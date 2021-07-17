import numpy as np

def generate_transverse_distribution(p0c, twiss_at_start, neps_x, neps_y,
                                        delta,n_macroparticles):
    assert(len(delta) == n_macroparticles)
    alfa_x = twiss_at_start['alfx']
    alfa_y = twiss_at_start['alfy']
    beta_x = twiss_at_start['betx']
    beta_y = twiss_at_start['bety']
    dx = twiss_at_start['dx']
    dy = twiss_at_start['dy']
    dpx = twiss_at_start['dpx']
    dpy = twiss_at_start['dpy']

    sigma_x  = np.sqrt(neps_x * beta_x / temp.beta0 / temp.gamma0)
    sigma_y  = np.sqrt(neps_y * beta_y / temp.beta0 / temp.gamma0)
    sigma_px = np.sqrt(neps_x / beta_x / temp.beta0 / temp.gamma0)
    sigma_py = np.sqrt(neps_y / beta_y / temp.beta0 / temp.gamma0)

    x_wrt_CO = np.random.normal(loc=0.0, scale=sigma_x, size=n_macroparticles)
    y_wrt_CO = np.random.normal(loc=0.0, scale=sigma_y, size=n_macroparticles)
    px_wrt_CO = np.random.normal(loc=0.0, scale=sigma_px,
                            size=n_macroparticles) - alfa_x/beta_x * x_wrt_CO
    py_wrt_CO = np.random.normal(loc=0.0, scale=sigma_py,
                            size=n_macroparticles) - alfa_y/beta_y * y_wrt_CO
    # account for dispersion
    x_wrt_CO += delta * dx
    y_wrt_CO += delta * dy
    px_wrt_CO += delta * dpx
    py_wrt_CO += delta * dpy

    return x_wrt_CO, y_wrt_CO, px_wrt_CO, py_wrt_CO
