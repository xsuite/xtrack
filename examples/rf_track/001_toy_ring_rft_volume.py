import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import RF_Track as RFT

# Bend parameters
clight = 299792458 # m/s
p0c = 1.2e9  # eV
lbend = 3 # m
angle = np.pi / 2 # rad
rho = lbend / angle # m
By = p0c / rho / clight # T

#############################################
#######  RF-Track's part starts here  #######
#############################################

# Define the RF-Track element
vol = RFT.Volume()
vol.dt_mm = 0.1
vol.odeint_algorithm = 'rk2'
vol.set_static_Bfield(0.0, By, 0.0)
vol.set_s0(rho, 0.0, 0.0, 0.0, 0.0, 0.0)
vol.set_s1(0.0, 0.0, rho, 0.0, 0.0, -angle)
vol.set_length(lbend)

#############################################
#######  RF-Track's part ends here    #######
#############################################

# Back to Xsuite
pi = np.pi
lbend = 3
elements = {
    'd1.1':  xt.Drift(length=1),
    'mb1.1': xt.RFT_Element(element=vol),
    'd2.1':  xt.Drift(length=1),

    'mqd.1': xt.Quadrupole(length=0.3, k1=-0.7),
    'd3.1':  xt.Drift(length=1),
    'mb2.1': xt.RFT_Element(element=vol),
    'd4.1':  xt.Drift(length=1),

    'd1.2':  xt.Drift(length=1),
    'mb1.2': xt.RFT_Element(element=vol),
    'd2.2':  xt.Drift(length=1),

    'mqd.2': xt.Quadrupole(length=0.3, k1=-0.7),
    'd3.2':  xt.Drift(length=1),
    'mb2.2': xt.RFT_Element(element=vol),
    'd4.2':  xt.Drift(length=1),
}

# Build the ring
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV)
line.configure_bend_model(core='full', edge=None)

# track
## Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code
line.build_tracker(_context=context)

## Build particle object on context
n_part = 200

rng = np.random.default_rng(2021)

particles = xp.Particles(p0c=p0c, #eV
                        q0=1, mass0=xp.PROTON_MASS_EV,
                        x=rng.uniform(-1e-3, 1e-3, n_part),
                        px=rng.uniform(-1e-5, 1e-5, n_part),
                        y=rng.uniform(-2e-3, 2e-3, n_part),
                        py=rng.uniform(-3e-5, 3e-5, n_part),
                        zeta=rng.uniform(-1e-2, 1e-2, n_part),
                        delta=rng.uniform(-1e-3, 1e-3, n_part),
                        _context=context)

print('tracking starts')
line.track(particles)
print('tracking ends')

from scipy.io import savemat
v = np.transpose(np.array([particles.x, particles.px, particles.y, particles.py, particles.zeta, particles.delta]))
dict = { "v": v }
savemat("particles_rft_volume.mat", dict)
