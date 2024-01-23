import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo
import xpart as xp

import RF_Track as RFT

p0c = 1.2e9  # eV
Ld = 0.2 # m

#############################################
#######  RF-Track's part starts here  #######
#############################################

# Define the RF-Track element
dft = RFT.Drift(Ld)
dft.set_aperture(0.08, 0.02, 'rectangular')

#############################################
#######  RF-Track's part ends here    #######
#############################################

# Back to Xsuite
pi = np.pi
lbend = 3
elements = {
    'a1': xt.LimitRect(min_x=-0.02, max_x=0.02, min_y=-0.08, max_y=0.08),
    'd1': xt.Drift(length=Ld),
    'd2': xt.RFT_Element(element=dft),
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
n_part = 10000

rng = np.random.default_rng(12345)

particles = xp.Particles(p0c=p0c, #eV
                         q0=1, mass0=xp.PROTON_MASS_EV,
                         x=rng.uniform(-0.10, 0.10, n_part),
                         y=rng.uniform(-0.10, 0.10, n_part),
                         px=np.zeros(n_part),
                         py=np.zeros(n_part),
                         zeta=np.zeros(n_part),
                         delta=np.zeros(n_part),
                         _context=context)

print('tracking starts')
line.track(particles)
print('tracking ends')

plt.figure(1)
plt.scatter(particles.x*1e3, particles.y*1e3, s=10, facecolors='none', edgecolors='b')
plt.xlabel("$x$ [mm]")
plt.ylabel("$y$ [mm]")
plt.xlim([-80, 80])
plt.ylim([-80, 80])
plt.show()

