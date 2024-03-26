import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo
import xpart as xp

import RF_Track as RFT

#############################################
##########    Input parameters    ###########
#############################################

# Bunch parameters
q0 = -1 # electrons
P0c = 50e6 # reference momentum, eV/c

# Solenoid parameters
B0 = 0.5 # T, on-axis field
R = 0.1 # m, aperture radius
Lsol = 1 # m, length

#############################################
#######  RF-Track's part starts here  #######
#############################################

# Setup the element
Sol = RFT.Solenoid(Lsol, B0, R)

# Setup the volume
Vsol = RFT.Volume()
Vsol.dt_mm = 1 # mm/c, integration step
Vsol.odeint_algorithm = 'rk2' # integration algorithm, e.g., 'rk2', 'rkf45', 'leapfrog' 
Vsol.set_s0(-0.75) # m, longitudinal start point
Vsol.set_s1( 0.75) # m, longitudinal end point
Vsol.add(Sol, 0, 0, 0, 'center');

#############################################
#######  RF-Track's part ends here    #######
#############################################

# Back to Xsuite

elements = {
    'sol': xt.RFT_Element(element=Vsol),
}

# Build the line
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=P0c, mass0=xt.ELECTRON_MASS_EV, q0=q0)

# Define the beam using
sigma_x = 0.01 # m, sigma x
sigma_y = 0.01 # m, sigma y

## Build particle object on context
n_part = 10000

rng = np.random.default_rng(12345)

# track
## Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code
line.build_tracker(_context=context)

particles = xp.Particles(
    _context=context,
    x=rng.normal(scale=sigma_x, size=n_part),
    y=rng.normal(scale=sigma_y, size=n_part),
    p0c=P0c, mass0=xt.ELECTRON_MASS_EV, q0=q0)

#############################################
#########   Xsuite tracking    ##############
#############################################

print('tracking starts')
line.track(particles)
print('tracking ends')

#############################################
## Retrieve Twiss from RF-Track and plot  ###
#############################################

# Plot the transverse plane
plt.figure(1)
plt.scatter(particles.x*1e3, particles.y*1e3, s=10, facecolors='none', edgecolors='b')
plt.xlabel("$x$ [mm]")
plt.ylabel("$y$ [mm]")
plt.show()

# Plot the field along the axis
Z = np.linspace(-0.75, 0.75, 1001) # m
O = np.zeros(1001)
[E,B] = Vsol.get_field(O, O, Z*1e3, O) # X [mm], Y [mm], Z [mm], T [mm/c]
plt.figure(2)
plt.plot(Z.transpose(), B[:,2])
plt.xlabel("$S$ [m]")
plt.ylabel("$B_z$ [T]")
plt.show()
