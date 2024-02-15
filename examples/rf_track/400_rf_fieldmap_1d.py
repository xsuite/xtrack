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
P0c = 100e6 # reference momentum, eV/c

# Load the 1D field map of the CLIC RF structure
T = np.loadtxt(fname='400_Ez_on-axis_TD26__vg1p8_r05_CC_EDMS.txt', delimiter='\t')
Ez = np.complex64(T[:,1] + 1.j*T[:,2]) # V/m, take columns 1 and 2 to make a complex number
hz = T[1,0] - T[0,0] # mm, mesh step
freq = 11.994e9 # Hz, RF frequency

#############################################
#######  RF-Track's part starts here  #######
#############################################

# Setup the element
RF = RFT.RF_FieldMap_1d(Ez, hz/1e3, -1, freq, +1);
RF.set_P_map(4) # W, field map input power
RF.set_P_actual(10e6) # W, actual input power
RF.set_odeint_algorithm('rk2')
RF.set_smooth(10)

#############################################
#######  RF-Track's part ends here    #######
#############################################

# Back to Xsuite

elements = {
    'sol': xt.RFT_Element(element=RF),
}

# Build the line
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=P0c, mass0=xt.ELECTRON_MASS_EV, q0=q0)

# Define the beam using
sigma_x = 0.0001 # m, sigma x
sigma_y = 0.0001 # m, sigma y
sigma_z = 0.0005 # m, sigma z

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
    zeta=rng.normal(scale=sigma_z, size=n_part),
    p0c=P0c, mass0=xt.ELECTRON_MASS_EV, q0=q0)

#############################################
#########   Xsuite tracking    ##############
#############################################

print('tracking starts')
line.track(particles)
print('tracking ends')

print('initial reference momentum = ', P0c/1e6, 'MeV/c')
print('final reference momentum = ', particles.p0c[0]/1e6, 'MeV/c')

#############################################
## Retrieve Twiss from RF-Track and plot  ###
#############################################

# Plot the transverse phase space
plt.figure(1)
plt.scatter(particles.x*1e3, particles.y*1e3, s=10, facecolors='none', edgecolors='b')
plt.xlabel("$x$ [mm]")
plt.ylabel("$y$ [mm]")
plt.show()

# Plot the longitudinal phase space
plt.figure(2)
plt.scatter(particles.zeta*1e3, particles.delta*1e3, s=10, facecolors='none', edgecolors='b')
plt.xlabel("$z$ [mm]")
plt.ylabel("$\delta$ [permil]")
plt.show()

# Plot the field along the axis
Z = np.linspace(0.0, RF.get_length(), 1001) # m
O = np.zeros(1001)
[E,B] = RF.get_field(O, O, Z*1e3, O) # X [mm], Y [mm], Z [mm], T [mm/c]
plt.figure(3)
plt.plot(Z.transpose(), E[:,2]/1e6)
plt.xlabel("$S$ [m]")
plt.ylabel("$E_z$ [MV/m]")
plt.show()
