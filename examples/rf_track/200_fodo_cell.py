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
mass = xp.ELECTRON_MASS_EV # eV/c^2
q0 = -1 # electrons
P0c = 50e6 # reference momentum, eV/c
P_over_q = P0c / q0 # V/c, reference rigidity

# FODO cell paramters
mu = np.pi/2 # rad, phase advance
Lcell = 2 # m, fodo cell length
Lquad = 0 # m, a thin quad

Ldrift = Lcell/2 - Lquad # m
k1L = np.sin(mu/2) / (Lcell/4) # 1/m, analytic quad integrated strength

#############################################
#######  RF-Track's part starts here  #######
#############################################

# Setup the elements
strength = k1L * P_over_q / 1e6 # MeV/m
Qf = RFT.Quadrupole(Lquad/2, strength/2) # 1/2 quad
QD = RFT.Quadrupole(Lquad, -strength) # full quad
Dr = RFT.Drift(Ldrift)
Dr.set_tt_nsteps(100) # for plots

# Setup the lattice
FODO = RFT.Lattice()
FODO.append(Qf) # 1/2 quad
FODO.append(Dr)
FODO.append(QD) # full quad
FODO.append(Dr)
FODO.append(Qf) # 1/2 quad

#############################################
#######  RF-Track's part ends here    #######
#############################################

# Back to Xsuite

elements = {
    'fodo': xt.RFT_Element(element=FODO),
}

# Build the line
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=P0c, mass0=xt.ELECTRON_MASS_EV, q0=q0)

# Define the beam using the Twiss parameters
beta_gamma = P0c / mass
norm_emitt_x = 0.001 # mm.mrad, normalized emittance
norm_emitt_y = 0.001 # mm.mrad
geom_emitt_x = norm_emitt_x / beta_gamma / 1e6 # m.rad, geometric emittance
geom_emitt_y = norm_emitt_x / beta_gamma / 1e6 # m.rad

beta_x = Lcell * (1 + np.sin(mu/2)) / np.sin(mu) # m, analytic matched beta
beta_y = Lcell * (1 - np.sin(mu/2)) / np.sin(mu) # m

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
    x=rng.normal(scale=np.sqrt(geom_emitt_x*beta_x), size=n_part),
    y=rng.normal(scale=np.sqrt(geom_emitt_y*beta_y), size=n_part),
    px=rng.normal(scale=np.sqrt(geom_emitt_x/beta_x), size=n_part),
    py=rng.normal(scale=np.sqrt(geom_emitt_y/beta_y), size=n_part),
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

# Retrieve the Twiss plot and the phase space
T = FODO.get_transport_table('%S %beta_x %beta_y')
plt.figure(1)
plt.plot(T[:,0], T[:,1], 'b-', linewidth=2, label='beta x')
plt.plot(T[:,0], T[:,2], 'r-', linewidth=2, label='beta y')
plt.legend(loc="upper left")
plt.xlabel('S [m]')
plt.ylabel('beta [m]')
plt.show()



