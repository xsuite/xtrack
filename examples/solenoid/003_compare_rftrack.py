import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo
import xpart as xp

from scipy.constants import c as clight

import RF_Track as RFT

#############################################
##########    Input parameters    ###########
#############################################

# Bunch parameters
q0 = -1 # electrons
P0c = 200e9 # reference momentum, eV/c

# Solenoid parameters
B0 = 2 # T, on-axis field
R = 0.1 # m, aperture radius
Lsol = 1 # m, length

#############################################
#######  RF-Track's part starts here  #######
#############################################

# Element setup
Sol = RFT.Solenoid(Lsol, B0, R)

# Effects

ISR = RFT.IncoherentSynchrotronRadiation(quantum=False)

# Setup the volume
Vsol = RFT.Volume()
Vsol.dt_mm = 1 # mm/c, integration step
Vsol.tt_dt_mm = 10; # mm/c tabulate average quantities every tt_dt_mm
Vsol.cfx_dt_mm = 1; # mm/c apply a kick of collective effects (ISR) every cfx_dt_mm
Vsol.odeint_algorithm = 'rk2' # integration algorithm, e.g., 'rk2', 'rkf45', 'leapfrog' 

# Add the solenoid
Vsol.add(Sol, 0, 0, 0, 'center'); # add the solenoid

# Set boundaries
vol_length = 3
Vsol.set_s0(-vol_length/2) # m, longitudinal start point
Vsol.set_s1( vol_length/2) # m, longitudinal end point

# Add collective effects
Vsol.add_collective_effect (ISR)

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

# track
## Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code
line.build_tracker(_context=context)

particle0 = xp.Particles(
    _context=context,
    x=0.01,
    p0c=P0c, mass0=xt.ELECTRON_MASS_EV, q0=q0)
p_rft = particle0.copy()

#############################################
#########   Xsuite tracking    ##############
#############################################

print('tracking starts')
line.track(p_rft)
print('tracking ends')

#############################################
#######      Back to RF-Track      ##########
#######  for field investigation   ##########
#############################################

# Since Vsol is passed 'by refernce', we can inquire it directly
# Alternatively, Vsol = line.get('sol').lattice[0];

table = Vsol.get_transport_table('%mean_Z %mean_X %mean_Y %mean_Px %mean_Py %mean_E') # mm mm mm MeV/c MeV/c MeV

# Extract trajectory
ztraj = table[:,0]
xtraj = table[:,1]
ytraj = table[:,2]
etraj = table[:,5]

# Inquire field
[E,B] = Vsol.get_field(xtraj, ytraj, ztraj, 0.0)

# Field on axis
z_axis = np.linspace(-2, 2, 1001)
z_axis_mm = z_axis * 1000
[E_axis,B_axis] = Vsol.get_field(0, 0, z_axis_mm, 0.0)

###################
# Xsuite solenoid #
###################

particle_ref = xt.Particles(q0=q0, mass0=xt.ELECTRON_MASS_EV, p0c=P0c)

B_slice_center = 0.5 * (B_axis[:-1, 2] + B_axis[1:, 2])
l_slice = (z_axis[1:] - z_axis[:-1])

brho = particle_ref.p0c[0] / clight / particle_ref.q0

env = xt.Environment()
env.particle_ref = particle_ref

slices = []
slices.append(env.new('sol_entry', xt.Solenoid, ks=0, length=0))
for ii, bb in enumerate(B_slice_center):
    slices.append(env.new(f'sol_{ii}', xt.Solenoid, ks=bb/brho, length=l_slice[ii]))
slices.append(env.new('sol_exit', xt.Solenoid, ks=0, length=0))


line_sol_xs = env.new_line(components=slices)
line_sol_xs.insert('mid_sol', xt.Marker(), line_sol_xs.get_length()/2)


p_xs = particle0.copy()
line_sol_xs.configure_radiation('mean')
tw = line_sol_xs.twiss(init=xt.TwissInit(betx=1, bety=1, particle_on_co=p_xs),
                       strengths=True, zero_at='mid_sol')

#############################################
#################   Plots    ################
#############################################

plt.close('all')

# Plot the field along the axis
plt.figure(1)
plt.plot(ztraj, B[:,0], label='B_x')
plt.plot(ztraj, B[:,1], label='B_y')
plt.plot(ztraj, B[:,2], label='B_z')
plt.xlabel("$Z$ [mm]")
plt.ylabel("$B$ [T]")
plt.legend()
plt.show()

plt.figure(2, figsize=(8, 6*1.5))
ax1 = plt.subplot(311)
plt.plot(tw.s, tw.y, label='xsuite')
plt.plot(ztraj*1e-3, ytraj*1e-3, label='RF-Track')
plt.ylabel("$y$ [m]")
plt.legend()
plt.subplot(312, sharex=ax1)
plt.plot(tw.s, tw.x, label='xsuite')
plt.plot(ztraj*1e-3, xtraj*1e-3, label='RF-Track')
plt.ylabel("$x$ [m]")
plt.subplot(313, sharex=ax1)
plt.plot(tw.s, tw.delta, label='xsuite')
energy_0_mev = tw.particle_on_co.energy0[0] / 1e6
plt.plot(ztraj*1e-3, (etraj - energy_0_mev) / energy_0_mev, label='RF-Track')
plt.ylabel("$\delta$")
plt.xlabel("$s$ [m]")
plt.show()
