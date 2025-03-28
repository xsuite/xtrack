# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# This file will generate and display the following IPAC results:
# https://accelconf.web.cern.ch/ipac2023/doi_per_institute/tupm027/index.html
# Specifications for a new electron cooler of the antiproton decelerator at CERN 


import numpy as np
import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt

######################################
# lattice parameters for AD at 300 MeV
######################################

#lattice parameters for AD at 300Mev/c
# from https://acc-models.web.cern.ch/acc-models/ad/scenarios/lowenergy/lowenergy.tfs
qx = 5.45020077392
qy = 5.41919929346
dqx=-20.10016919292
dqy=-22.29552755573
circumference = 182.43280000000 #m

# relativistic factors
gamma_rel = 1.04987215550 # at 300 MeV/c

# optics at e-cooler (approximate), in m
beta_x = 10 
beta_y = 4
D_x = 0

# electron cooler parameters
current = 2.4  # A current
length = 1.5 # m cooler length
radius_e_beam = 25*1e-3 #m radius of the electron beam
temp_perp = 100e-3 # <E> [eV] = kb*T
temp_long =  1e-3 # <E> [eV]
magnetic_field = 0.060 # 100 Gauss in ELENA
# idea is to study magnetic field imperfections
magnetic_field_ratio_list = [0,1e-4,5e-4,1e-3] #Iterate over different values of the magnetic field quality to see effect on cooling performance.
#magnetic_field_ratio is the ratio of transverse componenet of magnetic field and the longitudinal component. In the ideal case, the ratio is 0.

# some initial beam parameters
emittance = 35e-6
dp_p = 2e-3 
q0 = 1

# simulation parameters: simulate 20 s of cooling, and take data once every 100 ms
max_time_s = 20
int_time_s = 0.01

# some constants, and simple computations
clight = 299792458.0
mass0 = 938.27208816*1e6 #ev/c^2

beta_rel = np.sqrt(gamma_rel**2 - 1)/gamma_rel
p0c = mass0*beta_rel*gamma_rel #eV/c
T_per_turn = circumference/(clight*beta_rel)

# compute length of simulation, as well as sample interval, in turns
num_turns = int(max_time_s/T_per_turn)
save_interval = int(int_time_s/T_per_turn)

# compute initial beam parameters
x_init = np.sqrt(beta_x*emittance)
y_init = np.sqrt(beta_y*emittance)

particles0 = xp.Particles(
            mass0=mass0,
            p0c=p0c,
            q0=q0,
            x=x_init,
            px=0,
            y=0,
            py=0,
            delta=0,
            zeta=0)        

# arc to do linear tracking
arc = xt.LineSegmentMap(
        qx=qx, qy=qx,
        dqx=dqx, dqy=dqy,
        length=circumference,
        betx=beta_x,
        bety=beta_y,
        dx=D_x)

# compute length of simulation, as well as sample interval, in turns
num_turns = int((max_time_s / T_per_turn).item())
save_interval = int((int_time_s / T_per_turn).item())

# create a monitor object, to reduce holded data
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                        n_repetitions=int(num_turns/save_interval),
                        repetition_period=save_interval,
                        num_particles=1)

##############################
# electron cooler parameters #
##############################  
electron_cooler = xt.ElectronCooler(
                length=length,
                radius_e_beam=radius_e_beam,
                current=current,
                temp_perp=temp_perp,
                temp_long=temp_long,
                magnetic_field=magnetic_field, 
                magnetic_field_ratio=0,
                space_charge_factor=0)

line = xt.Line(elements=[monitor, electron_cooler, arc],element_names=['monitor','electron_cooler','arc'])
line.particle_ref = xp.Particles(mass0=mass0, q0=q0, p0c=p0c)
line.build_tracker()

################
# prepare plot #
################

plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 14}) 

##########################################
# scan over magnetic field imperfections #
##########################################

for magnetic_field_ratio in magnetic_field_ratio_list:        
        
        electron_cooler.magnetic_field_ratio=magnetic_field_ratio
        particles=particles0.copy()
        # track
        line.track(particles, num_turns=num_turns,
              turn_by_turn_monitor=False)
        x = monitor.x[:,:,0]
        px = monitor.px[:,:,0]        
        time = monitor.at_turn[:, 0, 0] * T_per_turn
        # compute action at the end for all turns
        action_x = (x**2/beta_x + beta_x*px**2)
        
        plt.plot(time,action_x*1e6,label='$B_{\\perp}/B_{\\parallel}$='f'{magnetic_field_ratio:.0e}')

plt.xlabel('Time [s]')
plt.ylabel('$J_x$ $[\\mu m]$')
plt.legend()
plt.show()
