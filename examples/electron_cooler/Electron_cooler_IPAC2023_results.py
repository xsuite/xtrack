#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp
#This file will generate and display the following IPAC results:
#https://doi.org/10.18429/JACoW-IPAC2023-TUPM027
#https://www.ipac23.org/preproc/doi/jacow-ipac2023-tupm027/index.html

#lattice parameters for AD at 300 MeV
beta_rel = 0.305
gamma = 1.050

qx = 5.44
qy = 5.42

beta_x = 10 
beta_y = 4

clight = 299792458.0
circumference = 182.43280000000 #m
s_per_turn = circumference/(clight*beta_rel)

context = xo.ContextCpu(omp_num_threads='auto')

arc = xt.LineSegmentMap(
        qx=qx, qy=qx,
        dqx=0, dqy=0,
        length=circumference,
        betx=beta_x,
        bety=beta_y)

mass0 = 938.27208816*1e6 #ev/c^2
p0c = mass0*beta_rel*gamma #eV/c

#electron cooler parameters
current = 2.4 # A current
length = 1.5 # m cooler length
radius_e_beam = 25*1e-3 #m radius of the electron beam

temp_perp = 100e-3 # <E> [eV] = kb*T
temp_long =  1e-3 # <E> [eV]
magnetic_field = 0.060 # T for AD

emittance = 35*1e-6 #inital emittance
x_init = np.sqrt(beta_x*emittance)
y_init = np.sqrt(beta_y*emittance)

magnetic_field_ratio_list = [0,1e-4,5e-4,1e-3] #Iterate over different values of the magnetic field quality to see effect on cooling performance.
#magnetic_field_ratio is the ratio of transverse componenet of magnetic field and the longitudinal component. In the ideal case, the ratio is 0.

for magnetic_field_ratio in magnetic_field_ratio_list:

    particle = xp.Particles(
            
            mass0=mass0,
            p0c=p0c,
            x=x_init,
            px=0,
            y=0,
            py=0,
            delta=0,
            zeta=0)
    #load electron cooler beam element
    electron_cooler = xt.ElectronCooler(current=current,length=length,radius_e_beam=radius_e_beam,
                                            temp_perp=temp_perp,temp_long=temp_long,
                                            magnetic_field=magnetic_field,magnetic_field_ratio=magnetic_field_ratio,
                                            space_charge=0)
    num_turns = int(1*1e7)
    save_interval = 1000

    action_x = []

    for i in range(num_turns):
        if i % save_interval == 0:
            # # calculate action in horizontal plane
            action_x_temp = (particle.x**2/beta_x + beta_x*particle.px**2)
            action_x.append(action_x_temp)

        arc.track(particle)
        electron_cooler.track(particle)

    filepath = f'results/action_magnetic_field_ratio={magnetic_field_ratio:.0e}.npz'
    np.savez(filepath, action_x=action_x)

time = np.arange(0, num_turns, save_interval) * s_per_turn

np.savez('results/time.npz', time=time)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
magnetic_field_ratio_list=[0,1e-4,5e-4,1e-3]
time_data = np.load('results/time.npz')['time']
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 14})  # set the default fontsize to 14
for magnetic_field_ratio in magnetic_field_ratio_list:
    filepath = f'results/action_magnetic_field_ratio={magnetic_field_ratio:.0e}.npz'
    data = np.load(filepath)
    action_x = data['action_x']*1e6
    plt.plot(time,action_x,label='$B_{\perp}/B_{\parallel}$='f'{magnetic_field_ratio:.0e}')

plt.xlabel('Time [s]')
plt.ylabel('$J_x$ $[\mu m]$')
plt.legend()

plt.savefig('IPAC_results.eps', dpi=300)
plt.savefig('IPAC_results.png', dpi=300)

