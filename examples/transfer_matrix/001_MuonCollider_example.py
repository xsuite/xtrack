import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

import xobjects as xo
context = xo.ContextCpu(omp_num_threads=0)
import xtrack as xt
import xfields as xf
import xpart as xp

muonMass = constants.value('muon mass energy equivalent in MeV')*1E6

n_macroparticles = 10
energy = 1.25 # [GeV]
p0c = np.sqrt((energy*1E9)**2-muonMass**2)
gamma = np.sqrt(1+(p0c/muonMass)**2)
beta = np.sqrt(1-1/gamma**2)
normemit_x = 25E-6
normemit_y = 25E-6
sigma_z = 4.6E-2
emit_z = 7.5E-3
sigma_delta = emit_z/sigma_z/energy
beta_s = sigma_z/sigma_delta

beta_x_IP1 = 10.0
beta_y_IP1 = 10.0
alpha_x_IP1 = 0.0
alpha_y_IP1 = 0.0
dispersion_IP1 = 0.0

beta_x_IP2 = 0.1
beta_y_IP2 = 0.1
alpha_x_IP2 = 0.0
alpha_y_IP2 = 2.0
dispersion_IP2 = 1.0

nTurn = 500
energy_increment = (5.0-1.25)*1E9/nTurn/2
#energy_increment = 0.0

particles = xp.Particles(_context=context,
                         q0 = 1,
                         mass0 = muonMass,
                         p0c=p0c,
                         x=np.sqrt(normemit_x*beta_x_IP1/gamma)*np.linspace(0,6,n_macroparticles),
                         px=np.sqrt(normemit_x/beta_x_IP1/gamma)*np.zeros(n_macroparticles,dtype=float),
                         y=np.sqrt(normemit_y*beta_y_IP1/gamma)*np.linspace(0,6,n_macroparticles),
                         py=np.sqrt(normemit_y/beta_y_IP1/gamma)*np.zeros(n_macroparticles,dtype=float),
                         zeta=sigma_z*np.zeros(n_macroparticles),
                         delta=sigma_delta*np.ones(n_macroparticles,dtype=float),
                         )

arc12 = xt.LinearTransferMatrix(
        alpha_x_0=alpha_x_IP1, beta_x_0=beta_x_IP1, disp_x_0=dispersion_IP1,
        alpha_x_1=alpha_x_IP2, beta_x_1=beta_x_IP2, disp_x_1=dispersion_IP2,
        alpha_y_0=alpha_y_IP1, beta_y_0=beta_y_IP1, disp_y_0=0.0,
        alpha_y_1=alpha_y_IP2, beta_y_1=beta_y_IP2, disp_y_1=0.0,
        Q_x=0.155, Q_y=0.16,
        beta_s=beta_s, Q_s=0.155,
        energy_ref_increment=energy_increment,energy_increment=0.0)
arc21 = xt.LinearTransferMatrix(
        alpha_x_0=alpha_x_IP2, beta_x_0=beta_x_IP2, disp_x_0=dispersion_IP2,
        alpha_x_1=alpha_x_IP1, beta_x_1=beta_x_IP1, disp_x_1=dispersion_IP1,
        alpha_y_0=alpha_y_IP2, beta_y_0=beta_y_IP2, disp_y_0=0.0,
        alpha_y_1=alpha_y_IP1, beta_y_1=beta_y_IP1, disp_y_1=0.0,
        Q_x=0.155, Q_y=0.16,
        beta_s=beta_s, Q_s=0.155,
        energy_ref_increment=energy_increment,energy_increment=0.0)

plt.figure(12)
plt.plot(particles.zeta,particles.delta,'.b')
plt.axvline(sigma_z,color='k')
plt.axvline(-sigma_z,color='k')
plt.axhline(sigma_delta,color='k')
plt.axhline(-sigma_delta,color='k')


for turn in range(nTurn):
    plt.figure(0)
    plt.plot(particles.x,particles.px,'.b')
    plt.figure(1)
    plt.plot(particles.y,particles.py,'.g')
    plt.figure(3)
    plt.plot(turn,0.5*((particles.x[1]-dispersion_IP1*particles.delta[1])**2*(1+alpha_x_IP1**2)/beta_x_IP1+2*alpha_x_IP1*(particles.x[1]-dispersion_IP1*particles.delta[1])*particles.px[1]+particles.px[1]**2*beta_x_IP1)*particles.gamma0[1]*particles.beta0[1],'x')
    arc12.track(particles)
    plt.figure(10)
    plt.plot(particles.x,particles.px,'.b')
    plt.figure(11)
    plt.plot(particles.y,particles.py,'.g')
    plt.figure(13)
    plt.plot(turn,0.5*((particles.x[1]-dispersion_IP2*particles.delta[1])**2*(1+alpha_x_IP2**2)/beta_x_IP2+2*alpha_x_IP2*(particles.x[1]-dispersion_IP2*particles.delta[1])*particles.px[1]+particles.px[1]**2*beta_x_IP2)*particles.gamma0[1]*particles.beta0[1],'x')
    arc21.track(particles)

    plt.figure(12)
    plt.plot(particles.zeta,particles.delta,'.r')

    plt.figure(100)
    plt.plot(turn,particles.mass0*particles.gamma0[1]*1E-9,'xk')

plt.show()



