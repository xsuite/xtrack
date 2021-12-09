import numpy as np

from scipy.constants import m_e as me_kg
from scipy.constants import e as qe
from scipy.constants import c as clight

from cpymad.madx import Madx

import xpart as xp
import xtrack as xt

#############################################
## Generate machine model with MAD-X script #
#############################################
mad = Madx()
mad.call("madseq.madx")
line = xt.Line.from_madx_sequence(mad.sequence['clic_ffs'])

###########################
# Build tracker (compile) #
###########################
tracker  = xt.Tracker(line=line, save_source_as='gensource.c')

#######################
# Switch on radiation #
#######################
for ee in tracker.line.elements:
    if hasattr(ee, 'radiation_flag'):
        #ee.radiation_flag = 1 # no random part
        ee.radiation_flag = 2 # with random part

####################
# Import particles #
####################

my_data = np.genfromtxt('beam_dist.txt', delimiter=',')
me_eV = me_kg*clight**2/qe
particles = xp.Particles(
        p0c=1500e9, # eV
        q0=-1,
        mass0=me_eV,
        x =my_data[:,0],
        px=my_data[:,1],
        y =my_data[:,2],
        py=my_data[:,3],
        zeta=my_data[:,4],
        delta=my_data[:,5])

particles._init_random_number_generator()

#########
# Track #
#########

## If you need to inspect element by element
# n_elements = len(tracker.line.elements)
# for iele in range(n_elements):
#     tracker.track(particles, ele_start=iele, num_elements=1)
#     # Updated coordinates can be accessed in place, e.g.:
#     # print(particles.x[0])

# For multiple turns:
particles0 = particles.copy()
tracker.track(particles)

P0 = 1500.
E0 = np.hypot(0.0005109989275678127, P0)
gamma = E0 / 0.0005109989275678127
beta = np.sqrt(gamma*gamma-1)/gamma
#delta_p1 = np.sqrt(pt*pt + 2.*pt/beta + 1.);

px = particles.px[:]
py = particles.py[:]
delta = particles.delta[:]
x = particles.x[:]
y = particles.y[:]
z = particles.zeta[:]

P = P0 * (1. + delta)
Px = P0 * px
Py = P0 * py
Pz = np.sqrt(P*P - Px*Px - Py*Py) 

xp = Px / Pz * 1e6 # urad
yp = Py / Pz * 1e6 # urad
x *= 1e6 # um
y *= 1e6 # um
z *= 1e6 # um

x = x.reshape(100000,1)
y = y.reshape(100000,1)
z = z.reshape(100000,1)
xp = xp.reshape(100000,1)
yp = yp.reshape(100000,1)
P = P.reshape(100000,1)

np.savetxt('ipdist_xtrack_rad2.txt', np.concatenate([P,x,y,z,xp,yp], axis=1), fmt='%.15g')

if 1:
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure(1);
    plt.plot(particles.x, particles.px, '.', markersize=1)
    plt.figure(2);
    plt.plot(particles.y, particles.py, '.', markersize=1)
    plt.show()

