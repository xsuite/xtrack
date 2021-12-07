import numpy as np

from scipy.constants import m_e as me_kg
from scipy.constants import e as qe
from scipy.constants import c as clight

from cpymad.madx import Madx

import xline as xl
import xtrack as xt

#############################################
## Generate machine model with MAD-X script #
#############################################
mad = Madx()
mad.call("madseq.madx")
line = xl.Line.from_madx_sequence(mad.sequence['clic_ffs'])

###########################
# Build tracker (compile) #
###########################
tracker  = xt.Tracker(sequence=line, save_source_as='gensource.c')

#######################
# Switch on radiation #
#######################
for ee in tracker.line.elements:
    if hasattr(ee, 'radiation_flag'):
        ee.radiation_flag = 1 # no random part
        #ee.radiation_flag = 1 # no random part
        #ee.radiation_flag = 2 # with random part


######################
# Generate particles #
######################

me_eV = me_kg*clight**2/qe
n_part = 10000
particles = xt.Particles(
        p0c=1500e9, # eV
        mass0=me_eV,
        x=np.random.uniform(-1e-3, 1e-3, n_part),
        px=np.random.uniform(-1e-6, 1e-6, n_part),
        y=np.random.uniform(-1e-3, 1e-3, n_part),
        py=np.random.uniform(-1e-6, 1e-6, n_part),
        zeta=np.random.uniform(-1e-2, 1e-2, n_part),
        delta=np.random.uniform(-1e-4, 1e-4, n_part))

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
# particles0 = particles.copy()
tracker.track(particles)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1);
plt.plot(particles.x, particles.px, '.')
plt.figure(2);
plt.plot(particles.y, particles.py, '.')
plt.show()

