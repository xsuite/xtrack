# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time
import numpy as np

from cpymad.madx import Madx

import xtrack as xt
import xpart as xp
import xobjects as xo

# Import a thick sequence
mad = Madx()
mad.call('../../test_data/clic_dr/sequence.madx')
mad.use('ring')

# Makethin
mad.input(f'''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')

# Build xtrack line
print('Build xtrack line...')
line = xt.Line.from_madx_sequence(mad.sequence['RING'])
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=mad.sequence.ring.beam.gamma)

line.build_tracker()

################################
# Enable synchrotron radiation #
################################

# we choose the `mean` mode in which the mean power loss is applied without
# stochastic fluctuations (quantum excitation).
line.configure_radiation(model='mean')

#########
# Twiss #
#########

# In the presence of radiation the stability tolerance needs to be increded to
# allow twiss matrix determinant to be different from one.
line.matrix_stability_tol = 1e-2
tw = line.twiss(eneloss_and_damping=True)

bunch_intensity = 1e11
sigma_z = 5e-3
n_part = int(5e5)
nemitt_x = 0.5e-6
nemitt_y = 0.5e-6

particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         line=line)

assert np.isclose(np.std(particles.x),
                  np.sqrt(tw['dx'][0]**2*np.std(particles.delta)**2
                          + tw['betx'][0]*nemitt_x/mad.sequence.ring.beam.gamma),
                  atol=0, rtol=1e-2)

assert np.isclose(np.std(particles.y),
                  np.sqrt(tw['dy'][0]**2*np.std(particles.delta)**2
                          + tw['bety'][0]*nemitt_y/mad.sequence.ring.beam.gamma),
                  atol=0, rtol=1e-2)

assert np.isclose(np.std(particles.zeta), sigma_z, atol=0, rtol=1e-3)