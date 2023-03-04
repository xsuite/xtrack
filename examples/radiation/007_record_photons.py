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

# Build line
line.build_tracker()

line.configure_radiation(model='quantum')

record = line.start_internal_logging_for_elements_of_type(
                                                xt.Multipole, capacity=100000)

particles = xp.build_particles(line=line, x=[0,0,0,0])

line.track(particles, num_turns=10)

import matplotlib.pyplot as plt
hist, bin_edges = np.histogram(record.photon_energy[:record._index.num_recorded], bins=100)
plt.close('all')
plt.loglog(bin_edges[1:], hist/np.diff(bin_edges))
plt.xlabel('Photon energy [eV]')
plt.ylabel('dN/dE [1/eV]')
plt.grid(True)
plt.show()