import numpy as np
from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
import xpart as xp

# Choose a context
ctx = xo.ContextCpu()

# Import SPS lattice and build a tracker
mad = Madx()
seq_name = 'sps'
mad.call('../../test_data/sps_w_spacecharge/sps_thin.seq')
mad.use(seq_name)
madtw = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence[seq_name])
line.particle_ref = xp.Particles(p0c=400e9, mass0=xp.PROTON_MASS_EV)
tracker = line.build_tracker(_context=ctx)

line['acta.31637'].voltage = 7e9
line['acta.31637'].lag = 180.

# Extract list of elements to trim (all focusing quads)
elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf.')]
# => contains ['qf.52010', 'qf.52210', 'qf.52410', 'qf.52610', 'qf.52810',
#              'qf.53010', 'qf.53210', 'qf.53410', 'qf.60010', 'qf.60210', ...]

# Build a custom setter
qf_setter = xt.CustomSetter(tracker, elements_to_trim,
                            field='knl', index=1 # we want to change knl[1]
                            )

# Get the initial values of the quand strength
k1l_0 = qf_setter.get_values()

# Generate particles to be tracked
# (we choose to match the distribution without accounting for spacecharge)
particles = xp.generate_matched_gaussian_bunch(_context=ctx,
         num_particles=1000, total_intensity_particles=1e10,
         nemitt_x=3e-6, nemitt_y=3e-6, sigma_z=15e-2,
         tracker=tracker)





