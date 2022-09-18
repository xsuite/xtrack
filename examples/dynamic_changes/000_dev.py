import numpy as np

from cpymad.madx import Madx

import xtrack as xt

# Import SPS lattice
mad = Madx()
seq_name = 'sps'
mad.call('../../test_data/sps_w_spacecharge/sps_thin.seq')
mad.use(seq_name)
madtw = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence[seq_name])
tracker = line.build_tracker()

# Extract list of elements to trim (all focusing quads)
elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf.')]

field_to_trim = 'knl'
index_to_trim = 1

cs = xt.CustomSetter(tracker=tracker, elements=elements_to_trim,
                  field=field_to_trim, index=index_to_trim)
values = cs.get_values()
cs.set_values(values*1.1)

