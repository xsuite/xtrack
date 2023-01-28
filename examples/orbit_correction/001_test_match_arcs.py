import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
line = xt.Line.from_dict(dct_b1)
tracker = line.build_tracker()

# Load line with knobs on correctors only
from cpymad.madx import Madx
mad = Madx()
mad.call('../../test_data/hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx')
mad.use(sequence='lhcb1')
line_co_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
    deferred_expressions=True,
    expressions_for_element_types=('kicker', 'hkicker', 'vkicker'))

tracker_co_ref = line_co_ref.build_tracker()
tracker_co_ref.particle_ref = tracker.particle_ref