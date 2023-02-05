# Need to:
# - have indepedent coupling knobs for the two beams
# - have independent octupole knobs for the two beams
# - add knobs for optics correction


import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
line = xt.Line.from_dict(dct_b1)
