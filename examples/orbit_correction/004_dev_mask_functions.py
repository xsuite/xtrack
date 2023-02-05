# Need to:
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
line.build_tracker()

from line_preparation import rename_coupling_knobs_and_coefficients

# Reanme coupling knobs to `c_minus_re_b1` and `c_minus_im_b1`
rename_coupling_knobs_and_coefficients(line=line, beamn=1)

# Check new knobs
assert np.abs(line.twiss().c_minus) < 1e-4
line.vars['c_minus_re_b1'] = 1e-3
assert np.isclose(line.twiss().c_minus, 1e-3, atol=1e-4, rtol=0)
line.vars['c_minus_re_b1'] = 0
line.vars['c_minus_im_b1'] = 1e-3
assert np.isclose(line.twiss().c_minus, 1e-3, atol=1e-4, rtol=0)
