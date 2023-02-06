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
from line_preparation import define_octupole_current_knobs
from line_preparation import add_correction_term_to_dipole_correctors

# Reanme coupling knobs to `c_minus_re_b1` and `c_minus_im_b1`
rename_coupling_knobs_and_coefficients(line=line, beamn=1)

# Check new knobs
assert np.abs(line.twiss().c_minus) < 1e-4
line.vars['c_minus_re_b1'] = 1e-3
assert np.isclose(line.twiss().c_minus, 1e-3, atol=1e-4, rtol=0)
line.vars['c_minus_re_b1'] = 0
line.vars['c_minus_im_b1'] = 1e-3
assert np.isclose(line.twiss().c_minus, 1e-3, atol=1e-4, rtol=0)

# Switch on octupoles
define_octupole_current_knobs(line=line, beamn=1)

# Check against these references
i_ioct_ref = -235.
k_ref = -6.928009
knl_ref = -2.21696289

line.vars['i_oct_b1'] = i_ioct_ref/2
assert np.isclose(line.vars['kof.a12b1']._value, k_ref/2, atol=0, rtol=1e-5)
assert np.isclose(line.vars['kod.a45b1']._value, k_ref/2, atol=0, rtol=1e-5)
assert np.isclose(line['mo.24r2.b1'].knl[3], knl_ref/2, atol=0, rtol=1e-5)

add_correction_term_to_dipole_correctors(line=line)

