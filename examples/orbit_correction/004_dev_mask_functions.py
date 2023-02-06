# Need to:
# - have independent octupole knobs for the two beams
# - add knobs for optics correction


import json

import numpy as np
import xtrack as xt
import xobjects as xo
from scipy.constants import c as clight

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

# Switch on octupoles
beamn = 1
line.vars[f'p0c_b{beamn}'] = line.particle_ref.p0c[0]
line.vars[f'q0_b{beamn}'] = line.particle_ref.q0
line.vars[f'brho0_b{beamn}'] = (line.vars[f'p0c_b{beamn}']
                               / line.vars[f'q0_b{beamn}'] / clight)

line.vars[f'i_oct_b{beamn}'] = 0
for ss in '12 23 34 45 56 67 78 81'.split():
    line.vars[f'kof.a{ss}b{beamn}'] = (
        line.vars['kmax_mo']
        * line.vars[f'i_oct_b{beamn}'] / line.vars['imax_mo']
        / line.vars[f'brho0_b{beamn}'])
    line.vars[f'kod.a{ss}b{beamn}'] = (
        line.vars['kmax_mo']
        * line.vars[f'i_oct_b{beamn}'] / line.vars['imax_mo']
        / line.vars[f'brho0_b{beamn}'])

# Check against these references
i_ioct_ref = -235.
k_ref = -6.928009
knl_ref = -2.21696289

line.vars['i_oct_b1'] = i_ioct_ref
assert np.isclose(line.vars['kof.a12b1']._value, k_ref, atol=0, rtol=1e-5)
assert np.isclose(line.vars['kod.a45b1']._value, k_ref, atol=0, rtol=1e-5)
assert np.isclose(line['mo.24r2.b1'].knl[3], knl_ref, atol=0, rtol=1e-5)

