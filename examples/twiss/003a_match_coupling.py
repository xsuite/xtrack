import json

import numpy as np
import xtrack as xt

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
line_b1 = xt.Line.from_dict(dct_b1)

tracker = line_b1.build_tracker()

target_qx = 62.315
target_qy = 60.325

tw = tracker.twiss()

# Try to measure and match coupling
tracker.vars['cmrskew'] = 1e-3
tracker.vars['cmiskew'] = 1e-3

# Match coupling
tracker.match(vary=['cmrskew', 'cmiskew'],
    targets = [('c_minus', 0, 1e-4)])

####################
# Funky feature


tracker.vars['cmrskew'] = 7.628484268860683e-05
tracker.vars['cmiskew'] = 7.628484268860683e-05

cr_values = np.linspace(0.5e-4, 0.8e-4, 100)
c_minus_values = cr_values * 0

for ii, cr in enumerate(cr_values):
    print(f"cr = {cr}", end='\r', flush=True)
    tracker.vars['cmrskew'] = cr
    tw = tracker.twiss()
    c_minus_values[ii] = tw.c_minus

