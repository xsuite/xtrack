import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
line_b1 = xt.Line.from_dict(dct_b1)

context = xo.ContextCupy()
#context = xo.ContextCpu()
context = xo.ContextPyopencl()

tracker = line_b1.build_tracker(_context=context)
target_qx = 62.315
target_qy = 60.325

tw = tracker.twiss()

# Try to measure and match coupling
tracker.vars['cmrskew'] = 1e-3
tracker.vars['cmiskew'] = 1e-3

# Match coupling
tracker.match(verbose=True,
    vary=[
        xt.Vary(name='cmrskew', limits=[-0.5e-2, 0.5e-2], step=1e-5),
        xt.Vary(name='cmiskew', limits=[-0.5e-2, 0.5e-2], step=1e-5)],
    targets=[
        # xt.Target('c_r1_avg', 0, tol=2e-2),
        # xt.Target('c_r2_avg', 0, tol=2e-2),
        xt.Target('c_minus', 0, tol=1e-4),
        ])
