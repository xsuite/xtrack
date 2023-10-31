import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct)
line.cycle('ip1', inplace=True)

line.build_tracker()

tw_before = line.twiss()

Range = xt.Range
GreaterThan = xt.GreaterThan
LessThan = xt.LessThan

opt = line.match(
    solve=False,
    ele_start='ip1', ele_stop='ip1.l1',
    twiss_init='preserve',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8),
    ],
    targets=[
        xt.TargetRelPhaseAdvance('mux', Range(62.26, 62.27),
                                 ele_start='ip1', ele_stop='ip1.l1', tol=1e-4),
        xt.TargetRelPhaseAdvance('muy', LessThan(60.28),
                                  ele_start='ip1', ele_stop='ip1.l1', tol=1e-4),
    ]
)

opt.target_status()

opt.targets[0].freeze()
opt.target_status()
opt.solve()

tt = line.twiss()
assert np.isclose(tt['mux'][-1], 62.31, atol=1e-4)
assert tt['muy'][-1] < 60.281

opt.targets[0].unfreeze()
opt.target_status()
opt.solve()
tt = line.twiss()
assert tt.mux[-1] > 62.26 - 1e-3
assert tt.mux[-1] < 62.27 + 1e-3
assert tt.muy[-1] < 60.281


#!end-doc-part
