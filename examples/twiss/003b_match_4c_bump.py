import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct)

line.build_tracker()
'''
mq.28l4.b1
acbv30.l4b1
acbv28.l4b1
acbv26.l4b1
acbv24.l4b1
'''

import pdb; pdb.set_trace()
line.match(verbose=True,
    vary=[
        xt.Vary(name='acbv32.l4b1', limits=[-1e-6, 1e-6], step=1e-10),
        xt.Vary(name='acbv30.l4b1', limits=[-1e-6, 1e-6], step=1e-10),
        xt.Vary(name='acbv26.l4b1', limits=[-1e-6, 1e-6], step=1e-10),
        xt.Vary(name='acbv24.l4b1', limits=[-1e-6, 1e-6], step=1e-10),
    ],
    targets=[
        # I want the orbit to be 1 mm at mq.28l4.b1 with zero angle
        xt.Target('y', at='mq.28l4.b1', value=1e-3, tol=1e-6),
        xt.Target('py', at='mq.28l4.b1', value=0, tol=1e-10),
        # I want the bump to be closed
        xt.Target('y', at='mq.30l4.b1', value=1e-3, tol=1e-6),
        xt.Target('y', at='mq.24l4.b1', value=1e-3, tol=1e-6),
    ]
)
