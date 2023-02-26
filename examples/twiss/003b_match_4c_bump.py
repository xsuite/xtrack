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

tw_before = line.twiss()

res1 =line.match(
    ele_start='mq.33l8.b1',
    ele_stop='mq.23l8.b1',
    twiss_init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
    vary=[
        xt.Vary(name='acbv30.l8b1', limits=[-500e-6, 500e-6], step=1e-10),
        xt.Vary(name='acbv28.l8b1', limits=[-500e-6, 500e-6], step=1e-10),
        xt.Vary(name='acbv26.l8b1', limits=[-500e-6, 500e-6], step=1e-10),
        xt.Vary(name='acbv24.l8b1', limits=[-500e-6, 500e-6], step=1e-10),
    ],
    targets=[
        # I want the orbit to be 3 mm at mq.28l8.b1 with zero angle
        xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-5, scale=1),
        xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-7, scale=1000),
        # I want the bump to be closed
        xt.Target('y', at='mq.23l8.b1', value=0, tol=1e-5, scale=1),
        xt.Target('py', at='mq.23l8.b1', value=0, tol=1e-7, scale=1000),
    ]
)

# res2 =line.match(verbose=True,
#     ele_start='mq.33l8.b1',
#     ele_stop='mq.23l8.b1',
#     twiss_init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
#     vary=[
#         #xt.Vary(name='acbv32.l8b1', limits=[-100e-6, 100e-6], step=1e-10),
#         #xt.Vary(name='acbv30.l8b1', limits=[-100e-6, 100e-6], step=1e-10),

#     ],
#     targets=[
#         # I want the orbit to be 1 mm at mq.28l8.b1 with zero angle
#         #xt.Target('y', at='mb.b28l8.b1', value=1e-3, tol=1e-5, scale=1),
#         #xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-7, scale=1000),
#         # I want the bump to be closed

#     ]
# )

tw = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
plt.plot(tw.s, tw.y, label='y')
plt.axvline(x=line.get_s_position('mb.b28l8.b1'), color='k')
plt.axvline(x=line.get_s_position('mcbv.30l8.b1'), color='k', linestyle='--', alpha=0.5)
plt.axvline(x=line.get_s_position('mcbv.28l8.b1'), color='k', linestyle='--', alpha=0.5)
plt.axvline(x=line.get_s_position('mcbv.26l8.b1'), color='k', linestyle='--', alpha=0.5)
plt.axvline(x=line.get_s_position('mcbv.24l8.b1'), color='k', linestyle='--', alpha=0.5)
plt.axvline(x=line.get_s_position('mq.33l8.b1'), color='g', linestyle='--', alpha=0.5)
plt.axvline(x=line.get_s_position('mq.23l8.b1'), color='g', linestyle='--', alpha=0.5)
plt.show()