import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct)

line.build_tracker()

tw_before = line.twiss()

line.match(
    solver='jacobian',
    # Portion of the beam line to be modified and initial conditions
    ele_start='mq.33l8.b1',
    ele_stop='mq.21l8.b1',
    twiss_init='preserve',
    # Dipole corrector strengths to be varied
    vary=[
        xt.Vary(name='acbv32.l8b1', step=1e-10),
        xt.Vary(name='acbv28.l8b1', step=1e-10),
        xt.Vary(name='acbv26.l8b1', step=1e-10),
        xt.Vary(name='acbv24.l8b1', step=1e-10),
        xt.Vary(name='acbv22.l8b1', step=1e-10),
    ],
    targets=[
        # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
        xt.Target('y', at='mb.b26l8.b1', value=3e-3, tol=1e-4, scale=1),
        xt.Target('py', at='mb.b26l8.b1', value=0, tol=1e-6, scale=1000),
        # I want the bump to be closed
        xt.Target('y', at='mq.21l8.b1', value='preserve', tol=1e-6, scale=1),
        xt.Target('py', at='mq.21l8.b1', value='preserve', tol=1e-7, scale=1000),
    ]
)

#!end-doc-part

tw = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.2, 4.8*0.8))
ax = fig.add_subplot(111)
ax.plot(tw_before.s, tw_before.y*1000, label='y')
ax.plot(tw.s, tw.y*1000, label='y')
# Target
ax.axvline(x=line.get_s_position('mb.b26l8.b1'), color='r', linestyle='--', alpha=0.5)
# Correctors
ax.axvline(x=line.get_s_position('mcbv.32l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.28l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.26l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.24l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.22l8.b1'), color='k', linestyle='--', alpha=0.5)
# Boundaries
ax.axvline(x=line.get_s_position('mq.33l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mq.21l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.set_xlim(line.get_s_position('mq.33l8.b1') - 10,
            line.get_s_position('mq.21l8.b1') + 10)
ax.set_xlabel('s [m]')
ax.set_ylabel('y [mm]')
ax.set_ylim(-10, 10)
plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)
plt.show()
