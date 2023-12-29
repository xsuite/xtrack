import numpy as np
from cpymad.madx import Madx

import xtrack as xt

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use('lhcb1')

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
line.particle_ref = xt.Particles(p0c=7000e9, mass0=xt.PROTON_MASS_EV)

line.build_tracker()

tw = line.twiss()

delta_values = np.linspace(-5e-3, 5e-3, 100)

qx_values = delta_values * 0
qy_values = delta_values * 0
for i, delta in enumerate(delta_values):
    print(f'Xsuite working on {i} of {len(delta_values)}  ', end='\r', flush=True)
    tt = line.twiss(method='4d', delta0=delta)

    qx_values[i] = tt.qx
    qy_values[i] = tt.qy

#!end-doc-part

qx_values_mad = delta_values * 0
qy_values_mad = delta_values * 0
for i, delta in enumerate(delta_values):
    print(f'Working on {i} of {len(delta_values)}   ' , end='\r', flush=True)
    mad.input(f'twiss, deltap={delta};')
    qx_values_mad[i] = mad.table.summ.q1
    qy_values_mad[i] = mad.table.summ.q2

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
plt.subplot(211)
plt.plot(delta_values, qx_values, label='xtrack')
plt.plot(delta_values, qx_values_mad, label='madx')
plt.ylabel(r'$Q_x$')

plt.subplot(212)
plt.plot(delta_values, qy_values, label='xtrack')
plt.plot(delta_values, qy_values_mad, label='madx')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$Q_y$')
plt.legend()

# Figure with Xsuite only
fig2 = plt.figure(2)
plt.subplot(211)
plt.plot(delta_values, qx_values, '.-', label='xtrack')
plt.ylabel(r'$Q_x$')

plt.subplot(212)
plt.plot(delta_values, qy_values, '.-', label='xtrack')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$Q_y$')

plt.show()



