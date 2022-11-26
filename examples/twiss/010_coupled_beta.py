import numpy as np
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use('lhcb1')

tw_mad_no_coupling = mad.twiss(ripken=True).dframe()

# introduce coupling
mad.sequence.lhcb1.expanded_elements[7].ksl = [0,1e-3]

tw_mad_coupling = mad.twiss(ripken=True).dframe()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
line.particle_ref = xp.Particles(p0c=7000e9, mass0=xp.PROTON_MASS_EV)

tracker = line.build_tracker()

tw = tracker.twiss()

Ws = np.array(tw.W_matrix)

bety1 = Ws[:, 2, 0]**2 + Ws[:, 2, 1]**2
betx2 = Ws[:, 0, 2]**2 + Ws[:, 0, 3]**2

r1 = np.sqrt(bety1)/np.sqrt(tw.betx)
r2 = np.sqrt(betx2)/np.sqrt(tw.bety)

cmin = np.sqrt(r1*r2)*np.abs(np.mod(tw.qx, 1) - np.mod(tw.qy,1))/(1+r1*r2)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(211)
plt.plot(tw.s, bety1, label='bety1')
plt.plot(tw_mad_coupling.s, tw_mad_coupling.beta21, '--')
plt.ylabel(r'$\beta_{1,y}$')
plt.subplot(212, sharex=sp1)
plt.plot(tw.s, betx2, label='betx2')
plt.plot(tw_mad_coupling.s, tw_mad_coupling.beta12, '--')
plt.ylabel(r'$\beta_{2,x}$')

plt.show()