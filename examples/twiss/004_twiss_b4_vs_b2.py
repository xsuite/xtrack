import numpy as np
from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad_b2 = Madx()
mad_b2.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad_b2.use(sequence='lhcb2')
twb2mad = mad_b2.twiss()
summb2mad = mad_b2.table.summ

mad_b4 = Madx()
mad_b4.call('../../test_data/hllhc15_noerrors_nobb/sequence_b4.madx')
mad_b4.use(sequence='lhcb2')
twb4mad = mad_b4.twiss()
summb4mad = mad_b4.table.summ

line_b4 = xt.Line.from_madx_sequence(mad_b4.sequence['lhcb2'],
                                     deferred_expressions=True)
line_b4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

line_b4.build_tracker()
twb4xt = line_b4.twiss()

twb2xt = line_b4.twiss(reverse=True)

# Compute twiss also from W matrix to check the W matrix
Ws = np.array(twb2xt.W_matrix)
betx = Ws[:, 0, 0]**2 + Ws[:, 0, 1]**2
bety = Ws[:, 2, 2]**2 + Ws[:, 2, 3]**2
gamx = Ws[:, 1, 0]**2 + Ws[:, 1, 1]**2
gamy = Ws[:, 3, 2]**2 + Ws[:, 3, 3]**2
alfx = - Ws[:, 0, 0] * Ws[:, 1, 0] - Ws[:, 0, 1] * Ws[:, 1, 1]
alfy = - Ws[:, 2, 2] * Ws[:, 3, 2] - Ws[:, 2, 3] * Ws[:, 3, 3]

import matplotlib.pyplot as plt

plt.close('all')

fig1 = plt.figure(1)
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(twb2mad['s'], twb2mad['betx'], 'b')
spbet.plot(twb2xt['s'], twb2xt['betx'], '--', color='lightblue')
spbet.plot(twb2mad['s'], twb2mad['bety'], 'r')
spbet.plot(twb2xt['s'], twb2xt['bety'], '--', color='darkred')

spco.plot(twb2mad['s'], twb2mad['x'], 'b')
spco.plot(twb2xt['s'], twb2xt['x'], '--', color='lightblue')
spco.plot(twb2mad['s'], twb2mad['y'], 'r')
spco.plot(twb2xt['s'], twb2xt['y'], '--', color='darkred')

spdisp.plot(twb2mad['s'], twb2mad['dx'], 'b')
spdisp.plot(twb2xt['s'], twb2xt['dx'], '--', color='lightblue')
spdisp.plot(twb2mad['s'], twb2mad['dy'], 'r')
spdisp.plot(twb2xt['s'], twb2xt['dy'], '--', color='darkred')

assert np.isclose(summb2mad.q1[0], twb2xt['qx'], rtol=1e-4)
assert np.isclose(summb2mad.q2[0], twb2xt['qy'], rtol=1e-4)
assert np.isclose(summb2mad.dq1, twb2xt['dqx'], atol=0.1, rtol=0)
assert np.isclose(summb2mad.dq2, twb2xt['dqy'], atol=0.1, rtol=0)
assert np.isclose(summb2mad.alfa[0],
    twb2xt['momentum_compaction_factor'], atol=7e-8, rtol=0)
assert np.isclose(twb2xt['qs'], 0.0021, atol=1e-4, rtol=0)

for name in ['mb.b19r5.b2', 'mb.b19r1.b2', 'ip1', 'ip2', 'ip5', 'ip8',
             'mbxf.4l1', 'mbxf.4l5']:
    imad = list(twb2mad['name']).index(name+':1')
    ixt = list(twb2xt['name']).index(name) + 1 # MAD measures at exit

    for sp in [spbet, spco, spdisp]:
        sp.axvline(x=twb2xt['s'][imad])

    assert np.isclose(twb2xt['betx'][ixt], twb2mad['betx'][imad],
                      atol=0, rtol=3e-4)
    assert np.isclose(betx[ixt], twb2mad['betx'][imad], atol=0, rtol=3e-4)
    assert np.isclose(twb2xt['bety'][ixt], twb2mad['bety'][imad],
                      atol=0, rtol=3e-4)
    assert np.isclose(bety[ixt], twb2mad['bety'][imad], atol=0, rtol=3e-4) 
    assert np.isclose(twb2xt['alfx'][ixt], twb2mad['alfx'][imad],
                      atol=1e-1, rtol=0)
    assert np.isclose(alfx[ixt], twb2mad['alfx'][imad], atol=1e-1, rtol=0)
    assert np.isclose(twb2xt['alfy'][ixt], twb2mad['alfy'][imad],
                      atol=1e-1, rtol=0)
    assert np.isclose(alfy[ixt], twb2mad['alfy'][imad], atol=1e-1, rtol=0)
    assert np.isclose(twb2xt['dx'][ixt], twb2mad['dx'][imad], atol=1e-2)
    assert np.isclose(twb2xt['dy'][ixt], twb2mad['dy'][imad], atol=1e-2)
    assert np.isclose(twb2xt['dpx'][ixt], twb2mad['dpx'][imad], atol=3e-4)
    assert np.isclose(twb2xt['dpy'][ixt], twb2mad['dpy'][imad], atol=3e-4)

    assert np.isclose(twb2xt['s'][ixt], twb2mad['s'][imad], atol=5e-6)
    assert np.isclose(twb2xt['x'][ixt], twb2mad['x'][imad], atol=5e-6)
    assert np.isclose(twb2xt['y'][ixt], twb2mad['y'][imad], atol=5e-6)
    assert np.isclose(twb2xt['px'][ixt], twb2mad['px'][imad], atol=1e-7)
    assert np.isclose(twb2xt['py'][ixt], twb2mad['py'][imad], atol=1e-7)

    assert np.isclose(twb2xt['mux'][ixt], twb2mad['mux'][imad], atol=1e-7)
    assert np.isclose(twb2xt['muy'][ixt], twb2mad['muy'][imad], atol=1e-7)

plt.show()