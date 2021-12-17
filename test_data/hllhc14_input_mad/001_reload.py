import sys
import numpy as np

import pymask as pm
import xtrack as xt
import xpart as xp

Madx = pm.Madxp
mad = Madx(command_log="mad_final.log")
mad.call("final_seq.madx")
mad.use(sequence="lhcb1")
mad.twiss()
mad.readtable(file="final_errors.tfs", table="errtab")
mad.seterr(table="errtab")
mad.set(format=".15g")
twmad = mad.twiss(rmatrix = True)

line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=True)
part_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

tracker = xt.Tracker(line=line)

twxt = tracker.twiss(particle_ref=part_ref)



import matplotlib.pyplot as plt

plt.close('all')

fig1 = plt.figure(1)
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(twmad['s'], twmad['betx'], 'b')
spbet.plot(twxt['s'], twxt['betx'], '--', color='lightblue')
spbet.plot(twmad['s'], twmad['bety'], 'r')
spbet.plot(twxt['s'], twxt['bety'], '--', color='darkred')

spco.plot(twmad['s'], twmad['x'], 'b')
spco.plot(twxt['s'], twxt['x'], '--', color='lightblue')
spco.plot(twmad['s'], twmad['y'], 'r')
spco.plot(twxt['s'], twxt['y'], '--', color='darkred')

spdisp.plot(twmad['s'], twmad['dx'], 'b')
spdisp.plot(twxt['s'], twxt['dx'], '--', color='lightblue')
spdisp.plot(twmad['s'], twmad['dy'], 'r')
spdisp.plot(twxt['s'], twxt['dy'], '--', color='darkred')

for name in ['mb.b19r5.b1', 'mb.b19r1.b1']:
    imad = list(twmad['name']).index(name+':1')
    ixt = list(twxt['name']).index(name)

    for sp in [spbet, spco, spdisp]:
        sp.axvline(x=twxt['s'][imad])

    assert np.isclose(twxt['betx'][ixt], twmad['betx'][imad],
                      atol=0, rtol=3e-4)
    assert np.isclose(twxt['bety'][ixt], twmad['bety'][imad],
                      atol=0, rtol=3e-4)
    assert np.isclose(twxt['dx'][ixt], twmad['dx'][imad], rtol=3e-4, atol=5e-3)
    assert np.isclose(twxt['dy'][ixt], twmad['dy'][imad], rtol=3e-4, atol=5e-3)
    assert np.isclose(twxt['dpx'][ixt], twmad['dpx'][imad], rtol=3e-4, atol=0.5e-4)
    assert np.isclose(twxt['dpy'][ixt], twmad['dpy'][imad], rtol=3e-4, atol=0.5e-4)



plt.show()
