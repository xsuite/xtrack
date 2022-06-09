import sys
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

from cpymad.madx import Madx

path = '../../test_data/hllhc14_input_mad/'

mad = Madx(command_log="mad_final.log")
mad.call(path + "final_seq.madx")
mad.use(sequence="lhcb1")
mad.twiss()
mad.readtable(file=path + "final_errors.tfs", table="errtab")
mad.seterr(table="errtab")
mad.set(format=".15g")
twmad = mad.twiss(rmatrix=True, chrom=True)

line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=True)
part_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

context = xo.ContextCpu()
tracker = xt.Tracker(_context=context, line=line)

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

assert np.isclose(mad.table.summ.q1[0], twxt['qx'], rtol=1e-4)
assert np.isclose(mad.table.summ.q2[0], twxt['qy'], rtol=1e-4)
assert np.isclose(mad.table.summ.dq1, twxt['dqx'], atol=0.1, rtol=0)
assert np.isclose(mad.table.summ.dq2, twxt['dqy'], atol=0.1, rtol=0)

for name in ['mb.b19r5.b1', 'mb.b19r1.b1', 'ip1', 'ip2', 'ip5', 'ip8',
             'mbxf.4l1', 'mbxf.4l5']:
    imad = list(twmad['name']).index(name+':1')
    ixt = list(twxt['name']).index(name)

    for sp in [spbet, spco, spdisp]:
        sp.axvline(x=twxt['s'][imad])

    assert np.isclose(twxt['betx'][ixt], twmad['betx'][imad],
                      atol=0, rtol=3e-4)
    assert np.isclose(twxt['bety'][ixt], twmad['bety'][imad],
                      atol=0, rtol=3e-4)
    assert np.isclose(twxt['alfx'][ixt], twmad['alfx'][imad],
                      atol=1e-1, rtol=0)
    assert np.isclose(twxt['alfy'][ixt], twmad['alfy'][imad],
                      atol=1e-1, rtol=0)
    assert np.isclose(twxt['dx'][ixt], twmad['dx'][imad], atol=1e-2)
    assert np.isclose(twxt['dy'][ixt], twmad['dy'][imad], atol=1e-2)
    assert np.isclose(twxt['dpx'][ixt], twmad['dpx'][imad], atol=3e-4)
    assert np.isclose(twxt['dpy'][ixt], twmad['dpy'][imad], atol=3e-4)

    assert np.isclose(twxt['s'][ixt], twmad['s'][imad], atol=5e-6)
    assert np.isclose(twxt['x'][ixt], twmad['x'][imad], atol=5e-6)
    assert np.isclose(twxt['y'][ixt], twmad['y'][imad], atol=5e-6)
    assert np.isclose(twxt['px'][ixt], twmad['px'][imad], atol=1e-7)
    assert np.isclose(twxt['py'][ixt], twmad['py'][imad], atol=1e-7)

plt.show()
