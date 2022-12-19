# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import sys
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

from cpymad.madx import Madx

# path = '../../test_data/hllhc14_input_mad/'
# mad = Madx(command_log="mad_final.log")
# mad.call(path + "final_seq.madx")
# mad.use(sequence="lhcb1")
# mad.twiss()
# mad.readtable(file=path + "final_errors.tfs", table="errtab")
# mad.seterr(table="errtab")
# mad.set(format=".15g")

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use('lhcb1')

# I want only the betatron part in the sigma matrix
mad.sequence.lhcb1.beam.sigt = 1e-10
mad.sequence.lhcb1.beam.sige = 1e-10
mad.sequence.lhcb1.beam.et = 1e-10

twmad = mad.twiss(rmatrix=True, chrom=True)

line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

context = xo.ContextCpu()
tracker = xt.Tracker(_context=context, line=line)

twxt = tracker.twiss()

import matplotlib.pyplot as plt

plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(twmad['s'], twmad['betx'], 'b', label=r'madx $\beta_x$')
spbet.plot(twxt['s'], twxt['betx'], '--', color='lightblue', label=r'xtrack $\beta_x$')
spbet.plot(twmad['s'], twmad['bety'], 'r', label=r'madx $\beta_y$')
spbet.plot(twxt['s'], twxt['bety'], '--', color='darkred', label=r'xtrack $\beta_y$')
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spbet.legend(loc='upper right')

spco.plot(twmad['s'], twmad['x'], 'b')
spco.plot(twxt['s'], twxt['x'], '--', color='lightblue')
spco.plot(twmad['s'], twmad['y'], 'r')
spco.plot(twxt['s'], twxt['y'], '--', color='darkred')
spco.set_ylabel(r'$x,y$ [m]')

spdisp.plot(twmad['s'], twmad['dx'], 'b')
spdisp.plot(twxt['s'], twxt['dx'], '--', color='lightblue')
spdisp.plot(twmad['s'], twmad['dy'], 'r')
spdisp.plot(twxt['s'], twxt['dy'], '--', color='darkred')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel(r'$s$ [m]')

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
Sigmas = twxt.get_betatron_sigmas(nemitt_x, nemitt_y)

assert np.isclose(mad.table.summ.q1[0], twxt['qx'], rtol=1e-4)
assert np.isclose(mad.table.summ.q2[0], twxt['qy'], rtol=1e-4)
assert np.isclose(mad.table.summ.dq1, twxt['dqx'], atol=0.1, rtol=0)
assert np.isclose(mad.table.summ.dq2, twxt['dqy'], atol=0.1, rtol=0)
assert np.isclose(mad.table.summ.alfa[0],
    twxt['momentum_compaction_factor'], atol=1e-8, rtol=0)
assert np.isclose(twxt['qs'], 0.0021, atol=1e-4, rtol=0)

for name in ['mb.b19r5.b1', 'mb.b19r1.b1', 'ip1', 'ip2', 'ip5', 'ip8',
             'mbxf.4l1', 'mbxf.4l5']:
    imad = list(twmad['name']).index(name+':1')
    ixt = list(twxt['name']).index(name)

    # for sp in [spbet, spco, spdisp]:
    #     sp.axvline(x=twxt['s'][imad])

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

    assert np.isclose(Sigmas.Sigma11[ixt], twmad['sig11'][imad], atol=5e-10)
    assert np.isclose(Sigmas.Sigma12[ixt], twmad['sig12'][imad], atol=1e-12)
    assert np.isclose(Sigmas.Sigma13[ixt], twmad['sig13'][imad], atol=1e-10)
    assert np.isclose(Sigmas.Sigma14[ixt], twmad['sig14'][imad], atol=1e-12)
    assert np.isclose(Sigmas.Sigma22[ixt], twmad['sig22'][imad], atol=1e-12)
    assert np.isclose(Sigmas.Sigma23[ixt], twmad['sig23'][imad], atol=1e-12)
    assert np.isclose(Sigmas.Sigma24[ixt], twmad['sig24'][imad], atol=1e-12)
    assert np.isclose(Sigmas.Sigma33[ixt], twmad['sig33'][imad], atol=5e-10)
    assert np.isclose(Sigmas.Sigma34[ixt], twmad['sig34'][imad], atol=3e-12)
    assert np.isclose(Sigmas.Sigma44[ixt], twmad['sig44'][imad], atol=1e-12)

    # check matrix is symmetric
    assert np.isclose(Sigmas.Sigma12[ixt], Sigmas.Sigma21[ixt], atol=1e-16)
    assert np.isclose(Sigmas.Sigma13[ixt], Sigmas.Sigma31[ixt], atol=1e-16)
    assert np.isclose(Sigmas.Sigma14[ixt], Sigmas.Sigma41[ixt], atol=1e-16)
    assert np.isclose(Sigmas.Sigma23[ixt], Sigmas.Sigma32[ixt], atol=1e-16)
    assert np.isclose(Sigmas.Sigma24[ixt], Sigmas.Sigma42[ixt], atol=1e-16)
    assert np.isclose(Sigmas.Sigma34[ixt], Sigmas.Sigma43[ixt], atol=1e-16)

    # check matrix consistency with Sigma.Sigma
    assert np.isclose(Sigmas.Sigma11[ixt], Sigmas.Sigma[ixt][0, 0], atol=1e-16)
    assert np.isclose(Sigmas.Sigma12[ixt], Sigmas.Sigma[ixt][0, 1], atol=1e-16)
    assert np.isclose(Sigmas.Sigma13[ixt], Sigmas.Sigma[ixt][0, 2], atol=1e-16)
    assert np.isclose(Sigmas.Sigma14[ixt], Sigmas.Sigma[ixt][0, 3], atol=1e-16)
    assert np.isclose(Sigmas.Sigma21[ixt], Sigmas.Sigma[ixt][1, 0], atol=1e-16)
    assert np.isclose(Sigmas.Sigma22[ixt], Sigmas.Sigma[ixt][1, 1], atol=1e-16)
    assert np.isclose(Sigmas.Sigma23[ixt], Sigmas.Sigma[ixt][1, 2], atol=1e-16)
    assert np.isclose(Sigmas.Sigma24[ixt], Sigmas.Sigma[ixt][1, 3], atol=1e-16)
    assert np.isclose(Sigmas.Sigma31[ixt], Sigmas.Sigma[ixt][2, 0], atol=1e-16)
    assert np.isclose(Sigmas.Sigma32[ixt], Sigmas.Sigma[ixt][2, 1], atol=1e-16)
    assert np.isclose(Sigmas.Sigma33[ixt], Sigmas.Sigma[ixt][2, 2], atol=1e-16)
    assert np.isclose(Sigmas.Sigma34[ixt], Sigmas.Sigma[ixt][2, 3], atol=1e-16)
    assert np.isclose(Sigmas.Sigma41[ixt], Sigmas.Sigma[ixt][3, 0], atol=1e-16)
    assert np.isclose(Sigmas.Sigma42[ixt], Sigmas.Sigma[ixt][3, 1], atol=1e-16)
    assert np.isclose(Sigmas.Sigma43[ixt], Sigmas.Sigma[ixt][3, 2], atol=1e-16)
    assert np.isclose(Sigmas.Sigma44[ixt], Sigmas.Sigma[ixt][3, 3], atol=1e-16)

    # Check sigma_x, sigma_y
    assert np.isclose(Sigmas.sigma_x[ixt], np.sqrt(Sigmas.Sigma11[ixt]), atol=1e-16)
    assert np.isclose(Sigmas.sigma_y[ixt], np.sqrt(Sigmas.Sigma33[ixt]), atol=1e-16)



plt.show()
