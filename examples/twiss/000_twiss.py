# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp

#################################
# Load a line and build tracker #
#################################

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

#########
# Twiss #
#########

tw = line.twiss()

#!end-doc-part
tw_sel = tw[['ip6', 'ip5'], :]

try:
    tw['ip0', :]
except IndexError: # expected exception
    pass
else:
    raise Exception('Expected exception not raised')

try:
    tw[['ip1', 'ip2', 'ip0'], :]
except IndexError: # expected exception
    pass

# this does not give a clear error message
# tw['ip.*']

tw[['ip1', 'ip2'], :]
tw['ip.*', :]



# Test custom s locations
s_test = [2e3, 1e3, 3e3, 10e3]
twats = line.twiss(at_s = s_test)
for ii, ss in enumerate(s_test):
    assert np.isclose(twats['s'][ii], ss, rtol=0, atol=1e-14)
    i_prev = np.where(tw['s']<=ss)[0][-1]
    assert np.isclose(twats['alfx'][ii], np.interp(ss, tw['s'], tw['alfx']),
                      rtol=0, atol=1e-9)
    assert np.isclose(twats['alfy'][ii], np.interp(ss, tw['s'], tw['alfy']),
                     rtol=0, atol=1e-9)
    assert np.isclose(twats['dpx'][ii], np.interp(ss, tw['s'], tw['dpx']),
                      rtol=0, atol=1e-9)
    assert np.isclose(twats['dpy'][ii], np.interp(ss, tw['s'], tw['dpy']),
                      rtol=0, atol=1e-9)


twmb19r5 = tw.get_twiss_init(at_element='mb.b19l5.b1')

tw_part = line.twiss(ele_start='mb.b19l5.b1', ele_stop='mb.b19r5.b1',
                        twiss_init=twmb19r5)



import matplotlib.pyplot as plt

plt.close('all')
plt.ion()

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw['s'], tw['betx'])
spbet.plot(tw['s'], tw['bety'])

spco.plot(tw['s'], tw['x'])
spco.plot(tw['s'], tw['y'])

spdisp.plot(tw['s'], tw['dx'])
spdisp.plot(tw['s'], tw['dy'])

spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

fig1.suptitle(
    r'$q_x$ = ' f'{tw["qx"]:.5f}' r' $q_y$ = ' f'{tw["qy"]:.5f}' '\n'
    r"$Q'_x$ = " f'{tw["dqx"]:.2f}' r" $Q'_y$ = " f'{tw["dqy"]:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(tw["momentum_compaction_factor"]):.2f}'
)

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()
