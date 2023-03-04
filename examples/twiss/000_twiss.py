# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

###############
# Load a line #
###############

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
#fname_line_particles = '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json' #!skip-doc

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

#################
# Build tracker #
#################

line.build_tracker()

#########
# Twiss #
#########

tw = line.twiss()

#!end-doc-part

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
