# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt

import matplotlib.pyplot as plt

#################################
# Load a line and build tracker #
#################################

line = xt.Line.from_json(
    '../../test_data/lhc_no_bb/line_and_particle.json')
line.particle_ref = xt.Particles(
                    mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

tw = line.twiss()
tw_htg = line.twiss(hide_thin_groups=True)

plt.close('all')
plt.plot(tw.s, tw.x, label='hide_thin_groups=False')
plt.plot(tw.s, tw_htg.x, label='hide_thin_groups=True')
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.legend()
plt.show()

for nn in ('x y px py zeta delta ptau '
           'betx bety alfx alfy gamx gamy dx dy dpx dpy').split():
    assert np.isnan(tw_htg[nn]).sum() == 2281
    assert np.isnan(tw[nn]).sum() == 0

    # Check in presence of a srotation
    assert tw.name[11197] == 'mbxws.1r8_pretilt'
    assert tw.name[11198] == 'mbxws.1r8'
    assert tw.name[11199] == 'mbxws.1r8_posttilt'

    assert tw_htg[nn][11197] == tw[nn][11197]
    assert np.isnan(tw_htg[nn][11198])
    assert np.isnan(tw_htg[nn][11199])
    assert tw_htg[nn][11200] == tw[nn][11200]

