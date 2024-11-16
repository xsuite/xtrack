# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
from scipy.constants import c as clight

T_rev = 23e-6 # 23 us

line = xt.Line.from_json('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')
line.build_tracker()

tt = line.get_table(attr=True)
tt_cav = tt.rows[tt.element_type=='Cavity']
tt_cav.cols['frequency voltage lag'].show()

line['acta.31637'].absolute_time = 1
line['acta.31637'].frequency = 200.266e6 + 1e3
line['acta.31637'].voltage = 3e6

# line['actb.31739'].absolute_time = 1
# line['actb.31739'].frequency = 200.266e6 - 1e3
# line['actb.31739'].voltage = 3e6

# tw = line.twiss(search_for_t_rev=True)

# particles = xt.Particles(p0c=26e9, zeta=np.linspace(-1, 1, 40), delta=tw.delta[0])
particles = xt.Particles(p0c=26e9, delta=np.linspace(-7e-3, 7e-3, 1000))

particles.t_sim = line.get_length() / line.particle_ref._xobject.beta0[0] / clight
line.track(particles, num_turns=1000, turn_by_turn_monitor=True, with_progress=10)



rec = line.record_last_track
import matplotlib.pyplot as plt


plt.close('all')
plt.figure(1)
for ii in range(rec.x.shape[0]):
    mask = rec.state[ii, :]>0
    plt.plot(rec.zeta[ii, mask], rec.delta[ii, mask])

plt.grid(linestyle='--')
plt.xlabel('z [m]')
plt.ylabel(r'$\Delta p / p_0$')

plt.figure(2)
plt.plot(rec.zeta.T)
plt.show()
