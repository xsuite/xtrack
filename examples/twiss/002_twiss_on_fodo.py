# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
import xpart as xp

q1 = xt.Multipole(knl=[0, 2e-2])
q2 = xt.Multipole(knl=[0, -2e-2])

line = xt.Line(
        elements=[
        xt.Drift(length=1),
        xt.Cavity(frequency=400e6, voltage=1e6, lag=0),
        q1,
        xt.Drift(length=1),
        xt.Multipole(knl=[1e-2], hxl=1e-2),
        q2,
        ])

line.build_tracker(global_xy_limit=1e10)

line.particle_ref = xp.Particles(p0c=6500e9)
import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
fig2 = plt.figure()
axe = plt.subplot()
eigs = []
factors = np.linspace(80, 120, 100)
for factor in factors:
    q1.knl[1] = factor*0.020
    q2.knl[1] = -factor*0.020
    line.track(xp.Particles(p0c=6500e9, x=0.01, y=0.01),
                  num_turns=5000, turn_by_turn_monitor=True)
    mon = line.record_last_track
    RR = line.compute_one_turn_matrix_finite_differences(
                                               particle_on_co=line.particle_ref)
    ax1.semilogy(np.abs(mon.x.T), label=f'{factor}, {np.trace(RR)}')
    ax2.semilogy(np.abs(mon.y.T), label=f'{factor}, {np.trace(RR)}')

    w0, v0 = np.linalg.eig(xt.linear_normal_form.healy_symplectify(RR))
    eigs.append(w0)
    #axe.plot(w0.real, w0.imag, '.')

eigs = np.array(eigs)

plt.legend(loc='upper left')

plt.figure()
for ii in range(6):
    plt.plot(factors, np.abs(eigs[:, ii]).T, '.')
plt.show()
