# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

import matplotlib.pyplot as plt

# Load a line and build tracker
line = xt.Line.from_json("../../test_data/hllhc15_thick/lhc_thick_with_knobs.json")
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)

# Twiss
tw = line.twiss4d()

pl = tw.plot(figlabel="tw")
pl = tw.plot(figlabel="tw")

fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
pl1 = tw.plot(ax=ax1)
pl2 = tw.plot(yl="mux muy", ax=ax2)
fig.subplots_adjust(right=0.7)
pl1.move_legend(1.5, 1.0)
pl2.move_legend(1.5, 1.0)
