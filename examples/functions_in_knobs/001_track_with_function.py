import numpy as np

from cpymad.madx import Madx
import xtrack as xt
import xdeps as xd

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=7e12)
line.build_tracker()

# Cycle the line to the middle of the bump
line.cycle('ip1', inplace=True)
line.build_tracker()

line.vars['on_x1'] = 0
tw = line.twiss(method='4d')

# Define the function
line.functions['f_on_sep1'] = xd.FunctionPieceWiseLinear(
    x=np.array([0, 5, 20, 40, 50]) * 1e-3, # ms
    y=np.array([0, 0,  2,  2,  1]) # knob value
)

# Drive the knob with the function
line.vars['on_sep1'] = line.functions.f_on_sep1(line.vars['t_turn_s'])

# Track a particle on the closed orbit
p = line.build_particles(x=0)
line.enable_time_dependent_vars = True
line.track(p, num_turns=1000, turn_by_turn_monitor=True)
mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(mon.at_turn.T * tw.T_rev0*1e3, mon.y.T)
plt.xlabel('time [ms]')
plt.ylabel('y [m]')
plt.show()



