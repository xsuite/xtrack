import numpy as np

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xdeps as xd

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")

line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'], deferred_expressions=True)
line.cycle('ip1', inplace=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)
line.build_tracker()
tw = line.twiss()

line.vars['on_x1'] = 0

# Define the function
line.functions['ramp_on_sep1'] = xd.FunctionPieceWiseLinear(
    x=np.array([0, 5, 20, 40, 50,  60]) * 1e-3, # ms
    y=np.array([0, 0,  2,  2,  1, 0.5]) # knob value
)

# Drive the knob with the function
line.vars['on_sep1'] = line.functions.ramp_on_sep1(line.vars['t_turn_s'])

# Track a particle on the closed orbit
p = line.build_particles(x=0)
line.enable_time_dependent_vars = True
line.track(p, num_turns=1000, turn_by_turn_monitor=True)
mon = line.record_last_track



