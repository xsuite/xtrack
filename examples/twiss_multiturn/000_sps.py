import xtrack as xt

import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')

mad.input('beam, particle=proton, pc=26;')
v_mv = 7
mad.call('../../test_data/sps_thick/lhc_q20.str')
mad.use(sequence='sps')
mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

mad.sequence.sps.elements['actcse.31632'].volt = v_mv
mad.sequence.sps.elements['actcse.31632'].freq = 200
mad.sequence.sps.elements['actcse.31632'].lag = 0.5

line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                    q0=1, gamma0=mad.sequence.sps.beam.gamma)

# # Optionally thin
# Strategy = xt.slicing.Strategy
# Teapot = xt.slicing.Teapot
# line.slice_thick_elements(slicing_strategies=[
#     Strategy(slicing=Teapot(1)),  # Default
#     Strategy(slicing=Teapot(2), element_type=xt.Bend),
#     Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
# ])
# line.build_tracker()

tw = line.twiss()

opt = line.match(
    solve=False,
    vary=[
        xt.VaryList(['kqf', 'kqd'], step=1e-5, tag='tune'),
        xt.VaryList(['klsda', 'klsdb', 'klsfa', 'klsfb', 'klsfc'], step=1e-4, tag='chrom'),
    ],
    targets=[
        xt.TargetSet(qx=20.13, qy=20.251, tol=1e-7, tag='tune'),
        xt.TargetSet(dqx=2.0, dqy=2.0, tol=1, tag='chrom'),
        ],
)
opt.solve()

r0 = np.linspace(0, 3, 50)
p = line.build_particles(
    y_norm=r0*np.cos(np.pi/20.),
    py_norm=r0*np.sin(np.pi/20.),
    nemitt_x=1e-6, nemitt_y=1e-6)


line['lod.60702'].knl[3] = 5
line['lod.60702'].knl[2] = 10e-1

line.track(p, num_turns=1000, turn_by_turn_monitor=True)
mon = line.record_last_track

# Compute
# p_co_guess = line.build_particles(y=2e-3)
tw_mt = line.twiss(co_guess={'y': 2e-3}, num_turns=4)

# Inspect and plot
tw_start_turns = tw_mt.rows['_turn_.*']
tw_start_turns.show()
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(mon.y.flatten(), mon.py.flatten(), '.', markersize=1)
plt.plot(tw_start_turns.y, tw_start_turns.py, '*r')
plt.show()
