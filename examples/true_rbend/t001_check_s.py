import xtrack as xt
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('rb', xt.RBend, length_straight=2, angle=np.pi/3, rbend_model='straight-body'),
])

line_thin = line.copy(shallow=True)
line_thin.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(3, mode='thin'))
    ]
)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=45.6e9)
p_thick = p0.copy()
p_thin = p0.copy()

line_thick_cut = line_thin.copy(shallow=True)
line_thick_cut.cut_at_s(line_thin.get_table().s)

line.reset_s_at_end_turn = False
line_thin.reset_s_at_end_turn = False
line.discard_tracker()
line_thin.discard_tracker()
line.build_tracker(compile=True, use_prebuilt_kernels=False)
line_thin.build_tracker(compile=True, use_prebuilt_kernels=False)
line.track(p_thick, num_turns=1)
line_thin.track(p_thin, num_turns=1)

line.set_particle_ref(p0.copy())
line_thin.set_particle_ref(p0.copy())
line_thick_cut.set_particle_ref(p0.copy())

tw_thick = line.twiss(betx=1, bety=1)
tw_thin = line_thin.twiss(betx=1, bety=1)
tw_thick_cut = line_thick_cut.twiss(betx=1, bety=1)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(tw_thick_cut.s, tw_thick_cut.x, '.-', label='thick cut')
plt.plot(tw_thin.s, tw_thin.x, 'x--', label='thin')
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.legend()
plt.show()