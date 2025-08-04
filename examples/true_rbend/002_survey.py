import xtrack as xt
import numpy as np
import xobjects as xo

# TODO
#  - check backtrack
#  - survey
#  - properties
#  - linear edge
# -  test sps

edge_model = 'full' # Remember to test both!!!!!!

env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

line = env.new_line(length=5, components=[
    env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
            rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
            at=2.5)])

line.cut_at_s(np.linspace(0, line.get_length(), 11))
line.insert('mid', xt.Marker(), at=2.5)

line['mb'].rbend_model = 'straight-body'
sv_straight = line.survey(element0='mid')
tt_straight = line.get_table(attr=True)

line['mb'].rbend_model = 'curved-body'
sv_curved = line.survey(element0='mid')
tt_curved = line.get_table(attr=True)

import matplotlib.pyplot as plt
plt.close('all')
sv_straight.plot()
plt.suptitle('Straight body')

sv_curved.plot()
plt.suptitle('Curved body')

plt.show()