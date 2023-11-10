import numpy as np
import xtrack as xt

line = xt.Line.from_json('lhc_thin.json')

s0 = 'mq.28r3.b1_entry'
s1 = 'mq.29r3.b1_exit'

s_cuts = np.linspace(line.get_s_position(s0), line.get_s_position(s1), 100)

s_tol = 0.5e-6

tt = line.get_table()

i_next = np.array([np.argmax(tt['s'] > s_cut) for s_cut in s_cuts])
i_ele_containing = i_next - 1

needs_cut = np.abs(tt['s'][i_ele_containing] - s_cuts) > s_tol

assert np.all(s_cuts[needs_cut] > tt['s'][i_ele_containing[needs_cut]])
assert np.all(s_cuts[needs_cut] < tt['s'][i_ele_containing[needs_cut]+1])