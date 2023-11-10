import numpy as np
import xtrack as xt

line = xt.Line.from_json('lhc_thin.json')

s0 = 'mq.28r3.b1_entry'
s1 = 'mq.29r3.b1_exit'

s_cuts = np.linspace(line.get_s_position(s0), line.get_s_position(s1), 100)

s_tol = 0.5e-6

tt = line.get_table()

