import time
import numpy as np
import xtrack as xt

line = xt.Line.from_json('lhc_thin.json')
tw0 = line.twiss()

line.discard_tracker()

e0 = 'mq.28r3.b1_entry'
e1 = 'mq.29r3.b1_exit'

s0 = line.get_s_position(e0)
s1 = line.get_s_position(e1)

# elements_to_insert = [
#     # s .    # elements to insert (name, element)
#     (s0,     [(f'm0_at_a', xt.Marker()), (f'm1_at_a', xt.Marker()), (f'm2_at_a', xt.Marker())]),
#     (s0+10., [(f'm0_at_b', xt.Marker()), (f'm1_at_b', xt.Marker()), (f'm2_at_b', xt.Marker())]),
#     (s1,     [(f'm0_at_c', xt.Marker()), (f'm1_at_c', xt.Marker()), (f'm2_at_c', xt.Marker())]),
# ]


s_input = np.linspace(0, 26000, 3000)
elements_to_insert = [
    # s .    # elements to insert (name, element)
    (ss,     [(f'm0_at_{ii}', xt.Marker()), (f'm1_at_{ii}', xt.Marker())]) for ii, ss in enumerate(s_input)
]

t1 = time.time()
line._insert_thin_elements_at_s(elements_to_insert)
t2 = time.time()
print('\nTime insert thin: ', t2-t1)

tt_final = line.get_table()



line.build_tracker()
