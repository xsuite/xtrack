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
s2 = line.get_length()

elements_to_insert = [
    # s .    # elements to insert (name, element)
    (s0,     [(f'm0_at_a', xt.Marker()), (f'm1_at_a', xt.Marker()), (f'm2_at_a', xt.Marker())]),
    (s0+10., [(f'm0_at_b', xt.Marker()), (f'm1_at_b', xt.Marker()), (f'm2_at_b', xt.Marker())]),
    (s1,     [(f'm0_at_c', xt.Marker()), (f'm1_at_c', xt.Marker()), (f'm2_at_c', xt.Marker())]),
    (s2,     [(f'm0_at_d', xt.Marker()), (f'm1_at_d', xt.Marker()), (f'm2_at_d', xt.Marker())]),
]

line._insert_thin_elements_at_s(elements_to_insert)

tt = line.get_table()
assert np.isclose(tt['s', 'm0_at_a'], s0, rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm1_at_a'], s0, rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm2_at_a'], s0, rtol=0, atol=1e-6)

assert np.isclose(tt['s', 'm0_at_b'], s0 + 10., rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm1_at_b'], s0 + 10., rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm2_at_b'], s0 + 10., rtol=0, atol=1e-6)

assert np.isclose(tt['s', 'm0_at_c'], s1, rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm1_at_c'], s1, rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm2_at_c'], s1, rtol=0, atol=1e-6)

assert np.isclose(tt['s', 'm0_at_d'], s2, rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm1_at_d'], s2, rtol=0, atol=1e-6)
assert np.isclose(tt['s', 'm2_at_d'], s2, rtol=0, atol=1e-6)

pppppp
# Check insertion at beginning and end of the line
l1 = xt.Line(elements=5*[xt.Drift(length=1)])

l1._insert_thin_elements_at_s([
    (0, [(f'm0_at_a', xt.Marker()), (f'm1_at_a', xt.Marker())]),
    (5, [(f'm0_at_b', xt.Marker()), (f'm1_at_b', xt.Marker())]),
])

t1 = l1.get_table()
assert t1.name[0] == 'm0_at_a'
assert t1.name[1] == 'm1_at_a'
assert t1.name[-1] == '_end_point'
assert t1.name[-2] == 'm1_at_b'
assert t1.name[-3] == 'm0_at_b'

assert t1.s[0] == 0
assert t1.s[1] == 0
assert t1.s[-1] == 5.
assert t1.s[-2] == 5.
assert t1.s[-3] == 5.

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
