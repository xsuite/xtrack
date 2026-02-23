import numpy as np

import xtrack as xt
import xobjects as xo

env = xt.Environment()

# A simple line made of quadrupoles spaced by 5 m
env.new_line(name='l1', components=[
    env.new('q1', 'Quadrupole', length=2.0, at=0., anchor='start'),
    env.new('q2', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q1@end'),
    env.new('q3', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q2@end'),
    env.new('q4', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q3@end'),
    env.new('q5', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q4@end'),
])

# Test absolute anchor of start 'l1'
env.new_line(name='lstart', components=[
    env.place('l1', anchor='start', at=10.),
])

# Test absolute anchor of end 'l1'
env.new_line(name='lend', components=[
    env.place('l1', anchor='end', at=40.),
])

# Test absolute anchor of center 'l1'
env.new_line(name='lcenter', components=[
    env.place('l1', anchor='center', at=25.),
])

# Test relative anchor of start 'l1' to start of another element
env.new_line(name='lstcnt', components=[
    env.new('q0', 'Quadrupole', length=2.0, at=5.),
    env.place('l1', anchor='start', at=5., from_='q0@center'),
])

# Test relative anchor of start 'l1' to end of another element
env.new_line(name='lstst', components=[
    env.place('q0', at=5.),
    env.place('l1', anchor='start', at=5. + 1., from_='q0@end'),
])


# Test relative anchor of start 'l1' to end of another element
env.new_line(name='lstend', components=[
    env.place('q0', at=5.),
    env.place('l1', anchor='start', at=5. - 1., from_='q0@end'),
])

tt_l1 = env['l1'].get_table()
tt_test = tt_l1
assert np.all(tt_test.name == np.array(
    ['q1', 'drift_1', 'q2', 'drift_2', 'q3', 'drift_3', 'q4', 'drift_4',
     'q5', '_end_point']))
xo.assert_allclose(tt_test.s, np.array(
    [ 0.,  2.,  7.,  9., 14., 16., 21., 23., 28., 30.]),
    rtol=0., atol=1e-15)
xo.assert_allclose(tt_test.s_start, np.array(
    [ 0.,  2.,  7.,  9., 14., 16., 21., 23., 28., 30.]),
    rtol=0., atol=1e-15)
xo.assert_allclose(tt_test.s_center, np.array(
    [ 1. ,  4.5,  8. , 11.5, 15. , 18.5, 22. , 25.5, 29. , 30. ]),
    rtol=0., atol=1e-15)
xo.assert_allclose(tt_test.s_end, np.array(
    [ 2.,  7.,  9., 14., 16., 21., 23., 28., 30., 30.]),
    rtol=0., atol=1e-15)

tt_lstart = env['lstart'].get_table()
tt_test = tt_lstart
assert np.all(tt_test.name == np.array(
    ['drift_5', 'q1', 'drift_1', 'q2', 'drift_2', 'q3', 'drift_3', 'q4',
       'drift_4', 'q5', '_end_point']))
xo.assert_allclose(tt_test.s, np.array(
    [ 0., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
    rtol=0., atol=1e-15)

tt_lend = env['lend'].get_table()
tt_test = tt_lend
assert np.all(tt_test.name == np.array(
    ['drift_6', 'q1', 'drift_1', 'q2', 'drift_2', 'q3', 'drift_3', 'q4',
    'drift_4', 'q5', '_end_point']))
xo.assert_allclose(tt_test.s, np.array(
    [ 0., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
    rtol=0., atol=1e-15)

tt_lcenter = env['lcenter'].get_table()
tt_test = tt_lcenter
assert np.all(tt_test.name == np.array(
    ['drift_7', 'q1', 'drift_1', 'q2', 'drift_2', 'q3', 'drift_3', 'q4',
    'drift_4', 'q5', '_end_point']))
xo.assert_allclose(tt_test.s, np.array(
    [ 0., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
    rtol=0., atol=1e-15)

tt_lstcnt = env['lstcnt'].get_table()
tt_test = tt_lstcnt
assert np.all(tt_test.name == np.array(
    ['drift_8', 'q0', 'drift_9', 'q1', 'drift_1', 'q2', 'drift_2', 'q3',
     'drift_3', 'q4', 'drift_4', 'q5', '_end_point']))
xo.assert_allclose(tt_test.s, np.array(
    [ 0.,  4.,  6., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
    rtol=0., atol=1e-15)

tt_lstst = env['lstst'].get_table()
tt_test = tt_lstst
assert np.all(tt_test.name == np.array(
    ['drift_10', 'q0', 'drift_11', 'q1', 'drift_1', 'q2', 'drift_2', 'q3',
     'drift_3', 'q4', 'drift_4', 'q5', '_end_point']))

tt_lstend = env['lstend'].get_table()
tt_test = tt_lstend
assert np.all(tt_test.name == np.array(
    ['drift_12', 'q0', 'drift_13', 'q1', 'drift_1', 'q2', 'drift_2', 'q3',
     'drift_3', 'q4', 'drift_4', 'q5', '_end_point']))
