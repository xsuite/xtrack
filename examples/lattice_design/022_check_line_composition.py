import numpy as np
import xtrack as xt
import xobjects as xo

env = xt.Environment()

env['a'] = 1.

l1 = env.new_line(compose=True)
l1.new('q1', 'Quadrupole', length='a', at='0.5*a')
l1.new('q2', 'q1', at='4*a', from_='q1@center')

l2 = env.new_line(compose=True)
l2.new('s1', 'Sextupole', length='a', at='1.5*a')
l2.new('s2', 's1', at='5*a', from_='s1@center')

assert l1.mode == 'compose'
assert l2.mode == 'compose'

ss = l1 + l2
assert ss.mode == 'compose'
assert len(ss.composer.components) == 2
tss = ss.get_table()
tss.cols['s element_type env_name']
# Table: 8 rows, 4 cols
# name                   s element_type env_name
# q1                     0 Quadrupole   q1
# drift_1                1 Drift        drift_1
# q2                     4 Quadrupole   q2
# drift_2                5 Drift        drift_2
# s1                     6 Sextupole    s1
# drift_3                7 Drift        drift_3
# s2                    11 Sextupole    s2
# _end_point            12              _end_point

assert np.all(tss.name == np.array(
    ['q1', 'drift_1', 'q2', 'drift_2', 's1', 'drift_3', 's2', '_end_point']))
xo.assert_allclose(tss.s,
        [ 0.,  1.,  4.,  5.,  6.,  7., 11., 12.],
        rtol=0, atol=1e-12)
assert np.all(tss.element_type ==
    ['Quadrupole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole',
     'Drift', 'Sextupole', ''])

ss.end_compose()
assert ss.mode == 'normal'
tss2 = ss.get_table()
# is the same as tss apart from different drift names
assert np.all(tss2.name == np.array(
    ['q1', 'drift_4', 'q2', 'drift_5', 's1', 'drift_6', 's2', '_end_point']))
xo.assert_allclose(tss2.s,
        [ 0.,  1.,  4.,  5.,  6.,  7., 11., 12.],
        rtol=0, atol=1e-12)
assert np.all(tss2.element_type ==
    ['Quadrupole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole',
     'Drift', 'Sextupole', ''])

l1.end_compose()
l2.end_compose()
ss2 = l1 + l2
assert ss2.mode == 'normal'
tss3 = ss2.get_table()
# is the same as tss apart from different drift names
assert np.all(tss3.name == np.array(
    ['q1', 'drift_7', 'q2', 'drift_8', 's1', 'drift_9', 's2', '_end_point']))
xo.assert_allclose(tss3.s,
        [ 0.,  1.,  4.,  5.,  6.,  7., 11., 12.],
        rtol=0, atol=1e-12)
assert np.all(tss3.element_type ==
    ['Quadrupole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole',
     'Drift', 'Sextupole', ''])

l1.regenerate_from_composer()
l2.regenerate_from_composer()

mm = 3*l1
assert len(mm.composer.components) == 3
tmm = mm.get_table()
# tmm.cols['s element_type env_name'] is:
# Table: 10 rows, 4 cols
# name                   s element_type env_name
# q1::0                  0 Quadrupole   q1
# drift_10               1 Drift        drift_10
# q2::0                  4 Quadrupole   q2
# q1::1                  5 Quadrupole   q1
# drift_11               6 Drift        drift_11
# q2::1                  9 Quadrupole   q2
# q1::2                 10 Quadrupole   q1
# drift_12              11 Drift        drift_12
# q2::2                 14 Quadrupole   q2
# _end_point            15              _end_point
assert np.all(tmm.name == np.array(
    ['q1::0', 'drift_10', 'q2::0',
     'q1::1', 'drift_11', 'q2::1',
     'q1::2', 'drift_12', 'q2::2',
     '_end_point']))
xo.assert_allclose(tmm.s,
        [ 0.,  1.,  4.,  5.,  6.,  9., 10., 11., 14., 15.],
        rtol=0, atol=1e-12)
assert np.all(tmm.element_type ==
    ['Quadrupole', 'Drift', 'Quadrupole',
     'Quadrupole', 'Drift', 'Quadrupole',
     'Quadrupole', 'Drift', 'Quadrupole',
     ''])