import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

env['a'] = 1.

env.new_line(name='l1', compose=True)
env['l1'].new('q1', 'Quadrupole', length='a', at='0.5*a')
env['l1'].new('q2', 'q1', at='4*a', from_='q1@center')

b_compose = env.new_line(compose=True,
                components=[
                    env.place('l1', at='7.5*a'),
                    env.place(-env['l1'], at='17.5*a'),
                ])
tt1 = b_compose.get_table()
# tt1.cols['s', 'name', 'element_type', 'env_name'] is:
# Table: 9 rows, 4 cols
# name                   s element_type env_name
# drift_3                0 Drift        drift_3
# q1::0                  5 Quadrupole   q1
# drift_1                6 Drift        drift_1
# q2::0                  9 Quadrupole   q2
# drift_4               10 Drift        drift_4
# q2::1                 15 Quadrupole   q2
# drift_2               16 Drift        drift_2
# q1::1                 19 Quadrupole   q1
# _end_point            20              _end_point

env['a'] = 2.
b_compose.regenerate_from_composer()
tt2 = b_compose.get_table()
# tt2.cols['s', 'name', 'element_type', 'env_name'] is:
# Table: 9 rows, 4 cols
# name                   s element_type env_name
# drift_7                0 Drift        drift_7
# q1::0                 10 Quadrupole   q1
# drift_5               12 Drift        drift_5
# q2::0                 18 Quadrupole   q2
# drift_8               20 Drift        drift_8
# q2::1                 30 Quadrupole   q2
# drift_6               32 Drift        drift_6
# q1::1                 38 Quadrupole   q1
# _end_point            40              _end_point

# Check tt1
assert np.all(tt1.name ==
    ['drift_4', 'q1::0', 'drift_2', 'q2::0', 'drift_5', 'q2::1',
       'drift_3', 'q1::1', '_end_point'])
xo.assert_allclose(tt1.s,
        [ 0.,  5.,  6.,  9., 10., 15., 16., 19., 20.],
        rtol=0, atol=1e-12)
assert np.all(tt1.element_type ==
    ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
     'Quadrupole', 'Drift', 'Quadrupole', ''])
assert np.all(tt1.env_name ==
    ['drift_4', 'q1', 'drift_2', 'q2', 'drift_5',
     'q2', 'drift_3', 'q1', '_end_point'])

# Check tt2
assert np.all(tt2.name ==
    ['drift_8', 'q1::0', 'drift_6', 'q2::0', 'drift_9', 'q2::1',
       'drift_7', 'q1::1', '_end_point'])
xo.assert_allclose(tt2.s, 2 * tt1.s, rtol=0, atol=1e-12)
assert np.all(tt2.element_type ==
    ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
     'Quadrupole', 'Drift', 'Quadrupole', ''])
assert np.all(tt2.env_name ==
    ['drift_8', 'q1', 'drift_6', 'q2', 'drift_9',
        'q2', 'drift_7', 'q1', '_end_point'])

# Same in MAD-X

mad_src = """
a = 1;
q1: quadrupole, L:=a;
q2: q1;
d1: drift, L:=3*a;

d5: drift, L:=5*a;

l1: line=(q1,d1,q2);
l2: line=(d5, l1, d5, -l1);

s1: sequence, refer=centre, l:=5*a;
    q1, at:=0.5*a;
    q2, at:=4.5*a;
endsequence;
sm1: sequence, refer=centre, l:=5*a;
    q2, at:=0.5*a;
    q1, at:=4.5*a;
endsequence;
s2: sequence, refer=centre, l:=20*a;
    s1, at:=7.5*a;
    sm1, at:=17.5*a;
endsequence;

a=2;

"""
from cpymad.madx import Madx
madx = Madx()
madx.input(mad_src)
madx.beam()
madx.use('l2')
tt_mad_l2 = xt.Table(madx.twiss(betx=1, bety=1), _copy_cols=True)
madx.use('s2')
tt_mad_s2 = xt.Table(madx.twiss(betx=1, bety=1), _copy_cols=True)

env_mad = xt.load(string=mad_src, format='madx')
tt_xs_mad_l2 = env_mad['l2'].get_table()
# tt_xs_mad_l2.cols['s', 'name', 'element_type', 'env_name'] is:
# Table: 9 rows, 4 cols
# name                   s element_type env_name
# d5::0                  0 Drift        d5
# q1::0                 10 Quadrupole   q1
# d1::0                 12 Drift        d1
# q2::0                 18 Quadrupole   q2
# d5::1                 20 Drift        d5
# q2::1                 30 Quadrupole   q2
# d1::1                 32 Drift        d1
# q1::1                 38 Quadrupole   q1
# _end_point            40              _end_point

tt_xs_mad_s2 = env_mad['s2'].get_table()
# tt_xs_mad_s2.cols['s', 'name', 'element_type', 'env_name'] is:
# Table: 9 rows, 4 cols
# Table: 9 rows, 4 cols
# name                   s element_type env_name
# drift_3                0 Drift        drift_3
# q1::0                 10 Quadrupole   q1
# drift_1               12 Drift        drift_1
# q2::0                 18 Quadrupole   q2
# drift_4               20 Drift        drift_4
# q2::1                 30 Quadrupole   q2
# drift_2               32 Drift        drift_2
# q1::1                 38 Quadrupole   q1
# _end_point            40              _end_point

assert np.all(tt_xs_mad_l2.name ==
    ['d5::0', 'q1::0', 'd1::0', 'q2::0', 'd5::1', 'q2::1',
       'd1::1', 'q1::1', '_end_point'])
xo.assert_allclose(tt_xs_mad_l2.s,
        2 * tt1.s,
        rtol=0, atol=1e-12)
assert np.all(tt_xs_mad_l2.element_type ==
    ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
     'Quadrupole', 'Drift', 'Quadrupole', ''])
assert np.all(tt_xs_mad_l2.env_name ==
    ['d5', 'q1', 'd1', 'q2', 'd5',
     'q2', 'd1', 'q1', '_end_point'])
xo.assert_allclose(tt_mad_l2.s[:-1], tt_xs_mad_l2.s, atol=1e-12, rtol=0)

assert np.all(tt_xs_mad_s2.name ==
    ['drift_3', 'q1::0', 'drift_1', 'q2::0', 'drift_4', 'q2::1',
       'drift_2', 'q1::1', '_end_point'])
xo.assert_allclose(tt_xs_mad_s2.s,
        2 * tt1.s,
        rtol=0, atol=1e-12)
assert np.all(tt_xs_mad_s2.element_type ==
    ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
     'Quadrupole', 'Drift', 'Quadrupole', ''])
assert np.all(tt_xs_mad_s2.env_name ==
    ['drift_3', 'q1', 'drift_1', 'q2', 'drift_4',
     'q2', 'drift_2', 'q1', '_end_point'])