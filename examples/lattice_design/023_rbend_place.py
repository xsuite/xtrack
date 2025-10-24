import xtrack as xt
import xobjects as xo

env = xt.Environment()
env['angle'] = 0.5

l = env.new_line(compose=True, length=10.0)
l.new('rbend1', 'RBend', length_straight=1, angle='angle', at=5.0)

t1 = l.get_table()
# t1.cols['s s_center s_start s_end'] is:
# Table: 4 rows, 5 cols
# name                   s      s_center       s_start         s_end
# drift_1                0       2.24738             0       4.49475
# rbend1           4.49475             5       4.49475       5.50525
# drift_2          5.50525       7.75262       5.50525            10
# _end_point            10            10            10            10
xo.assert_allclose(t1['s_center', 'rbend1'], 5, rtol=0, atol=1e-12)
xo.assert_allclose(l.get_length(), 10, rtol=0, atol=1e-12)
xo.assert_allclose(t1.s, [0, 4.49475287, 5.50524713, 10],
        rtol=0, atol=1e-5)
xo.assert_allclose(t1.s_center, [2.24737644, 5, 7.75262356, 10],
        rtol=0, atol=1e-5)
xo.assert_allclose(t1.s_start, [0, 4.49475287, 5.50524713, 10],
        rtol=0, atol=1e-5)
xo.assert_allclose(t1.s_end, [4.49475287, 5.50524713, 10, 10],
        rtol=0, atol=1e-5)

env['angle'] = 0.4
l.regenerate_from_composer()

t2 = l.get_table()
# t2.cols['s s_center s_start s_end'] is:
# Table: 4 rows, 5 cols
# name                   s      s_center       s_start         s_end
# drift_3                0       2.24833             0       4.49665
# rbend1           4.49665             5       4.49665       5.50335
# drift_4          5.50335       7.75167       5.50335            10
# _end_point            10            10            10            10
xo.assert_allclose(t2['s_center', 'rbend1'], 5, rtol=0, atol=1e-12)
xo.assert_allclose(l.get_length(), 10, rtol=0, atol=1e-12)
xo.assert_allclose(t2.s, [0, 4.49665277, 5.50334723, 10],
        rtol=0, atol=1e-5)
xo.assert_allclose(t2.s_center, [2.24832639, 5, 7.75167361, 10],
        rtol=0, atol=1e-5)
xo.assert_allclose(t2.s_start, [0, 4.49665277, 5.50334723, 10],
        rtol=0, atol=1e-5)
xo.assert_allclose(t2.s_end, [4.49665277, 5.50334723, 10, 10],
        rtol=0, atol=1e-5)

