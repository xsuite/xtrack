import xtrack as xt
import xobjects as xo

# env = xt.load('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])

line = xt.Line(elements=[xt.Quadrupole(length=1, k1=0.1)])

line['a'] = 2.

line['e0'].extra = {}
line['e0'].extra['my_extra_1'] = 3.14
line['e0'].extra['my_extra_2'] = '3*a'

assert 'my_extra_1' in line.get('e0').extra
xo.assert_allclose(line['e0'].extra['my_extra_1'], 3.14)
assert 'my_extra_2' in line.get('e0').extra
xo.assert_allclose(line['e0'].extra['my_extra_2'], 6.)

# line2 = xt.Line.from_dict(line.to_dict())
# xo.assert_allclose(line2['e0'].extra['my_extra_1'], 3.14)


