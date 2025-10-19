import xtrack as xt

env = xt.Environment()

env['a'] = 1.

env.new_builder(name='l1')
env['l1'].new('q1', 'Quadrupole', length='a', at='0.5*a')
env['l1'].new('q2', 'q1', at='4*a', from_='q1@center')

b_compose = env.new_builder(components=[
                    env.place('l1', at='7.5*a'),
                    env.place(-env['l1'], at='17.5*a'),
                ])
tt1 = b_compose.build().get_table()

env['a'] = 2.
tt2 = b_compose.build().get_table()

# Same in MAD-X

mad_src = """
a = 1;
q1: quadrupole, L:=a;
q2: q1;
d1: drift, L:=3*a;

d5: drift, L:=5*a;

l1: line=(q1,d1,q2);
l2: line=(d5, l1, d5, -l1);

a=2;

"""
from cpymad.madx import Madx
madx = Madx()
madx.input(mad_src)
madx.beam()
madx.use('l2')
tt_mad = xt.Table(madx.twiss(betx=1, bety=1), _copy_cols=True)

env_mad = xt.load(string=mad_src, format='madx')
