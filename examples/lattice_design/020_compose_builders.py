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
l1 = b_compose.build()
tt1 = l1.twiss4d()

env['a'] = 2.
l2 = b_compose.build()
tt2 = l2.twiss4d()
