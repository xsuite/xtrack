import xtrack as xt

env = xt.Environment()

env['a'] = 3.

env.new_builder(name='l1')
env['l1'].new('q1', 'Quadrupole', length='a/3.', at=2.5)
env['l1'].new('q2', 'q1', at='a', from_='q1@center')

env.new_builder(name='l_compose',
                components=[
                    env.place('l1', at='3*a'),
                    env.place(-env['l1'], at='10*a'),
                ])
env['l_compose'].build()

