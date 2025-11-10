import xtrack as xt

env = xt.Environment()
env['aa'] = 5
env['bb'] = '2 * aa'

env.new('mb', 'Bend', length=1.0, angle='bb * 1e-3', knl=[0.0, '3*bb'])

env.vars.rename('bb', 'cc')