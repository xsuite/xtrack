import xtrack as xt

# We create an environment and define a quadrupole and a drift
env = xt.Environment()
env['kq'] = 0.1
env.new('mq', xt.Quadrupole, length=0.3, k1='kq')
env.new('dd', xt.Drift, length=1)

#####################
# Repeated elements #
#####################

# We create a line with repeated elements
line = env.new_line(components=['mq', 'dd', 'mq', 'dd'])

line.get_table(attr=True).cols['name s rot_s_rad'].show()
# is:
# name                   s     rot_s_rad
# mq::0                  0             0
# dd::0                0.3             0
# mq::1                1.3             0
# dd::1                1.6             0
# _end_point           2.6             0

# Here 'mq::0' and 'mq::1' are actually the same element. Any modification on
# 'mq' is seen directly on 'mq::0' and 'mq::1'. For example, we  tilt mq by 1 mrad:
line['mq'].rot_s_rad = 1e-3

line.get_table(attr=True).cols['name s rot_s_rad'].show()
# is:
# name                   s     rot_s_rad
# mq::0                  0         0.001
# dd::0                0.3             0
# mq::1                1.3         0.001
# dd::1                1.6             0
# _end_point           2.6             0

############
# Replicas #
############

# Replicas behave in the same way as repeted elements, but allow assignining
# a different name to each replica. For example:
env.new('mq_1', 'mq', mode='replica')
env.new('mq_2', 'mq', mode='replica')

line = env.new_line(components=['mq_1', 'dd', 'mq_2', 'dd'])
line.get_table(attr=True).cols['name s length'].show()
# is:
# name                   s        length
# mq_1                   0           0.5
# dd::0                0.5             1
# mq_2                 1.5           0.5
# dd::1                  2             1
# _end_point             3             0



