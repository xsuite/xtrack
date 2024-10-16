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

line.get_table(attr=True).cols['name s rot_s_rad']
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

line.get_table(attr=True).cols['name s rot_s_rad']
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

# Replicas behave in the same way as repeated elements, but allow assigning
# a different name to each replica. For example:
env.new('my_mq_1', 'mq', mode='replica')
env.new('my_mq_2', 'mq', mode='replica')

line = env.new_line(components=['my_mq_1', 'dd', 'my_mq_2', 'dd'])

# Here 'mq_1' and 'mq::1' are actually the same element. Any modification on
# 'mq' is seen directly on 'mq::0' and 'mq::1'. For example, we  set the tilt of
# mq by 3 mrad:
line['mq'].rot_s_rad = 3e-3

line.get_table(attr=True).cols['name s rot_s_rad']
# is:
# Table: 5 rows, 3 cols
# name                   s     rot_s_rad
# my_mq_1                0         0.001
# dd::0                0.3             0
# my_mq_2              1.3         0.001
# dd::1                1.6             0
# _end_point           2.6             0

##################
# Element clones #
##################

# Element clones are different from replicas and repeated elements. They are
# actual copies of the element, which inherit the deferred expressions controlling
# their attributes from the original element. For example:

env.new('mq_clone_1', 'mq', mode='clone')
env.new('mq_clone_2', 'mq', mode='clone')

line = env.new_line(components=['mq_clone_1', 'dd', 'mq_clone_2', 'dd'])

line['mq'].get_expr('k1') # is 'kq'
line['mq_clone_1'].get_expr('k1') # is 'kq'
line['mq_clone_2'].get_expr('k1') # is 'kq'

# When changing the value of 'kq', all three elements are affected
env['kq'] = 0.2
line['mq'].k1 # is 0.2
line['mq_clone_1'].k1 # is 0.2
line['mq_clone_2'].k1 # is 0.2

# Note instead that if we alter the expression controlling the attribute of mq after
# its creation, the clones are not affected. For example:
line['mq'].k1 = '2 * kq'

line['mq'].get_expr('k1')         # is '2 * kq'
line['mq_clone_1'].get_expr('k1') # is 'kq'
line['mq_clone_2'].k1             # is 'kq'

# Clones allow for example, specifying different values for tilts and offsets
# for different elements. For example:

line['mq_clone_1'].rot_s_rad = 2e-3
line['mq_clone_2'].rot_s_rad = 3e-3

line.get_table(attr=True).cols['name s k1l rot_s_rad']
# is:
# Table: 5 rows, 4 cols
# name                   s           k1l     rot_s_rad
# mq_clone_1             0          0.03         0.002
# dd::0                0.3             0             0
# mq_clone_2           1.3          0.03         0.003
# dd::1                1.6             0             0
# _end_point           2.6             0             0

##########################################
# Replace repleted elements and replicas #
##########################################

# The line provides methods to automatically replace repeated elements and replicas
# with clones. For example:

line = env.new_line(components=['mq', 'dd', 'mq', 'dd', 'my_mq_1'])
line.get_table(attr=True).cols['name s isreplica']
# is:
# name                   s isreplica
# mq::0                  0     False
# dd::0                0.3     False
# mq::1                1.3     False
# dd::1                1.6     False
# my_mq_1              2.6      True
# _end_point           2.9     False

line.replace_all_repeated_elements()
line.replace_all_replicas()

line.get_table(attr=True).cols['name s isreplica']
# is:
# name                   s isreplica
# mq.0                   0     False
# dd.0                 0.3     False
# mq.1                 1.3     False
# dd.1                 1.6     False
# my_mq_1              2.6     False
# _end_point           2.9     False
