import xtrack as xt
import numpy as np

env = xt.Environment()
env['pi'] = np.pi

env['l_bend'] = 3.5
env['l_quad'] = 1.
env['l_cell'] = 20.
env['n_bends'] = 24.

env['h_bend']= 'pi / n_bends / l_bend'

env.new('mq', xt.Quadrupole, length='l_quad')
env.new('mb', xt.Bend, length='l_bend', h='h_bend', k0='h_bend')

env.new('mqf', 'mq', k1=0.1)
env.new('mqd', 'mq', k1=-0.1)

# We create a line which defines a single arc cell
arc_cell = env.new_line(components=[
    env.new('s.acell', xt.Marker), # At start cell
    env.place('mqf'),
    env.place('mqd', at='l_cell/2', from_='mqf'), # At mid cell
    env.place('mb', at='l_cell/4 - (l_bend/2 + 0.2)', from_='mqf'),
    env.place('mb', at='l_cell/4 + (l_bend/2 + 0.2)', from_='mqf'),
    env.place('mb', at='l_cell/4 - (l_bend/2 + 0.2)', from_='mqd'),
    env.place('mb', at='l_cell/4 + (l_bend/2 + 0.2)', from_='mqd'),
    env.new('e.cell', xt.Marker, at='l_cell')])

# We create a line defining a single straight cell
straight_cell = env.new_line(components=[
    env.new('s.scell', xt.Marker), # At start cell
    env.place('mqf'),
    env.place('mqd', at='l_cell/2', from_='mqf'), # At mid cell
    env.new('e.scell', xt.Marker, at='l_cell')])

# We build an arc concatenating two arc cells
arc = 2 * arc_cell
arc.get_table()
# gives the following (note that the occurrence of each repeated element is
# indiceted together with the element name):
# name                   s element_type isthick isreplica parent_name ...
# s.acell::0             0 Marker         False     False None
# mqf::0                 0 Quadrupole      True     False None
# drift_1::0             1 Drift           True     False None
# mb::0                1.8 Bend            True     False None
# drift_2::0           5.3 Drift           True     False None
# mb::1                5.7 Bend            True     False None
# drift_3::0           9.2 Drift           True     False None
# mqd::0                10 Quadrupole      True     False None
# drift_4::0            11 Drift           True     False None
# mb::2               11.8 Bend            True     False None
# drift_5::0          15.3 Drift           True     False None
# mb::3               15.7 Bend            True     False None
# drift_6::0          19.2 Drift           True     False None
# e.cell::0             20 Marker         False     False None
# s.acell::1            20 Marker         False     False None
# mqf::1                20 Quadrupole      True     False None
# drift_1::1            21 Drift           True     False None
# mb::4               21.8 Bend            True     False None
# drift_2::1          25.3 Drift           True     False None
# mb::5               25.7 Bend            True     False None
# drift_3::1          29.2 Drift           True     False None
# mqd::1                30 Quadrupole      True     False None
# drift_4::1            31 Drift           True     False None
# mb::6               31.8 Bend            True     False None
# drift_5::1          35.3 Drift           True     False None
# mb::7               35.7 Bend            True     False None
# drift_6::1          39.2 Drift           True     False None
# e.cell::1             40 Marker         False     False None
# _end_point            40                False     False None

# Similarly we build a straight section from two straight cells
straight = 2 * straight_cell

# We can assemble a ring by concatenating arcs and straight sections
ring  = 6 * (arc + straight)


