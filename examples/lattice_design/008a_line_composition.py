import xtrack as xt
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
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

# We create a line which defines half of a FODO cell
arc_half_cell = env.new_line(components=[
    env.place('mqf'),
    env.place('mb', at='l_cell/4 - (l_bend/2 + 0.2)', from_='mqf'),
    env.place('mb', at='l_cell/4 + (l_bend/2 + 0.2)', from_='mqf'),
    env.place('mqd', at='l_cell/2 - l_quad/2', from_='mqf'),
    ])
arc_half_cell.survey().plot() # plots the layout of the created line
arc_half_cell.get_table()
# is:
#
# name                   s element_type isthick isreplica parent_name ...
# mqf                    0 Quadrupole      True     False None
# drift_1                1 Drift           True     False None
# mb::0                1.8 Bend            True     False None
# drift_2              5.3 Drift           True     False None
# mb::1                5.7 Bend            True     False None
# drift_3              9.2 Drift           True     False None
# mqd                  9.5 Quadrupole      True     False None
# _end_point          10.5                False     False None

# We can mirror a line by simply using the minus sign
(-arc_half_cell).get_table()
# is:
# name                   s element_type isthick isreplica parent_name ...
# mqd                    0 Quadrupole      True     False None
# drift_3                1 Drift           True     False None
# mb::0                1.3 Bend            True     False None
# drift_2              4.8 Drift           True     False None
# mb::1                5.2 Bend            True     False None
# drift_1              8.7 Drift           True     False None
# mqf                  9.5 Quadrupole      True     False None
# _end_point          10.5                False     False None

# We concatenate the two to obtain a full cell
arc_cell = -arc_half_cell + arc_half_cell
arc_cell.get_table()
# gives the following (note that the occurrence of each repeated element is
# indicated together with the element name):
#
# Table: 15 rows, 8 cols
# name                   s element_type isthick isreplica parent_name ...
# mqd::0                 0 Quadrupole      True     False None
# drift_3::0             1 Drift           True     False None
# mb::0                1.3 Bend            True     False None
# drift_2::0           4.8 Drift           True     False None
# mb::1                5.2 Bend            True     False None
# drift_1::0           8.7 Drift           True     False None
# mqf::0               9.5 Quadrupole      True     False None
# mqf::1              10.5 Quadrupole      True     False None
# drift_1::1          11.5 Drift           True     False None
# mb::2               12.3 Bend            True     False None
# drift_2::1          15.8 Drift           True     False None
# mb::3               16.2 Bend            True     False None
# drift_3::1          19.7 Drift           True     False None
# mqd::1                20 Quadrupole      True     False None
# _end_point            21                False     False None

# We create another line defining a single straight cell
straight_cell = env.new_line(components=[
    env.new('s.scell', xt.Marker), # At start cell
    env.place('mqf'),
    env.place('mqd', at='l_cell/2', from_='mqf'), # At mid cell
    env.new('e.scell', xt.Marker, at='l_cell')])

# We a ring composing arcs and straight sections
ring = 6 * (2 * arc_cell + 3 * straight_cell)

# We plot the layout of the ring
ring.survey().plot()
