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
    env.new('mb1', 'mb', at='l_cell/4 - (l_bend/2 + 0.2)', from_='mqf'),
    env.new('mb2', 'mb', at='l_cell/4 + (l_bend/2 + 0.2)', from_='mqf'),
    env.place('mqd', at='l_cell/2 - l_quad/2', from_='mqf'),
    ])


# We create two replicas of the half cell, one mirrored and one not and we
# call them left and right
arc_half_cell_left = env.new('l', arc_half_cell, mirror=True, mode='replica')
arc_half_cell_right = env.new('r', arc_half_cell, mode='replica')

# We concatenate the two half cells to create a full cell
cell = arc_half_cell_left + arc_half_cell_right
cell.get_table()
# this gives the following (note that the name given to the line is added to the
# element name ad suffix):
#
# name                   s element_type isthick isreplica parent_name ...
# mqd.l                  0 Quadrupole      True      True mqd
# drift_3.l              1 Drift           True      True drift_3
# mb2.l                1.3 Bend            True      True mb2
# drift_2.l            4.8 Drift           True      True drift_2
# mb1.l                5.2 Bend            True      True mb1
# drift_1.l            8.7 Drift           True      True drift_1
# mqf.l                9.5 Quadrupole      True      True mqf
# mqf.r               10.5 Quadrupole      True      True mqf
# drift_1.r           11.5 Drift           True      True drift_1
# mb1.r               12.3 Bend            True      True mb1
# drift_2.r           15.8 Drift           True      True drift_2
# mb2.r               16.2 Bend            True      True mb2
# drift_3.r           19.7 Drift           True      True drift_3
# mqd.r                 20 Quadrupole      True      True mqd
# _end_point            21                False     False None

# Similarly, we can create two cells with different names
env.new('cell1', cell, mode='replica')
env.new('cell2', cell, mode='replica')

# And we can create an arc concatenating the two cells
env['arc'] = env['cell1'] + env['cell2']

# We can make 6 arcs with different names
env.new('arc1', 'arc', mode='replica')
env.new('arc2', 'arc', mode='replica')
env.new('arc3', 'arc', mode='replica')
env.new('arc4', 'arc', mode='replica')
env.new('arc5', 'arc', mode='replica')
env.new('arc6', 'arc', mode='replica')
# Also in this case the name of the line is added as a suffix to the element
# names.

# We concatenated them in a ring. This can be done using the '+' operator
# as shown before or using the arcs as subsections of a new line, i.e.:
ring = env.new_line(components=['arc1', 'arc2', 'arc3', 'arc4', 'arc5', 'arc6'])

# One can see that having used named lines and replicas to build the ring has
# generated a regular name pattern for the elements in the ring, as we can see
# for example by listing all focusing quadrupoles in the ring:
ring.get_table().rows['mqf.*']
# is:
#
# Table: 24 rows, 8 cols
# name                         s element_type isthick isreplica ...
# mqf.l.cell1.arc1           9.5 Quadrupole      True      True
# mqf.r.cell1.arc1          10.5 Quadrupole      True      True
# mqf.l.cell2.arc1          30.5 Quadrupole      True      True
# mqf.r.cell2.arc1          31.5 Quadrupole      True      True
# mqf.l.cell1.arc2          51.5 Quadrupole      True      True
# mqf.r.cell1.arc2          52.5 Quadrupole      True      True
# mqf.l.cell2.arc2          72.5 Quadrupole      True      True
# mqf.r.cell2.arc2          73.5 Quadrupole      True      True
# mqf.l.cell1.arc3          93.5 Quadrupole      True      True
# mqf.r.cell1.arc3          94.5 Quadrupole      True      True
# mqf.l.cell2.arc3         114.5 Quadrupole      True      True
# mqf.r.cell2.arc3         115.5 Quadrupole      True      True
# etc...