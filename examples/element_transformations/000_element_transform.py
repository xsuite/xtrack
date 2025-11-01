import xtrack as xt
from cpymad.madx import Madx

env = xt.Environment()
line = env.new_line(length=4.0, name='seq',
    components=[
        env.new('b1', 'Bend', length=1.0, angle=0.2, at=0.5),
        env.new('b2', 'Bend', length=1.0, angle=0.3, at=2.5),
    ])

print('The line as created:')
line.get_table().show()
# name                   s element_type isthick isreplica parent_name ...
# b1                     0 Bend            True     False None
# drift_1                1 Drift           True     False None
# b2                     2 Bend            True     False None
# drift_2                3 Drift           True     False None
# _end_point             4                False     False None

# Shift and tilt selected elements
line['b1'].shift_x = -0.01
line['b1'].rot_s_rad = 0.8
line['b2'].shift_s = 0.02
line['b2'].rot_s_rad = -0.8

tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'shift_x', 'shift_y', 'shift_s', 'rot_s_rad']
# returns:
#
# Table: 5 rows, 8 cols
# name        s element_type isthick   shift_x   shift_y   shift_s   rot_s_rad
# b1          0 Bend            True     -0.01         0         0         0.8
# drift_1     1 Drift           True         0         0         0           0
# b2          2 Bend            True         0         0      0.02        -0.8
# drift_2     3 Drift           True         0         0         0           0
# _end_point  4                False         0         0         0           0
