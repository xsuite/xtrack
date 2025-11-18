import xtrack as xt
from cpymad.madx import Madx

# Build a simple line
env = xt.Environment()
line = env.new_line(length=4.0, name='seq',
    components=[
        env.new('b1', 'Bend', length=1.0, angle=0.2, k0_from_h=True, at=0.5),
        env.new('q1', 'Quadrupole', length=1.0, k1=0.1, at=2.5),
    ])

print('The line as created:')
line.get_table().show()

# Add multipolar components to elements
line['b1'].knl[2] = 0.001 # Normal sextupole component
line['q1'].ksl[3] = 0.002 # Skew octupole component

tt = line.get_table(attr=True)
tt.cols['s', 'element_type', 'isthick', 'k2l', 'k3sl'].show()
# returns:
#
# name       s element_type isthick   k2l  k3sl
# b1         0 Bend            True 0.001     0
# drift_0    1 Drift           True     0     0
# q1         2 Quadrupole      True     0 0.002
# drift_1    3 Drift           True     0     0
# _end_point 4                False     0     0
