import xtrack as xt

# Create an environment
env = xt.Environment()

# Create a line with two quadrupoles and a marker
line = env.new_line(name='myline', components=[
    env.new('q0', xt.Quadrupole, length=2.0, at=10.),
    env.new('q1', xt.Quadrupole, length=2.0, at=20.),
    env.new('m0', xt.Marker, at=40.),
    ])

tt0 = line.get_table()
tt0.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2               11            15            19
# q1                    19            20            21
# drift_3               21          30.5            40
# m0                    40            40            40
# _end_point            40            40            40

# Create a set of new elements to be placed
env.new('s1', xt.Sextupole, length=0.1, k2=0.2)
env.new('s2', xt.Sextupole, length=0.1, k2=-0.2)
env.new('m1', xt.Marker)
env.new('m2', xt.Marker)
env.new('m3', xt.Marker)

# Insert the new elements in the line
line.append(['m1', 's1', 'm2', 's2', 'm3'])

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2               11            15            19
# q1                    19            20            21
# drift_3               21          30.5            40
# m0                    40            40            40
# m1                    40            40            40
# s1                    40         40.05          40.1
# m2                  40.1          40.1          40.1
# s2                  40.1         40.15          40.2
# m3                  40.2          40.2          40.2
# _end_point          40.2          40.2          40.2

# Elements can be appended also when they are created using the class directly.
# The element name is defined contextually:
myoct = xt.Octupole(length=0.1, k3=0.3)
line.append('o1', myoct)

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2               11            15            19
# q1                    19            20            21
# drift_3               21          30.5            40
# m0                    40            40            40
# m1                    40            40            40
# s1                    40         40.05          40.1
# m2                  40.1          40.1          40.1
# s2                  40.1         40.15          40.2
# m3                  40.2          40.2          40.2
# o1                  40.2         40.25          40.3
# _end_point          40.3          40.3          40.3
