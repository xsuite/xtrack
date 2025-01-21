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
line.insert([
    env.place('s1', at=5.),
    env.place('s2', anchor='end', at=-5., from_='q1@start'),
    env.place(['m1', 'm2'], at='m0@start'),
    env.place('m3', at='m0@end'),
    ])

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1..0             0         2.475          4.95
# s1                  4.95             5          5.05
# drift_1..2          5.05         7.025             9
# q0                     9            10            11
# drift_2..0            11         12.45          13.9
# s2                  13.9         13.95            14
# drift_2..2            14          16.5            19
# q1                    19            20            21
# drift_3               21          30.5            40
# m1                    40            40            40
# m0                    40            40            40
# m3                    40            40            40
