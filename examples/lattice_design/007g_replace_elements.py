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

# Create a new element
env.new('s1', xt.Sextupole, length=2, k2=0.2)

# Replace `q1` with `s1`
line.replace('q1', 's1')

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2               11            15            19
# s1                    19            20            21
# drift_3               21          30.5            40
# m0                    40            40            40
# _end_point            40            40            40