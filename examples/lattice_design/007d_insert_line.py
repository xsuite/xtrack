import xtrack as xt

# Create an environment
env = xt.Environment()

# Create a line with two quadrupoles and a marker
line = env.new_line(name='myline', components=[
    env.new('q0', xt.Quadrupole, length=2.0, at=10.),
    env.new('q1', xt.Quadrupole, length=2.0, at=20.),
    env.new('m0', xt.Marker, at=50),
    ])

tt0 = line.get_table()
tt0.show(cols=['s_start', 's_center', 's_end'])
# is:
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2               11            15            19
# q1                    19            20            21
# drift_3               21          35.5            50
# m0                    50            50            50

# Create a set of new elements to be placed
env.new('s1', xt.Sextupole, length=2., k2=0.2)
env.new('s2', xt.Sextupole, length=2., k2=-0.2)
env.new('m1', xt.Marker)
env.new('m2', xt.Marker)
env.new('m3', xt.Marker)

subline = env.new_line(components=[
    env.place('s1', at=1.0),
    env.place('s2', at=5.0),
    env.place('m1', at='start@s1'),
    env.place('m2', at='end@s2')])

# Insert the new elements in the line
line.insert([
    env.place(subline, anchor='start', at=1., from_='end@q1'),
    env.place('m3', at=30.)])

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2               11            15            19
# q1_entry              19            19            19
# q1..0                 19          19.5            20
# drift_4               20        20.475         20.95
# m1                 20.95         20.95         20.95
# s1                 20.95            21         21.05
# drift_5            21.05          21.5         21.95
# s2                 21.95            22         22.05
# m2                 22.05         22.05         22.05
# drift_3..3         22.05        26.025            30
# m3                    30            30            30
# drift_3..4            30            40            50
# m0                    50            50            50
# _end_point            50            50            50
