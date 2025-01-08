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

# Instantiate elements using the class directly
mysext =  xt.Sextupole(length=0.1, k2=0.2)
myoct =  xt.Octupole(length=0.1, k3=0.3)

# It is possible to insert the element in the line and, contextually, define its
# name:
line.insert('s1', mysext, at=5., from_='q1')

# Alternatively it is possible to add the element to the environment and then do
# the insertion:
env.elements['oc1'] = myoct
line.insert('oc1', at=5. , from_='q0')

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# q0                     9            10            11
# drift_2..0            11        12.975         14.95
# oc1                14.95            15         15.05
# drift_2..2         15.05        17.025            19
# q1                    19            20            21
# drift_3..0            21        22.975         24.95
# s1                 24.95            25         25.05
# drift_3..2         25.05        32.525            40
# m0                    40            40            40
# _end_point            40            40            40
