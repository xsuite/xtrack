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
myaperture =  xt.LimitEllipse(a=0.01, b=0.02)

# Insert the element in the line and, contextually, define its name:
line.insert('s1', mysext, at=5., from_='q1')

# Alternatively, add the element to the environment and then do the insertion:
env.elements['ap1'] = myaperture
line.insert('ap1', at='q0@start')

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0           4.5             9
# ap1                    9             9             9
# q0                     9            10            11
# drift_2               11            15            19
# q1                    19            20            21
# drift_3..0            21        22.975         24.95
# s1                 24.95            25         25.05
# drift_3..2         25.05        32.525            40
# m0                    40            40            40
# _end_point            40            40            40

# The above is convenient when adding elements of types defined by the user.
# For example:

class MyElement:
    def __init__(self, param=1.0):
        self.param = param

    def track(self, particles):
        # Custom tracking logic
        pass

# Create multiple instances of MyElement and store them in the environment
env.elements['my_element1'] = MyElement(param=2.0)
env.elements['my_element2'] = MyElement(param=3.0)
env.elements['my_element3'] = MyElement(param=4.0)

# Insert in the lines
line.insert([
    env.place('my_element1', at=12.),
    env.place('my_element2', at=1, from_='my_element1'),
    env.place('my_element3', at=2, from_='my_element2')
]
)

tt = line.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])