import xtrack as xt

env = xt.Environment()

env.new('q1', 'Quadrupole', length=2.0)

line = env.new_line(components=[
   env.place('q1', anchor='start', at=1.),
])

line.get_table().show()