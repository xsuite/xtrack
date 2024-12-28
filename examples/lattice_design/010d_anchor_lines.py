import xtrack as xt

env = xt.Environment()

# A simple line made of quadrupoles spaced by 5 m
env.new_line(name='l1', components=[
    env.new('q1', 'Quadrupole', length=2.0, at=0., anchor='start'),
    env.new('q2', 'Quadrupole', length=2.0, anchor='start', at=5., from_='end@q1'),
    env.new('q3', 'Quadrupole', length=2.0, anchor='start', at=5., from_='end@q2'),
    env.new('q4', 'Quadrupole', length=2.0, anchor='start', at=5., from_='end@q3'),
    env.new('q5', 'Quadrupole', length=2.0, anchor='start', at=5., from_='end@q4'),
])

# Test absolute anchor of start 'l1'
env.new_line(name='lstart', components=[
    env.place('l1', anchor='start', at=10.),
])

# Test absolute anchor of end 'l1'
env.new_line(name='lend', components=[
    env.place('l1', anchor='end', at=40.),
])

# Test absolute anchor of center 'l1'
env.new_line(name='lcenter', components=[
    env.place('l1', anchor='center', at=25.),
])
