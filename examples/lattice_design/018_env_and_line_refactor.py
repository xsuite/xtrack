import xtrack as xt

line = xt.Line(elements={'e1': xt.Drift(length=1)})
dct_line = line.to_dict()

env = xt.Environment()
env['a'] = 5
env.new_line(name='myline', components=[
    env.new('e1', xt.Drift, length='3*a')
])

env2 = xt.Environment.from_dict(env.to_dict())
