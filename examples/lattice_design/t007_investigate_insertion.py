import xtrack as xt

# Create an environment
env = xt.Environment()

env.new('m0', xt.Marker)
env.new('m1', xt.Marker)
env.new('m2', xt.Marker)
env.new('m3', xt.Marker)

env.new_line(name='myline', compose=True)
composer = env['myline'].composer

composer.append(
    env.place('m0', at=0.0),
    env.place(['m1', 'm2'])
)