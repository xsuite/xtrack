import xtrack as xt

# Create an environment
env = xt.Environment()

env.new('m0', xt.Marker)
env.new('m1', xt.Marker)
env.new('m2', xt.Marker)
env.new('m3', xt.Marker)
env.new('m4', xt.Marker)
env.new('m5', xt.Marker)
env.new('m6', xt.Marker)
env.new('m7', xt.Marker)



env.new_line(name='myline', compose=True)
composer = env['myline'].composer

composer.components.extend([
    env.place('m0'),
    env.place('m3', at=10.),
    env.place(['m1', 'm2']),
    env.place(['m4', 'm5'], at='m3@start'),
    env.place(['m6', 'm7'], at='m3@end'),
])

tt_unsorted = composer.resolve_s_positions(sort=False)
tt_unsorted.cols['s from_ from_anchor'].show()

tt_sorted = composer.resolve_s_positions(sort=True)
tt_sorted.cols['s from_ from_anchor'].show()
