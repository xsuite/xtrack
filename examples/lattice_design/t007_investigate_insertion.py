import xtrack as xt
import xobjects as xo
import numpy as np

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
env.new('m8', xt.Marker)
env.new('m9', xt.Marker)
env.new('m10', xt.Marker)

env.new_line(name='myline', compose=True)
composer = env['myline'].composer

composer.components.extend([
    env.place('m0'),
    env.place('m3', at=10.),
    env.place(['m1', 'm2']),
    env.place(['m6', 'm7'], at='m3@end'),
    env.place(['m4', 'm5'], at='m3@start'),
    env.place('m8', at=10, from_='m0'),
    env.place('m9', at=20.),
    env.place('m10', at=-10, from_='m9'),
])

tt_unsorted = composer.resolve_s_positions(sort=False)
tt_unsorted.cols['s from_ from_anchor'].show()
# prints:
# name             s from_ from_anchor
# m0               0 None  None
# m3              10 None  None
# m1              10 m3    end
# m2              10 m1    end
# m6              10 m3    end
# m7              10 m6    end
# m4              10 m3    start
# m5              10 m4    end
# m8              10 m0    None
# m9              20 None  None
# m10             10 m9    None

assert np.all(tt_unsorted.name == [
    'm0', 'm3', 'm1', 'm2', 'm6', 'm7', 'm4', 'm5', 'm8', 'm9', 'm10'
])
xo.assert_allclose(tt_unsorted.s, [
    0., 10., 10., 10., 10., 10., 10., 10., 10., 20., 10.])
assert np.all(tt_unsorted.from_ == [
    None, None, 'm3', 'm1', 'm3', 'm6', 'm3', 'm4', 'm0', None, 'm9'
])
assert np.all(tt_unsorted.from_anchor == [
    None, None, 'end', 'end', 'end', 'end', 'start', 'end', None, None, None
])

tt_sorted = composer.resolve_s_positions(sort=True)
tt_sorted.cols['s from_ from_anchor'].show()
# prints:
# name             s from_ from_anchor
# m0               0 None  None
# m8              10 m0    None
# m4              10 m3    start
# m5              10 m4    end
# m3              10 None  None
# m1              10 m3    end
# m2              10 m1    end
# m6              10 m3    end
# m7              10 m6    end
# m10             10 m9    None
# m9              20 None  None

assert np.all(tt_sorted.name == [
    'm0', 'm8', 'm4', 'm5', 'm3', 'm1', 'm2', 'm6', 'm7', 'm10', 'm9'
])
xo.assert_allclose(tt_sorted.s, [
    0., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20.])
assert np.all(tt_sorted.from_ == [
    None, 'm0', 'm3', 'm4', None, 'm3', 'm1', 'm3', 'm6', 'm9', None
])
assert np.all(tt_sorted.from_anchor == [
    None, None, 'start', 'end', None, 'end', 'end', 'end', 'end', None, None
])
