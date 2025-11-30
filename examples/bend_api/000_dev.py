import xtrack as xt

bend1 = xt.Bend()
bend1.angle = 0.1
bend1.length = 1.0

bend2 = xt.Bend()
bend2.length = 1.0
bend2.angle = 0.1

assert bend1.h == 0.1
assert bend2.h == 0.1