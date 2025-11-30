import xtrack as xt

bend1 = xt.Bend()
bend1.angle = 0.1
bend1.length = 1.0

bend2 = xt.Bend()
bend2.length = 1.0
bend2.angle = 0.1

bend3 = xt.Bend(angle=0.1, length=1.0)

for bb in [bend1, bend2, bend3]:
    assert bb.h == 0.1
    assert bb.k0 == 'from_h'
    assert bb.k0_from_h == True
    assert bb.length == 1.0
    assert bb.angle == 0.1

bend1.k0 = 0.2
assert bend1.k0 == 0.2
assert bend1.h == 0.1
assert bend1.k0_from_h == False

bend2.k0_from_h = False
assert bend2.k0 == 0.1
assert bend2.h == 0.1
assert bend2.k0_from_h == False
