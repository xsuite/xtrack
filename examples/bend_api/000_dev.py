import xtrack as xt

bend1 = xt.Bend()
bend1.angle = 0.1
bend1.length = 1.0

bend2 = xt.Bend()
bend2.length = 1.0
bend2.angle = 0.1

bend3 = xt.Bend(angle=0.1, length=1.0)

bend1d = xt.Bend.from_dict(bend1.to_dict())
bend2d = xt.Bend.from_dict(bend2.to_dict())
bend3d = xt.Bend.from_dict(bend3.to_dict())

for bb in [bend1, bend2, bend3, bend1d, bend2d, bend3d]:
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
assert bend2.k0 == 0.
assert bend2.h == 0.1
assert bend2.k0_from_h == False


