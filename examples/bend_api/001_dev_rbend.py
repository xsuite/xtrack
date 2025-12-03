import xtrack as xt

bend1 = xt.RBend()
bend1.angle = 0.1
bend1.length_straight= 1.0

bend2 = xt.RBend()
bend2.length_straight= 1.0
bend2.angle = 0.1

bend3 = xt.RBend(angle=0.1, length_straight=1.0)

bend1d = xt.RBend.from_dict(bend1.to_dict())
bend2d = xt.RBend.from_dict(bend2.to_dict())
bend3d = xt.RBend.from_dict(bend3.to_dict())

for bb in [bend1, bend2, bend3, bend1d, bend2d, bend3d]:
    assert bb.h == 0.1 / bb.length
    assert bb.k0 == 'from_h'
    assert bb.k0_from_h == True
    assert bb.length_straight == 1.0
    assert bb.length > 1.0
    assert bb.angle == 0.1

bend1.k0 = 0.2
assert bend1.k0 == 0.2
assert bend1.h == 0.1 / bend1.length
assert bend1.k0_from_h == False

bend2.k0_from_h = False
assert bend2.k0 == 0.
assert bend2.h == 0.1 / bend2.length
assert bend2.k0_from_h == False