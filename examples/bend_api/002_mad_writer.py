import xtrack as xt

env = xt.Environment()
line = env.new_line(components=[
    env.new('b0', 'Bend', length=1.0, angle=0.1,),
    env.new('b1', 'Bend', length=1.0, angle=0.1, k0=0.01),
    env.new('b2', 'RBend', length_straight=1.0, angle=0.1,),
    env.new('b3', 'RBend', length_straight=1.0, angle=0.1, k0=0.01),
    ])

mad_src = line.to_madx_sequence('seq')

env2 = xt.load(string=mad_src, format='madx')
line2 = env2.lines['seq']

for nn in ['b0', 'b1', 'b2', 'b3']:
    el1 = line[nn]
    el2 = line2[nn]
    assert el1.length == el2.length
    assert el1.angle == el2.angle
    assert el1.k0 == el2.k0
