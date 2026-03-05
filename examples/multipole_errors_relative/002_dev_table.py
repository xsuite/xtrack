import xtrack as xt

# TODO:
# - Slices and replicas

env = xt.Environment()

line = env.new_line(components=[
    env.new('bend', xt.Bend, length=0.1, angle=0.01, at=0.3),
    env.new('rbend', xt.RBend, length_straight=0.1, angle=0.01, at=0.6),
    env.new('quad', xt.Quadrupole, length=0.1, k1=2, at=1),
    env.new('skew_quad', xt.Quadrupole, length=0.1, k1s=2, main_is_skew=True, at=2),
    env.new('sext', xt.Sextupole, length=0.1, k2=3, at=3),
    env.new('skew_sext', xt.Sextupole, length=0.1, k2s=3, main_is_skew=True, at=4),
    env.new('oct', xt.Octupole, length=0.1, k3=4, at=5),
    env.new('skew_oct', xt.Octupole, length=0.1, k3s=4, main_is_skew=True, at=6),
    env.new('multipole', xt.Multipole, length=0.1, knl=[0,0,0,0,2], at=7),
    env.new('skew_multipole', xt.Multipole, length=0.1, ksl=[0,0,0,0,2], main_is_skew=True, at=8),
])


tt = line.get_table(attr=True)