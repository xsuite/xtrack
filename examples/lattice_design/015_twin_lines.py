import xtrack as xt

env = xt.Environment()

env.vars.default_to_zero = True

line = env.new_line(
    components=[
        env.new('c1', xt.Multipole, knl=['k0.1'], length=0.1, at=1.0),
        env.new('c2', xt.Multipole, knl=['k0.2'], length=0.1, at=2.0),
        env.new('obstacle', xt.LimitRect, min_x=-0.15, max_x=-0.05, at=2.5),
        env.new('c3', xt.Multipole, knl=['k0.3'], length=0.1, at=3.0),
        env.new('c4', xt.Multipole, knl=['k0.4'], length=0.1, at=4.0),
        env.new('end', xt.Marker, at=5.),
    ]
)
line.particle_ref = xt.Particles(p0c=10e9)
line.build_tracker()

line.get_table().show() # gives:
# Table: 10 rows, 11 cols
# name                   s element_type isthick isreplica parent_name ...
# drift_1                0 Drift           True     False None
# c1                     1 Multipole      False     False None
# drift_2                1 Drift           True     False None
# c2                     2 Multipole      False     False None
# obstacle               2 LimitRect      False     False None
# drift_3                2 Drift           True     False None
# c3                     3 Multipole      False     False None
# drift_4                3 Drift           True     False None
# c4                     4 Multipole      False     False None
# _end_point             4                False     False None

# Track a particle
p = line.particle_ref.copy()
line.track(p)

# Particle is lost at the obstacle
p.get_table().show(cols=['state', 'at_element'])
# is:
# particle_id           state at_element
# 0                         0          4

# Build a twin line for twiss
ltwiss = line.copy(shallow=True)

# Remove the obstacle
env.new('obstacle_marker', xt.Marker)
ltwiss.replace('obstacle', 'obstacle_marker')

# Match a bump to avoid the obstacle
opt = ltwiss.match(
    solve=False,
    betx=1.0, bety=1.0, x=0, px=0, y=0, py=0,
    targets=[
        xt.TargetSet(x=-0.1, px=0, at='obstacle_marker'),
        xt.TargetSet(x=0, px=0, at=xt.END),
    ],
    vary=xt.VaryList(['k0.1', 'k0.2', 'k0.3', 'k0.4'], step=1e-4)
)
opt.solve()

tw = ltwiss.twiss(betx=1.0, bety=1.0)
tw.plot('x')

# As all elements apart from the obstacle are in common, the bump is applied
# also in the original line

p = line.particle_ref.copy()
line.build_tracker()
line.track(p)

# The particle goes around the obstacle and is not lost:
p.get_table().show(cols=['state', 'at_element'])
# is:
# particle_id           state at_element
# 0                         1          0
