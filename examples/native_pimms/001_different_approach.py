import xtrack as xt
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(kinetic_energy0=200e6)



n_bends = 16
env['ang_mb'] = 2*np.pi/n_bends
env['l_mb'] = 1.65
env['l_mq'] = 0.35

env.vars.default_to_zero = True
env.new('mb', xt.RBend,      length='l_mb', angle='ang_mb', k0_from_h=True)
env.new('mq', xt.Quadrupole, length='l_mq')


env.new('qfa', 'mq', k1= 'kqfa')
env.new('qfb', 'mq', k1= 'kqfb')
env.new('qd',  'mq', k1= 'kqd')


cell_a = env.new_line(name='cell_a', length=75.24/8 - 2.-1, components=[
    env.place('qfa', at=0.3875),
    env.place('mb', at=1.8125),
    env.place('qd', at=3.2925),
    env.place('mb', at=5.0475),
    env.place('qfa', at=6.3275),
])

cell_b = env.new_line(name='cell_b', length=75.24/8, components=[
    env.place('qfb', at=1.2725),
    env.place('mb', at= 2.7275),
    env.place('qd', at=4.8575),
    env.place('mb', at=6.5125),
    env.place('qfb', at=7.7925),
])

ss = env.new_line(name='ss', length=2., components=[
    env.new('mid.ss', xt.Marker, at=1.)
])

quarter_ring = ss + cell_a + cell_b

    # env.place('qfb', at=20.2475),
    # env.place('mb', at=21.7025),
    # env.place('qd', at=23.1825),
    # env.place('mb', at=25.4875),
    # env.place('qfb', at=26.7675),
    # env.place('qfa', at=29.1175),
    # env.place('mb', at=30.5725),
    # env.place('qd', at=32.1525),
    # env.place('mb', at=33.8075),
    # env.place('qfa', at=35.0575),
    # env.place('qfa', at=40.0075),
    # env.place('mb', at=41.4325),
    # env.place('qd', at=42.9125),
    # env.place('mb', at=44.6675),
    # env.place('qfa', at=45.9475),
    # env.place('qfb', at=48.2975),
    # env.place('mb', at=49.7525),
    # env.place('qd', at=51.8825),
    # env.place('mb', at=53.5375),
    # env.place('qfb', at=54.8175),
    # env.place('qfb', at=57.8675),
    # env.place('mb', at=59.3225),
    # env.place('qd', at=60.8025),
    # env.place('mb', at=63.1075),
    # env.place('qfb', at=64.3875),
    # env.place('qfa', at=66.7375),
    # env.place('mb', at=68.1925),
    # env.place('qd', at=69.7725),
    # env.place('mb', at=71.4275),
    # env.place('qfa', at=72.6775),
# ])

ring = env.new_line(name='ring', components=[
    quarter_ring, -quarter_ring, quarter_ring, -quarter_ring])

ring.insert('extr', obj=xt.Marker(), at=quarter_ring.get_length()*2)

env['kqfa'] = 0.01
env['kqd'] = -0.02
env['kqfb'] = 0.01

# First twiss
tw = ring.twiss4d(compute_chromatic_properties=False)


env['kqfb'] = 'kqfa'
opt = ring.match(
    solve=False, # <- prepare the match without running it
    compute_chromatic_properties=False,
    method='4d',
    vary=[
        xt.Vary('kqfa', limits=(0, 10),  step=1e-3),
        xt.Vary('kqd', limits=(-10, 0), step=1e-3),
    ],
    targets=[
        xt.TargetSet(qx=1.64, qy=1.72, tol=1e-6),
    ]
)

opt.solve()

opt = ring.match(
    solve=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=[
        xt.VaryList(['kqfa', 'kqfb'], limits=(0, 1),  step=1e-3),
        xt.Vary('kqd', limits=(-1, 0), step=1e-3),
    ],
    targets=[
        xt.TargetSet(qx=1.663, qy=1.72, tol=1e-6),
        xt.Target(dx=0, at='extr', tol=1e-6)
    ]
)

opt.solve()
