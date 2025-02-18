import xtrack as xt
import numpy as np

env_ref = xt.load_madx_lattice('../../test_data/pimms/PIMM.seq')
lref = env_ref.pimms

tt_ref = lref.get_table(attr=True)
tt_ref_quad = tt_ref.rows[tt_ref.element_type == 'Quadrupole']


env = xt.Environment()
env.particle_ref = xt.Particles(kinetic_energy0=200e6)

n_mb = 16
env['ang_mb'] = 2.0*np.pi/n_mb
env['l_mb'] = 1.4
env['l_mq'] = 0.3
env.new('mb', xt.RBend, length='l_mb', angle='ang_mb', k0_from_h=True)
env.new('mq', xt.Quadrupole, length='l_mq')

env.vars.default_to_zero = True
env.new('mqfa', 'mq', k1='kqfa')
env.new('mqfb', 'mq', k1='kqfb')
env.new('mqd',  'mq', k1='kqd')


arc = env.new_line(name='arc', length=11., components=[
    env.place('mqfa', at=1.),
    env.place('mb',   at=2.),
    env.place('mqd',  at=3.),
    env.place('mb',   at=4.),
    env.place('mqfa', at=5.),

    env.place('mqfb', at=6.),
    env.place('mb',   at=7.),
    env.place('mqd',  at=8.),
    env.place('mb',   at=9.),
    env.place('mqfb', at=10.),
])

ring = env.new_line(name='ring', components=[arc, -arc, arc, -arc])

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
        xt.TargetSet(qy=1.64, tol=1e-6),
    ]
)
