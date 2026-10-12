import xtrack as xt
import numpy as np

env = xt.get_environment()
env.particle_ref = xt.Particles(kinetic_energy0=200e6)
env.vars.default_to_zero = True

############
# Elements #
############

# Element geometry
n_bends = 16
env['ang_mb'] = 2 * np.pi / n_bends
env['l_mb'] = 1.65
env['l_mq'] = 0.35
env['l_ms'] = 0.2

# Magnet pipes
env.new('mb', xt.RBend, length_straight='l_mb', angle='ang_mb')
env.new('mq', xt.Quadrupole, length='l_mq')
env.new('ms', xt.Sextupole, length='l_ms')

# Quadrupole families
env.new('qfa', 'mq', k1='kqfa')
env.new('qfb', 'mq', k1='kqfb')
env.new('qd', 'mq', k1='kqd')

# Magnet instances
env.new('msf.1', 'ms', k2='ksf')
env.new('msf.2', 'ms', k2='ksf')
env.new('msd.1', 'ms', k2='ksd')
env.new('msd.2', 'ms', k2='ksd')
env.new('mse', 'ms', k2='kse')

# RF cavity
env.new('rf1', xt.Cavity, voltage='vrf', frequency='frf')

###########
# Lattice #
###########

# Cells
cell_a = env.new_line(
    name='cell_a',
    length=7.405,
    components=[
        env.place('qfa', at=0.3875),
        env.place('mb', at=1.8125),
        env.place('qd', at=3.2925),
        env.place('mb', at=5.0475),
        env.place('qfa', at=6.3275),
    ],
)

cell_b = env.new_line(
    name='cell_b',
    length=8.405,
    components=[
        env.place('qfb', at=1.2725),
        env.place('mb', at= 2.7275),
        env.place('qd', at=4.8575),
        env.place('mb', at=6.5125),
        env.place('qfb', at=7.7925),
    ],
)

# Arc
arc = cell_a + cell_b

# Straight sections
long_straight = env.new_line(
    length=2,
    components=[
        env.new('mid.lss', xt.Marker, at=1.)
    ],
)

short_straight = env.new_line(
    length=1,
    components=[
        env.new('mid.sss', xt.Marker, at=1.)
    ],
)

# Ring
ring = 2 * (long_straight + arc + short_straight - arc)

# Assign unique names to all elements
ring.replace_all_repeated_elements()

# Insert sextupoles
ring.insert([
    env.place('msf.1', at=-0.2, from_='qfb.0@start'),
    env.place('msf.2', at=-0.2, from_='qfb.4@start'),
    env.place('msd.1', at=0.3, from_='qd.2@end'),
    env.place('msd.2', at=0.3, from_='qd.6@end'),
    env.place('mse', at=-0.3, from_='qfa.4@start')
])

# Insert RF
ring.insert('rf1', at=0.5, from_='qfa.3@start')

# Select lines to keep
env['ring'] = ring

env.vars.default_to_zero = False


#############
# Strengths #
#############
env.vars.update({
    'kqfa': 0.33773079026867847,
    'kqfb': 0.5469421227593386,
    'kqd': -0.5904782303561069,
    'ksf': 0.5835000891312914,
    'ksd': -0.7807697760187184,
})


#############
# Apertures #
#############
ring_table = ring.get_table()

profile_straight = env.aperture.new_profile(
    'straight', 'Ellipse', half_major=0.07, half_minor=0.037)
profile_dipole = env.aperture.new_profile(
    'dipole', 'Ellipse', half_major=0.07, half_minor=0.032)

l_mb = float(env['mb'].length)
l_mq = float(env['mq'].length)
l_ms = float(env['ms'].length)
ang_mb = float(env['mb'].angle)

pipe_mb = env.aperture.new_pipe(
    'mb',
    curvature=ang_mb / l_mb,
    positions=[
        env.aperture.profiles.place(profile_dipole, at=0.0),
        env.aperture.profiles.place(profile_dipole, at=l_mb),
    ],
)
pipe_mq = env.aperture.new_pipe(
    'mq',
    positions=[
        env.aperture.profiles.place(profile_straight, at=0.0),
        env.aperture.profiles.place(profile_straight, at=l_mq),
    ],
)
pipe_ms = env.aperture.new_pipe(
    'ms',
    positions=[
        env.aperture.profiles.place(profile_straight, at=0.0),
        env.aperture.profiles.place(profile_straight, at=l_ms),
    ],
)

drift_pipes = {}

for row in ring_table.rows:
    name = row.name
    s_start = row.s_start
    s_end = row.s_end
    length = s_end - s_start

    if name.startswith('mb'):
        ring.aperture.place_pipe(name, pipe_mb, at=name)
    elif name.startswith('q'):
        ring.aperture.place_pipe(name, pipe_mq, at=name)
    elif name.startswith('ms'):
        ring.aperture.place_pipe(name, pipe_ms, at=name)
    elif length > 1e-6:
        key = round(length / 1e-6)
        if key in drift_pipes:
            drift_pipe = drift_pipes[key]
        else:
            drift_pipe = env.aperture.new_pipe(
                f'drift_{length:.6f}',
                positions=[
                    env.aperture.profiles.place(profile_straight, at=0.0),
                    env.aperture.profiles.place(profile_straight, at=length),
                ],
            )
            drift_pipes[key] = drift_pipe
        ring.aperture.place_pipe(name, drift_pipe, at=name)

ring.aperture.build_model()

ring.twiss_default['method'] = '4d'
