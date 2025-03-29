import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=450e9, mass0=xt.PROTON_MASS_EV, q0=1)
env.vars.default_to_zero = True

env.new('mq', 'Quadrupole', length=0.8)
env.new('bpm', 'Marker')
env.new('corrector', 'Multipole', knl=[0])

line = env.new_line(components=[

    env.new('line.start', 'Marker'),
    env.new('line.end', 'Marker', at=12.),

    env.new('mq1', 'mq', k1='kq1', at=3.),
    env.new('mq2', 'mq', k1='kq2', at=5.),
    env.new('mq3', 'mq', k1='kq3', at=7.),
    env.new('mq4', 'mq', k1='kq4', at=9.),

    env.new('bpm1', 'bpm', at='mq1@start'),
    env.new('bpm2', 'bpm', at='mq2@start'),
    env.new('bpm3', 'bpm', at='mq3@start'),
    env.new('bpm4', 'bpm', at='mq4@start'),

    env.new('corrector1', 'corrector', at='mq1@start'),
    env.new('corrector2', 'corrector', at='mq2@start'),
    env.new('corrector3', 'corrector', at='mq3@start'),
    env.new('corrector4', 'corrector', at='mq4@start'),

    env.new('bumper1', 'corrector', at=1., knl=['k0l_bumper1'], ksl=['k0sl_bumper1']),
    env.new('bumper2', 'corrector', at=2., knl=['k0l_bumper2'], ksl=['k0sl_bumper2']),
])

env.set(['mq1', 'mq2', 'mq3', 'mq4'], shift_x=1e-3, shift_y=1.5e-3,
        rot_s_rad=np.deg2rad(30.))

# Steer to enter at the center of the first quad
line.match(
    betx=1., bety=1.,
    vary=xt.VaryList(['k0l_bumper1', 'k0l_bumper2', 'k0sl_bumper1', 'k0sl_bumper2'],
                     step=1e-6),
    targets=xt.TargetSet(x=1e-3, px=0, y=1.5e-3, py=0, at='bpm1'),
)

tt = line.get_table()
tt_quad = tt.rows['mq.*']

env['kq1'] = 0.02
env['kq2'] = -0.02
env['kq3'] = 0.02
env['kq4'] = -0.02

dx = 2e-3

bpm_alignment ={
    'bpm1': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
    'bpm2': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
    'bpm3': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
    'bpm4': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
}

tw0 = line.twiss(betx=100, bety=80)

line.steering_monitors_x = ['bpm1', 'bpm2', 'bpm3', 'bpm4']
line.steering_monitors_y = ['bpm1', 'bpm2', 'bpm3', 'bpm4']
line.steering_correctors_x = ['corrector1', 'corrector3', ]
line.steering_correctors_y = ['corrector2', 'corrector4']

tw = line.twiss(betx=1, bety=1, x=1e-3+dx*np.cos(np.deg2rad(30)), y=1.5e-3+dx*np.sin(np.deg2rad(30)))


correction = line.correct_trajectory(twiss_table=tw0,
                                     x_init=dx * np.cos(np.deg2rad(30)),
                                     y_init=dx * np.sin(np.deg2rad(30)),
                                     start='line.start', end='line.end',
                                     monitor_alignment=bpm_alignment,
                                     run=False)

correction.correct(n_iter=1)

# Check that there is no vertical reading in the tilted bpm BPMs
xo.assert_allclose(correction.y_correction._position_before, 0, atol=1e-15, rtol=0)
