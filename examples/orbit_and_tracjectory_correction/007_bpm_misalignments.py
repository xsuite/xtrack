import xtrack as xt
import xobjects as xo

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=450e9, mass0=xt.PROTON_MASS_EV, q0=1)
env.vars.default_to_zero = True

env.new('mq', 'Quadrupole', length=0.8)
env.new('bpm', 'Marker')
env.new('corrector', 'Multipole', knl=[0])

line = env.new_line(components=[

    env.new('line.start', 'Marker'),
    env.new('line.end', 'Marker', at=10.),

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
])

env['kq1'] = 0.02
env['kq2'] = -0.02
env['kq3'] = 0.02
env['kq4'] = -0.02

bpm_alignment ={
    'bpm1': {'shift_x': 1e-3, 'shift_y': 2e-3},
    'bpm2': {'shift_x': 1e-3, 'shift_y': 2e-3},
    'bpm3': {'shift_x': 1e-3, 'shift_y': 2e-3},
    'bpm4': {'shift_x': 1e-3, 'shift_y': 2e-3},
}

tw0 = line.twiss(betx=100, bety=80)

env.set(['mq1', 'mq2', 'mq3', 'mq4'], shift_x=1e-3, shift_y=2e-3)

# Going through the center of all quads
tw = line.twiss(betx=100, bety=80, x=1e-3, y=2e-3)

line.steering_monitors_x = ['bpm1', 'bpm2', 'bpm3', 'bpm4']
line.steering_monitors_y = ['bpm1', 'bpm2', 'bpm3', 'bpm4']
line.steering_correctors_x = ['corrector1', 'corrector3', ]
line.steering_correctors_y = ['corrector2', 'corrector4']


correction = line.correct_trajectory(twiss_table=tw0,
                                     x_init=1e-3, y_init=2e-3,
                                     start='line.start', end='line.end',
                                     monitor_alignment=bpm_alignment,
                                     run=False)

correction.correct(n_iter=1)

xo.assert_allclose(correction.x_correction.shift_x_monitors, 1e-3, rtol=0, atol=1e-14)
xo.assert_allclose(correction.x_correction.shift_y_monitors, 2e-3, rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction.shift_x_monitors, 1e-3, rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction.shift_y_monitors, 2e-3, rtol=0, atol=1e-14)

# Data from previous step can be found in:
xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-14)