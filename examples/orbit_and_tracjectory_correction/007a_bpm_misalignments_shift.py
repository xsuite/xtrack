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
    env.new('line.end', 'Marker', at=12.),

    env.new('mq1', 'mq', k1='kq1', at=3.),
    env.new('mq2', 'mq', k1='kq2', at=5.),
    env.new('mq3', 'mq', k1='kq3', at=7.),
    env.new('mq4', 'mq', k1='kq4', at=9.),

    env.new('bpm.s.1', 'bpm', at=0.1),
    env.new('bpm.s.2', 'bpm', at=0.5),
    env.new('bpm.q.1', 'bpm', at='mq1@start'),
    env.new('bpm.q.2', 'bpm', at='mq2@start'),
    env.new('bpm.q.3', 'bpm', at='mq3@start'),
    env.new('bpm.q.4', 'bpm', at='mq4@start'),
    env.new('bpm.e.1', 'bpm', at=11.5),
    env.new('bpm.e.2', 'bpm', at=11.9),

    env.new('corrector1', 'corrector', at='mq1@start'),
    env.new('corrector2', 'corrector', at='mq2@start'),
    env.new('corrector3', 'corrector', at='mq3@start'),
    env.new('corrector4', 'corrector', at='mq4@start'),

    env.new('bumper1', 'corrector', at=1., knl=['k0l_bumper1'], ksl=['k0sl_bumper1']),
    env.new('bumper2', 'corrector', at=2., knl=['k0l_bumper2'], ksl=['k0sl_bumper2']),
    env.new('bumper3', 'corrector', at=10., knl=['k0l_bumper3'], ksl=['k0sl_bumper3']),
    env.new('bumper4', 'corrector', at=11., knl=['k0l_bumper4'], ksl=['k0sl_bumper4']),
])

env.set(['mq1', 'mq2', 'mq3', 'mq4'], shift_x=1e-3, shift_y=2e-3)

# # Steer to enter at the center of the first quad
# line.match(
#     betx=1., bety=1.,
#     vary=xt.VaryList(['k0l_bumper1', 'k0l_bumper2', 'k0sl_bumper1', 'k0sl_bumper2'],
#                      step=1e-6),
#     targets=xt.TargetSet(x=1e-3, px=0, y=2e-3, py=0, at='bpm1'),
# )

# # Steer to bring the beam back
# opt2 = line.match(
#     start='bumper3', end='line.end',
#     betx=1., bety=1., x=1e-3, y=2e-3,
#     vary=xt.VaryList(['k0l_bumper3', 'k0l_bumper4', 'k0sl_bumper3', 'k0sl_bumper4'],
#                      step=1e-6),
#     targets=xt.TargetSet(x=0e-3, px=0, y=0e-3, py=0, at='line.end'),
# )

env['kq1'] = 0.2
env['kq2'] = -0.2
env['kq3'] = 0.2
env['kq4'] = -0.2

bpm_alignment ={
    'bpm.q.1': {'shift_x': 1e-3, 'shift_y': 2e-3},
    'bpm.q.2': {'shift_x': 1e-3, 'shift_y': 2e-3},
    'bpm.q.3': {'shift_x': 1e-3, 'shift_y': 2e-3},
    'bpm.q.4': {'shift_x': 1e-3, 'shift_y': 2e-3},
}

tw0 = line.twiss4d()

# Going through the center of all quads
tw = line.twiss4d()

line.steering_monitors_x = ['bpm.s.1', 'bpm.s.2',
                            'bpm.q.1', 'bpm.q.2', 'bpm.q.3', 'bpm.q.4',
                            'bpm.e.1', 'bpm.e.2']
line.steering_monitors_y = line.steering_monitors_x
line.steering_correctors_x = ['bumper1', 'bumper2', 'bumper3', 'bumper4']
line.steering_correctors_y = ['bumper1', 'bumper2', 'bumper3', 'bumper4']

correction = line.correct_trajectory(twiss_table=tw0,
                                     monitor_alignment=bpm_alignment,
                                     run=False)

correction.correct()

#!end-doc-part

tw_corr = line.twiss4d()

xo.assert_allclose(correction.x_correction.shift_x_monitors,
                   [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
xo.assert_allclose(correction.x_correction.shift_y_monitors,
                   [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction.shift_x_monitors,
                   [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction.shift_y_monitors,
                   [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)

# Data from previous step can be found in:
correction.correct() # Some more steps to log the position
xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-12)
xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-12)

correction.clear_correction_knobs()

for nn in ['bumper1', 'bumper2', 'bumper3', 'bumper4']:
    assert line[nn].knl[0] == 0

correction.thread(ds_thread = 10)
tw_thread = line.twiss4d()

xo.assert_allclose(correction.x_correction.shift_x_monitors,
                   [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
xo.assert_allclose(correction.x_correction.shift_y_monitors,
                   [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction.shift_x_monitors,
                   [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
xo.assert_allclose(correction.y_correction.shift_y_monitors,
                   [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)

# Data from previous step can be found in:
correction.correct() # Some more steps to log the position
xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-12)
xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-12)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1)
tw_corr.plot('x y', figure=fig1)
plt.suptitle('After correction')

fig2 = plt.figure(2)
tw_thread.plot('x y', figure=fig2)
plt.suptitle('After threading')

plt.show()

