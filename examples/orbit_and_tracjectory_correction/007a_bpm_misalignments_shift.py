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

    env.new('mq1', 'mq', k1=0.2, at=3.),
    env.new('mq2', 'mq', k1=-0.2, at=5.),
    env.new('mq3', 'mq', k1=0.2, at=7.),
    env.new('mq4', 'mq', k1=-0.2, at=9.),

    env.new('bpm.s.1', 'bpm', at=0.1),
    env.new('bpm.s.2', 'bpm', at=0.5),
    env.new('bpm.q.1', 'bpm', at='mq1@start'),
    env.new('bpm.q.2', 'bpm', at='mq2@start'),
    env.new('bpm.q.3', 'bpm', at='mq3@start'),
    env.new('bpm.q.4', 'bpm', at='mq4@start'),
    env.new('bpm.e.1', 'bpm', at=11.5),
    env.new('bpm.e.2', 'bpm', at=11.9),

    env.new('corr1', 'corrector', at=1., knl=['k0l_corr1'], ksl=['k0sl_corr1']),
    env.new('corr2', 'corrector', at=2., knl=['k0l_corr2'], ksl=['k0sl_corr2']),
    env.new('corr3', 'corrector', at=10., knl=['k0l_corr3'], ksl=['k0sl_corr3']),
    env.new('corr4', 'corrector', at=11., knl=['k0l_corr4'], ksl=['k0sl_corr4']),
])

# Define monitors and correctors for orbit steering
line.steering_monitors_x = ['bpm.s.1', 'bpm.s.2',
                            'bpm.q.1', 'bpm.q.2', 'bpm.q.3', 'bpm.q.4',
                            'bpm.e.1', 'bpm.e.2']
line.steering_correctors_x = ['corr1', 'corr2', 'corr3', 'corr4']
line.steering_monitors_y = line.steering_monitors_x
line.steering_correctors_y = line.steering_correctors_x

# Twiss without misalignments
tw0 = line.twiss4d()

# Misalign all quadrupoles
env.set(['mq1', 'mq2', 'mq3', 'mq4'], shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.1)

# Define BPM alignment
bpm_alignment ={
    'bpm.q.1': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
    'bpm.q.2': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
    'bpm.q.3': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
    'bpm.q.4': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
}

# Correct orbit taking into account BPM alignment (centers the beam in all quadrupoles)
correction = line.correct_trajectory(twiss_table=tw0,
                                     monitor_alignment=bpm_alignment, # <--BPM alignment
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
xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-10)
xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-10)

correction.clear_correction_knobs()

for nn in ['corr1', 'corr2', 'corr3', 'corr4']:
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
xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-9)
xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-9)

xo.assert_allclose(tw_corr.rows['mq1':'corr3'].x, 1e-3, rtol=0, atol=1e-9)
xo.assert_allclose(tw_corr.rows['mq1':'corr3'].y, 2e-3, rtol=0, atol=1e-9)
xo.assert_allclose(tw_thread.rows['mq1':'corr3'].x, 1e-3, rtol=0, atol=1e-4)
xo.assert_allclose(tw_thread.rows['mq1':'corr3'].y, 2e-3, rtol=0, atol=1e-4)

xo.assert_allclose(tw_corr.x[0], 0, rtol=0, atol=1e-9)
xo.assert_allclose(tw_corr.y[0], 0, rtol=0, atol=1e-9)
xo.assert_allclose(tw_thread.x[0], 0, rtol=0, atol=1e-4)
xo.assert_allclose(tw_thread.y[0], 0, rtol=0, atol=1e-4)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1)
tw_corr.plot('x y', figure=fig1, grid=False)
plt.suptitle('After correction')

fig2 = plt.figure(2)
tw_thread.plot('x y', figure=fig2, grid=False)
plt.suptitle('After threading')

plt.show()

