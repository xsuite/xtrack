import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

env.vars({
    'l.b1': 1.0,
    'l.q1': 0.5,
    's.ip': 10,
    's.left': -5,
    's.right': 5,
    'l.before_right': 1,
    'l.after_left2': 0.5,
})

# names, tab_sorted = handle_s_places(seq)
line = env.new_line(components=[
    env.new('b1', xt.Bend, length='l.b1'),
    env.new('q1', xt.Quadrupole, length='l.q1'),
    env.new('ip', xt.Marker, at='s.ip'),
    (
        env.new('before_before_right', xt.Marker),
        env.new('before_right', xt.Sextupole, length=1),
        env.new('right',xt.Quadrupole, length=0.8, at='s.right', from_='ip'),
        env.new('after_right', xt.Marker),
        env.new('after_right2', xt.Marker),
    ),
    env.new('left', xt.Quadrupole, length=1, at='s.left', from_='ip'),
    env.new('after_left', xt.Marker),
    env.new('after_left2', xt.Bend, length='l.after_left2'),
])

tt = line.get_table(attr=True)
tt['s_center'] = tt['s'] + tt['length']/2
assert np.all(tt.name == np.array([
    'b1', 'q1', 'drift_1', 'left', 'after_left', 'after_left2',
    'drift_2', 'ip', 'drift_3', 'before_before_right', 'before_right',
    'right', 'after_right', 'after_right2', '_end_point']))

xo.assert_allclose(env['b1'].length, 1.0, rtol=0, atol=1e-14)
xo.assert_allclose(env['q1'].length, 0.5, rtol=0, atol=1e-14)
xo.assert_allclose(tt['s', 'ip'], 10, rtol=0, atol=1e-14)
xo.assert_allclose(tt['s', 'before_before_right'], tt['s', 'before_right'],
                   rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'before_right'] - tt['s_center', 'right'],
                   -(1 + 0.8)/2, rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'right'] - tt['s', 'ip'], 5, rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'after_right'] - tt['s_center', 'right'],
                     0.8/2, rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'after_right2'] - tt['s_center', 'right'],
                     0.8/2, rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'left'] - tt['s_center', 'ip'], -5,
                   rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'after_left'] - tt['s_center', 'left'], 1/2,
                     rtol=0, atol=1e-14)
xo.assert_allclose(tt['s_center', 'after_left2'] - tt['s_center', 'after_left'],
                   0.5/2, rtol=0, atol=1e-14)


import matplotlib.pyplot as plt
plt.close('all')
line.survey().plot()

plt.show()