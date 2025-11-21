import xtrack as xt
import numpy as np

env = xt.Environment()
env.vars.default_to_zero = True
line = env.new_line(length=10., components=[
    env.new('start', 'Marker', at=0.),
    env.new('d1a', 'RBend', length_straight=1.0, k0='k0d1a', anchor='start', at=1.),
    env.new('d1b', 'RBend', length_straight=1.0, k0='k0d1b', anchor='start',   at=3.),
    env.new('d2',  'RBend', length_straight=1.0, k0='k0d2',  anchor='start', at=8.),
    env.new('end', 'Marker', at=10.),
])
line.set_particle_ref('proton', p0c=1e9)
line.configure_drift_model('exact')

l_sliced =line.copy(shallow=True)
l_sliced.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Uniform(6, mode='thick'))
        ])

env['k0d1a'] = 'k0d1'
env['k0d1b'] = 'k0d1'

opt = line.match(
    solve=False,
    betx=1, bety=1,
    vary=[xt.VaryList(['k0d1', 'k0d2'], step=1e-5)],
    targets=xt.TargetSet(x=1., px=0.0, at='end'),
)
opt.solve()

tw0 = line.twiss(betx=1, bety=1)

# Introduce magnet curvatures
d1a_angle_in = np.arcsin(tw0['px', 'd1a'])
d1b_angle_in = np.arcsin(tw0['px', 'd1b'])
d2_angle_in  = np.arcsin(tw0['px', 'd2'])
d1a_angle_out = -d1b_angle_in
d1b_angle_out = -d2_angle_in
d2_angle_out  = -np.arcsin(tw0['px', 'end'])

for nn in ['d1a', 'd1b', 'd2']:
    line[nn].k0 = 0
    line[nn].k0_from_h = True
    line[nn].rbend_compensate_sagitta = False
    line[nn].rbend_model = 'straight-body'


line['d1a'].angle = d1a_angle_in + d1a_angle_out
line['d1b'].angle = d1b_angle_in + d1b_angle_out
line['d2'].angle  = d2_angle_in  + d2_angle_out

line['d1a'].rbend_angle_diff = d1a_angle_out - d1a_angle_in
line['d1b'].rbend_angle_diff = d1b_angle_out - d1b_angle_in
line['d2'].rbend_angle_diff  = d2_angle_out  - d2_angle_in

# d1a enters at zero
line['d1a'].rbend_shift += line['d1a']._x0_in

# d2b enters at x defined by twiss
line['d1b'].rbend_shift += line['d1b']._x0_in - tw0['x', 'd1b']

# d2 exits at 1
line['d2'].rbend_shift += line['d1a']._x0_out - 1.

sv = l_sliced.survey()



import matplotlib.pyplot as plt
plt.close('all')
tw0.plot('x')
sv.plot(element_width=4.)

plt.show()
