import xtrack as xt
import numpy as np
import xobjects as xo

edge_model = 'linear'

# TODO check angle column in survey

env = xt.Environment()
env.vars.default_to_zero = True
line = env.new_line(compose=True)
line.new('start', 'Marker', at=0.)
line.new('d1a', 'RBend', length_straight=1.0, k0='k0d1a', anchor='start', at='dz_d1a')
line.new('d1b', 'RBend', length_straight=1.0, k0='k0d1b', anchor='start', at='dz_d1b')
line.new('d2',  'RBend', length_straight=1.0, k0='k0d2',  anchor='start', at='dz_d2')
line.new('end', 'Marker', at='dz_end')

 # ------ measure geometry in the straight reference frame ------

# Positions in the straight reference frame
env['dz_d1a'] = 1.
env['dz_d1b'] = 3.
env['dz_d2'] = 8.
env['dz_end'] = 10.

line.end_compose()
line.set_particle_ref('proton', p0c=1e9)
line.configure_drift_model('exact')
line.set(env.elements.get_table().rows.match(element_type='RBend'),
         model='bend-kick-bend', edge_entry_model=edge_model,
         edge_exit_model=edge_model)

env['k0d1a'] = 'k0d1'
env['k0d1b'] = 'k0d1'

opt = line.match(
    solve=False,
    betx=1, bety=1,
    vary=[xt.VaryList(['k0d1', 'k0d2'], step=1e-5)],
    targets=xt.TargetSet(x=1., px=0.0, at='end'),
)
opt.solve()

# ---- build geometry with curved reference frame ----

# Twiss in the straight reference system
tw0 = line.twiss(betx=1, bety=1, strengths=True)

if edge_model == 'linear':
    # Set fdown angles to match the trajectory (used only for linear edges)
    for nn in ['d1a', 'd1b', 'd2']:
        line[nn].edge_entry_angle_fdown = np.arcsin(tw0['px', nn])
        line[nn].edge_exit_angle_fdown = -np.arcsin(tw0['px', nn + '>>1'])
    tw0 = line.twiss(betx=1, bety=1, strengths=True)
    for nn in ['d1a', 'd1b', 'd2']:
        line[nn].edge_entry_angle_fdown = 0
        line[nn].edge_exit_angle_fdown = 0

line.regenerate_from_composer()

# Update positions according to path length
env['dz_d1a'] = tw0['s', 'd1a'] - tw0['zeta', 'd1a']
env['dz_d1b'] = tw0['s', 'd1b'] - tw0['zeta', 'd1b']
env['dz_d2'] = tw0['s', 'd2'] - tw0['zeta', 'd2']
env['dz_end'] = tw0['s', 'end'] - tw0['zeta', 'end']

# Introduce magnet curvatures
for nn in ['d1a', 'd1b', 'd2']:
    line[nn].k0 = 0
    line[nn].k0_from_h = True
    line[nn].rbend_compensate_sagitta = False
    line[nn].rbend_model = 'straight-body'

d1a_angle_in = np.arcsin(tw0['px', 'd1a'])
d1b_angle_in = np.arcsin(tw0['px', 'd1b'])
d2_angle_in  = np.arcsin(tw0['px', 'd2'])
d1a_angle_out = -d1b_angle_in
d1b_angle_out = -d2_angle_in
d2_angle_out  = -np.arcsin(tw0['px', 'end'])

line['d1a'].angle = d1a_angle_in + d1a_angle_out
line['d1b'].angle = d1b_angle_in + d1b_angle_out
line['d2'].angle  = d2_angle_in  + d2_angle_out

line['d1a'].rbend_angle_diff = d1a_angle_out - d1a_angle_in
line['d1b'].rbend_angle_diff = d1b_angle_out - d1b_angle_in
line['d2'].rbend_angle_diff  = d2_angle_out  - d2_angle_in

# Set rbend shifts
line['d1a'].rbend_shift += line['d1a']._x0_in - tw0['x', 'd1a']
line['d1b'].rbend_shift += line['d1b']._x0_in - tw0['x', 'd1b']
line['d2'].rbend_shift += line['d2']._x0_out - tw0['x', 'end'] # to illustrate that out can be set as well

line.end_compose()

sv = line.survey()
tw = line.twiss(betx=1, bety=1)
sv_back = line.survey(element0='end', X0=sv.X[-1], Y0=sv.Y[-1], Z0=sv.Z[-1],
                     theta0=sv.theta[-1], phi0=sv.phi[-1], psi0=sv.psi[-1])
tw_back = line.twiss(init_at='end', init=tw.get_twiss_init('end'))

# slice for plot
l_sliced =line.copy(shallow=True)
l_sliced.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'))
        ])

sv_sliced = l_sliced.survey()
tw_sliced = l_sliced.twiss(betx=1, bety=1)

# Combine twiss and survey to get actual trajectory
trajectory = sv_sliced.p0 + tw_sliced.x[:, None] * sv_sliced.ex + tw_sliced.y[:, None] * sv_sliced.ey

tw0['path_length'] = tw0.s - tw0.zeta
tw0['diff_path_length'] = np.diff(tw0.path_length, append=tw0.path_length[-1])

xo.assert_allclose(tw0.path_length, tw.s, atol=1e-14)

xo.assert_allclose(tw0['diff_path_length', 'd1a'], line['d1a'].length, atol=1e-14)
xo.assert_allclose(tw0['diff_path_length', 'd1b'], line['d1b'].length, atol=1e-14)
xo.assert_allclose(tw0['diff_path_length', 'd2'], line['d2'].length, atol=1e-14)

xo.assert_allclose(tw0['px', 'd1a'], 0, atol=1e-14)
xo.assert_allclose(tw0['px', 'd1b'], np.sin(line['d1b']._angle_in), atol=1e-14)
xo.assert_allclose(tw0['px', 'd2'], np.sin(line['d2']._angle_in), atol=1e-14)

xo.assert_allclose(tw0['px', 'd1b'], -np.sin(line['d1a']._angle_out), atol=1e-14)
xo.assert_allclose(tw0['px', 'd2'], -np.sin(line['d1b']._angle_out), atol=1e-14)
xo.assert_allclose(tw0['px', 'end'], -np.sin(line['d2']._angle_out), atol=1e-14)

xo.assert_allclose(tw0['x', 'd1a'], line['d1a']._x0_in, atol=1e-14)
xo.assert_allclose(tw0['x', 'd1b'], line['d1b']._x0_in, atol=1e-14)
xo.assert_allclose(tw0['x', 'd2'], line['d2']._x0_in, atol=1e-14)

xo.assert_allclose(tw0['x', 'd1a>>1'], line['d1a']._x0_out, atol=1e-14)
xo.assert_allclose(tw0['x', 'd1b>>1'], line['d1b']._x0_out, atol=1e-14)
xo.assert_allclose(tw0['x', 'd2>>1'], line['d2']._x0_out, atol=1e-14)

xo.assert_allclose(sv.Z, tw0.s, atol=0, rtol=5e-9)
xo.assert_allclose(sv.X, tw0.x, atol=0, rtol=3e-8)
xo.assert_allclose(sv.Y, tw0.y, atol=1e-14)
xo.assert_allclose(sv.theta, np.arcsin(tw0.px), atol=1e-14)
xo.assert_allclose(sv.psi, 0., atol=1e-14)
xo.assert_allclose(sv.phi, 0., atol=1e-14)

xo.assert_allclose(tw.x, 0, atol=1e-14)
xo.assert_allclose(tw.zeta, 0, atol=1e-14)
xo.assert_allclose(tw.y, 0, atol=1e-14)

xo.assert_allclose(tw.betx[-1], tw0.betx[-1], rtol=1e-10)
xo.assert_allclose(tw.bety[-1], tw0.bety[-1], rtol=1e-10)
xo.assert_allclose(tw.alfx[-1], tw0.alfx[-1], rtol=1e-10)
xo.assert_allclose(tw.alfy[-1], tw0.alfy[-1], rtol=1e-10)
xo.assert_allclose(tw.dx[-1], tw0.dx[-1], rtol=1e-10)
xo.assert_allclose(tw.dpx[-1], tw0.dpx[-1], atol=1e-10)

xo.assert_allclose(tw_back.s, tw.s, atol=1e-14)
xo.assert_allclose(tw_back.x, tw.x, atol=1e-14)
xo.assert_allclose(tw_back.y, tw.y, atol=1e-14)
xo.assert_allclose(tw_back.betx, tw.betx, rtol=1e-10)
xo.assert_allclose(tw_back.bety, tw.bety, rtol=1e-10)
xo.assert_allclose(tw_back.alfx, tw.alfx, atol=1e-8)
xo.assert_allclose(tw_back.alfy, tw.alfy, atol=1e-8)
xo.assert_allclose(tw_back.dx, tw.dx, atol=1e-9)
xo.assert_allclose(tw_back.dpx, tw.dpx, atol=1e-9)

xo.assert_allclose(sv_back.s, sv.s, atol=1e-14)
xo.assert_allclose(sv_back.X, sv.X, atol=1e-14)
xo.assert_allclose(sv_back.Y, sv.Y, atol=1e-14)
xo.assert_allclose(sv_back.Z, sv.Z, atol=1e-14)
xo.assert_allclose(sv_back.theta, sv.theta, atol=1e-14)
xo.assert_allclose(sv_back.phi, sv.phi, atol=1e-14)
xo.assert_allclose(sv_back.psi, sv.psi, atol=1e-14)

sv_sliced.cols['s angle theta X'].show()
# name                       s         angle         theta             X
# start                      0             0             0             0
# ||drift_1                  0             0             0             0
# d1a_entry                  1             0             0             0
# d1a..entry_map             1             0             0             0
# d1a..0                     1             0             0             0
# d1a..1               1.33371             0             0             0
# d1a..2               1.66742             0             0             0
# d1a..exit_map        2.00114    -0.0824999             0             0
# d1a_exit             2.00114             0     0.0824999     0.0412734
# ||drift_3            2.00114             0     0.0824999     0.0412734
# d1b_entry            3.00455             0     0.0824999      0.123961
# d1b..entry_map       3.00455     0.0824999     0.0824999      0.123961
# d1b..0               3.00455             0  -1.14732e-17  -1.38778e-17
# d1b..1               3.34056             0  -1.14732e-17  -1.77022e-17
# d1b..2               3.67657             0  -1.14732e-17  -2.15266e-17
# d1b..exit_map        4.01258     -0.165568  -1.14732e-17   -2.5351e-17
# d1b_exit             4.01258             0      0.165568      0.248635
# ||drift_4            4.01258             0      0.165568      0.248635
# d2_entry             8.06804             0      0.165568      0.917026
# d2..entry_map        8.06804      0.165568      0.165568      0.917026
# d2..0                8.06804             0  -9.64632e-18   1.11022e-16
# d2..1                 8.4029             0  -9.64632e-18   1.07807e-16
# d2..2                8.73776             0  -9.64632e-18   1.04591e-16
# d2..exit_map         9.07262   1.38778e-17  -9.64632e-18   1.01376e-16
# d2_exit              9.07262             0  -9.64632e-18             1
# ||drift_5            9.07262             0  -9.64632e-18             1
# end                  10.0726             0  -9.64632e-18             1
# _end_point           10.0726             0  -9.64632e-18             1

xo.assert_allclose(sv_sliced.s[-1], tw0.path_length[-1], atol=0, rtol=1e-14)
xo.assert_allclose(sv_sliced.X[-1], tw0.x[-1], atol=0, rtol=1e-14)
xo.assert_allclose(sv_sliced.Y, 0, atol=1e-14)

xo.assert_allclose(sv_sliced['s', 'd1a..entry_map'], sv['s', 'd1a'], atol=1e-14)
xo.assert_allclose(sv_sliced['s', 'd1b..entry_map'], sv['s', 'd1b'], atol=1e-14)
xo.assert_allclose(sv_sliced['s', 'd2..entry_map'],  sv['s', 'd2'],  atol=1e-14)
xo.assert_allclose(sv_sliced['s', 'd1a..exit_map>>1'], sv['s', 'd1a>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['s', 'd1b..exit_map>>1'], sv['s', 'd1b>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['s', 'd2..exit_map>>1'],  sv['s', 'd2>>1'],  atol=1e-14)

xo.assert_allclose(sv_sliced['theta', 'd1a..entry_map'], sv['theta', 'd1a'], atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd1b..entry_map'], sv['theta', 'd1b'], atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd2..entry_map'],  sv['theta', 'd2'],  atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd1a..exit_map>>1'], sv['theta', 'd1a>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd1b..exit_map>>1'], sv['theta', 'd1b>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd2..exit_map>>1'],  sv['theta', 'd2>>1'],  atol=1e-14)

xo.assert_allclose(sv_sliced['X', 'd1a..entry_map'], sv['X', 'd1a'], atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd1b..entry_map'], sv['X', 'd1b'], atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd2..entry_map'],  sv['X', 'd2'],  atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd1a..exit_map>>1'], sv['X', 'd1a>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd1b..exit_map>>1'], sv['X', 'd1b>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd2..exit_map>>1'],  sv['X', 'd2>>1'],  atol=1e-14)

xo.assert_allclose(sv_sliced['Z', 'd1a..entry_map'], sv['Z', 'd1a'], atol=1e-14)
xo.assert_allclose(sv_sliced['Z', 'd1b..entry_map'], sv['Z', 'd1b'], atol=1e-14)
xo.assert_allclose(sv_sliced['Z', 'd2..entry_map'],  sv['Z', 'd2'],  atol=1e-14)
xo.assert_allclose(sv_sliced['Z', 'd1a..exit_map>>1'], sv['Z', 'd1a>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['Z', 'd1b..exit_map>>1'], sv['Z', 'd1b>>1'], atol=1e-14)
xo.assert_allclose(sv_sliced['Z', 'd2..exit_map>>1'],  sv['Z', 'd2>>1'],  atol=1e-14)

xo.assert_allclose(sv_sliced['theta', 'd1a..entry_map>>1'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd1b..entry_map>>1'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd2..entry_map>>1'],  0.,  atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd1a..exit_map'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd1b..exit_map'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['theta', 'd2..exit_map'],  0.,  atol=1e-14)

xo.assert_allclose(sv_sliced['X', 'd1a..entry_map>>1'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd1b..entry_map>>1'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd2..entry_map>>1'],  0.,  atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd1a..exit_map'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd1b..exit_map'], 0., atol=1e-14)
xo.assert_allclose(sv_sliced['X', 'd2..exit_map'],  0.,  atol=1e-14)

# Twiss checks

xo.assert_allclose(tw_sliced.s, sv_sliced.s, atol=0, rtol=1e-14)

xo.assert_allclose(tw_sliced['x', 'd1a..entry_map'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd1b..entry_map'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd2..entry_map'],  0,  atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd1a..exit_map>>1'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd1b..exit_map>>1'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd2..exit_map>>1'],  0,  atol=1e-14)

xo.assert_allclose(tw_sliced['px', 'd1a..entry_map'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['px', 'd1b..entry_map'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['px', 'd2..entry_map'],  0,  atol=1e-14)
xo.assert_allclose(tw_sliced['px', 'd1a..exit_map>>1'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['px', 'd1b..exit_map>>1'], 0, atol=1e-14)
xo.assert_allclose(tw_sliced['px', 'd2..exit_map>>1'],  0,  atol=1e-14)

xo.assert_allclose(tw_sliced['x', 'd1a..entry_map>>1'], tw0['x','d1a'], atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd1b..entry_map>>1'], tw0['x','d1b'], atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd2..entry_map>>1'],  tw0['x','d2'],  atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd1a..exit_map'], tw0['x','d1a>>1'], atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd1b..exit_map'], tw0['x','d1b>>1'], atol=1e-14)
xo.assert_allclose(tw_sliced['x', 'd2..exit_map'],  tw0['x','d2>>1'],  atol=1e-14)

xo.assert_allclose(tw_sliced.betx[-1], tw0.betx[-1], rtol=1e-10)
xo.assert_allclose(tw_sliced.bety[-1], tw0.bety[-1], rtol=1e-10)
xo.assert_allclose(tw_sliced.alfx[-1], tw0.alfx[-1], rtol=1e-10)
xo.assert_allclose(tw_sliced.alfy[-1], tw0.alfy[-1], rtol=1e-10)
xo.assert_allclose(tw_sliced.dx[-1], tw0.dx[-1], rtol=1e-10)
xo.assert_allclose(tw_sliced.dpx[-1], tw0.dpx[-1], atol=1e-10)

import matplotlib.pyplot as plt
plt.close('all')
tw0.plot('x')
sv_sliced.plot(element_width=4.)
plt.plot(trajectory[:, 2], trajectory[:, 0], color='C1', linestyle='--')
plt.plot(tw0.s, tw0.x, ':', color='C2')

plt.show()
