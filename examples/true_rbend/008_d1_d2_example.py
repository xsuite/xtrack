import xtrack as xt
import numpy as np
import xobjects as xo

edge_model = 'linear' # linear of full

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
if edge_model == 'linear':
    tw_back = line.twiss(init_at='end', init=tw.get_twiss_init('end'))

# slice for plot
l_sliced =line.copy(shallow=True)
l_sliced.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'))
        ])

sv_sliced = l_sliced.survey()
tw_sliced = l_sliced.twiss(betx=1, bety=1)
sv_sliced_back = l_sliced.survey(element0='end',
                                 X0=sv_sliced.X[-1], Y0=sv_sliced.Y[-1], Z0=sv_sliced.Z[-1],
                                 theta0=sv_sliced.theta[-1], phi0=sv_sliced.phi[-1],
                                 psi0=sv_sliced.psi[-1])
if edge_model == 'linear':
    tw_sliced_back = l_sliced.twiss(init_at='end',
                                    init=tw_sliced.get_twiss_init('end'))

# Combine twiss and survey to get actual trajectory
trajectory = sv_sliced.XYZ + tw_sliced.x[:, None] * sv_sliced.ex + tw_sliced.y[:, None] * sv_sliced.ey

tw0['path_length'] = tw0.s - tw0.zeta
tw0['diff_path_length'] = np.diff(tw0.path_length, append=tw0.path_length[-1])


# Twiss checks


import matplotlib.pyplot as plt
plt.close('all')
tw0.plot('x')
sv_sliced.plot(element_width=4.)
plt.plot(trajectory[:, 2], trajectory[:, 0], color='C1', linestyle='--')
plt.plot(tw0.s, tw0.x, ':', color='C2')

plt.show()
