import xtrack as xt
import numpy as np
import xobjects as xo

# Need to check what happens with rot_s_rad (v bend)
# Need to check diffrent element0

env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

edge_model = 'full'

line = env.new_line(length=5, components=[
    env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
            rot_s_rad=np.pi/2,
            model='bend-kick-bend',
            rbend_model='straight-body',
            edge_entry_model=edge_model, edge_exit_model=edge_model,
            at=2.5)])
line.insert('start', xt.Marker(), at=0)
line.append('end', xt.Marker())
line.config.XTRACK_USE_EXACT_DRIFTS = True

line_no_slice = line.copy(shallow=True)

line.slice_thick_elements(
        slicing_strategies=[
            # Slicing with thin elements
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Teapot(1000), element_type=xt.RBend),
        ])

line.insert('mid', xt.Marker(), at=2.5)

line['mb'].rbend_model = 'straight-body'
sv_straight = line.survey(element0='mid', Y0=-line['mb'].sagitta/2)
tt_straight = line.get_table(attr=True)
tw_straight = line.twiss(betx=1, bety=1)
p_straight = (sv_straight.p0 + tw_straight.x[:, None] * sv_straight['ex']
                             + tw_straight.y[:, None] * sv_straight['ey'])
tw_straight['X'] = p_straight[:, 0]
tw_straight['Y'] = p_straight[:, 1]
tw_straight['Z'] = p_straight[:, 2]

sv_straight_start = line.survey(element0='start',
                                X0=sv_straight['X', 'start'],
                                Y0=sv_straight['Y', 'start'],
                                Z0=sv_straight['Z', 'start'],
                                theta0=sv_straight['theta', 'start'],
                                phi0=sv_straight['phi', 'start'],
                                psi0=sv_straight['psi', 'start'])
sv_straight_end = line.survey(element0='end',
                                X0=sv_straight['X', 'end'],
                                Y0=sv_straight['Y', 'end'],
                                Z0=sv_straight['Z', 'end'],
                                theta0=sv_straight['theta', 'end'],
                                phi0=sv_straight['phi', 'end'],
                                psi0=sv_straight['psi', 'end'])

sv_no_slice_start = line_no_slice.survey(element0='start',
                                X0=sv_straight['X', 'start'],
                                Y0=sv_straight['Y', 'start'],
                                Z0=sv_straight['Z', 'start'],
                                theta0=sv_straight['theta', 'start'],
                                phi0=sv_straight['phi', 'start'],
                                psi0=sv_straight['psi', 'start'])
sv_no_slice_end = line_no_slice.survey(element0='end',
                                X0=sv_straight['X', 'end'],
                                Y0=sv_straight['Y', 'end'],
                                Z0=sv_straight['Z', 'end'],
                                theta0=sv_straight['theta', 'end'],
                                phi0=sv_straight['phi', 'end'],
                                psi0=sv_straight['psi', 'end'])
tw_no_slice_straight = line_no_slice.twiss(betx=1, bety=1)

line['mb'].rbend_model = 'curved-body'
sv_curved = line.survey(element0='mid')
tt_curved = line.get_table(attr=True)
tw_curved = line.twiss(betx=1, bety=1)
p_curved = (sv_curved.p0 + tw_curved.x[:, None] * sv_curved['ex']
                         + tw_curved.y[:, None] * sv_curved['ey'])
tw_curved['X'] = p_curved[:, 0]
tw_curved['Y'] = p_curved[:, 1]
tw_curved['Z'] = p_curved[:, 2]

sv_curved_start = line.survey(element0='start',
                                X0=sv_curved['X', 'start'],
                                Y0=sv_curved['Y', 'start'],
                                Z0=sv_curved['Z', 'start'],
                                theta0=sv_curved['theta', 'start'],
                                phi0=sv_curved['phi', 'start'],
                                psi0=sv_curved['psi', 'start'])
sv_curved_end = line.survey(element0='end',
                                X0=sv_curved['X', 'end'],
                                Y0=sv_curved['Y', 'end'],
                                Z0=sv_curved['Z', 'end'],
                                theta0=sv_curved['theta', 'end'],
                                phi0=sv_curved['phi', 'end'],
                                psi0=sv_curved['psi', 'end'])
sv_no_slice_curved_start = line_no_slice.survey(element0='start',
                                X0=sv_curved['X', 'start'],
                                Y0=sv_curved['Y', 'start'],
                                Z0=sv_curved['Z', 'start'],
                                theta0=sv_curved['theta', 'start'],
                                phi0=sv_curved['phi', 'start'],
                                psi0=sv_curved['psi', 'start'])
sv_no_slice_curved_end = line_no_slice.survey(element0='end',
                                X0=sv_curved['X', 'end'],
                                Y0=sv_curved['Y', 'end'],
                                Z0=sv_curved['Z', 'end'],
                                theta0=sv_curved['theta', 'end'],
                                phi0=sv_curved['phi', 'end'],
                                psi0=sv_curved['psi', 'end'])
tw_no_slice_curved = line_no_slice.twiss(betx=1, bety=1)


for nn in ['start', 'end']:
    xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=2e-7)
    xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=2e-7)
    xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=1e-14)

xo.assert_allclose(sv_straight_start['X'], sv_straight['X'], atol=1e-14)
xo.assert_allclose(sv_straight_start['Y'], sv_straight['Y'], atol=2e-7)
xo.assert_allclose(sv_straight_start['Z'], sv_straight['Z'], atol=1e-14)
xo.assert_allclose(sv_straight_start['theta'], sv_straight['theta'], atol=1e-14)
xo.assert_allclose(sv_straight_start['phi'], sv_straight['phi'], atol=1e-14)
xo.assert_allclose(sv_straight_start['psi'], sv_straight['psi'], atol=1e-14)

xo.assert_allclose(sv_straight_end['X'], sv_straight['X'], atol=1e-14)
xo.assert_allclose(sv_straight_end['Y'], sv_straight['Y'], atol=1e-14)
xo.assert_allclose(sv_straight_end['Z'], sv_straight['Z'], atol=1e-14)
xo.assert_allclose(sv_straight_end['theta'], sv_straight['theta'], atol=1e-14)
xo.assert_allclose(sv_straight_end['phi'], sv_straight['phi'], atol=1e-14)
xo.assert_allclose(sv_straight_end['psi'], sv_straight['psi'], atol=1e-14)

xo.assert_allclose(sv_curved_start['X'], sv_curved['X'], atol=1e-14)
xo.assert_allclose(sv_curved_start['Y'], sv_curved['Y'], atol=2e-7)
xo.assert_allclose(sv_curved_start['Z'], sv_curved['Z'], atol=2e-7)
xo.assert_allclose(sv_curved_start['theta'], sv_curved['theta'], atol=1e-14)
xo.assert_allclose(sv_curved_start['phi'], sv_curved['phi'], atol=1e-14)
xo.assert_allclose(sv_curved_start['psi'], sv_curved['psi'], atol=1e-14)

xo.assert_allclose(sv_curved_end['X'], sv_curved['X'], atol=1e-14)
xo.assert_allclose(sv_curved_end['Y'], sv_curved['Y'], atol=1e-12)
xo.assert_allclose(sv_curved_end['Z'], sv_curved['Z'], atol=1e-12)
xo.assert_allclose(sv_curved_end['theta'], sv_curved['theta'], atol=1e-14)
xo.assert_allclose(sv_curved_end['phi'], sv_curved['phi'], atol=1e-12)
xo.assert_allclose(sv_curved_end['psi'], sv_curved['psi'], atol=1e-12)
xo.assert_allclose(tw_straight['X', 'mid'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'mid'], 0, atol=1e-3) # I am truncating the hamiltonian
xo.assert_allclose(tw_straight['Z', 'mid'], 0, atol=1e-12)
xo.assert_allclose(tw_curved['X', 'mid'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['Y', 'mid'], 0,  atol=1e-12)
xo.assert_allclose(tw_curved['Z', 'mid'], 0, atol=1e-12)
xo.assert_allclose(tw_straight['X', 'mb_entry'], tw_curved['X', 'mb_entry'], atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'mb_entry'], tw_curved['Y', 'mb_entry'], atol=2e-7)
xo.assert_allclose(tw_straight['Z', 'mb_entry'], tw_curved['Z', 'mb_entry'], atol=2e-7)
xo.assert_allclose(tw_straight['X', 'mb_exit'], tw_curved['X', 'mb_exit'], atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'mb_exit'], tw_curved['Y', 'mb_exit'], atol=2e-7)
xo.assert_allclose(tw_straight['Z', 'mb_exit'], tw_curved['Z', 'mb_exit'], atol=2e-7)

xo.assert_allclose(tw_straight['x', 'mb_entry'], 0, atol=1e-13)
xo.assert_allclose(tw_straight['y', 'mb_entry'], 0, atol=1e-13)
xo.assert_allclose(tw_straight['x', 'mb_exit'], 0, atol=1e-13)
xo.assert_allclose(tw_straight['y', 'mb_exit'], 0, atol=1e-13)
xo.assert_allclose(tw_curved['x', 'mb_entry'], 0, atol=1e-13)
xo.assert_allclose(tw_curved['y', 'mb_entry'], 0, atol=1e-13)
xo.assert_allclose(tw_curved['x', 'mb_exit'], 0, atol=1e-13)
xo.assert_allclose(tw_curved['y', 'mb_exit'], 0, atol=1e-13)

xo.assert_allclose(tw_no_slice_curved['x', 'start'], tw_curved['x', 'start'],
                   atol=1e-14)
xo.assert_allclose(tw_no_slice_curved['y', 'start'], tw_curved['y', 'start'],
                   atol=1e-14)
xo.assert_allclose(tw_no_slice_curved['x', 'end'], tw_curved['x', 'end'],
                   atol=1e-14)
xo.assert_allclose(tw_no_slice_curved['y', 'end'], tw_curved['y', 'end'],
                   atol=1e-14)
xo.assert_allclose(tw_no_slice_straight['x', 'start'], tw_curved['x', 'start'],
                     atol=1e-14)
xo.assert_allclose(tw_no_slice_straight['y', 'start'], tw_curved['y', 'start'],
                     atol=1e-14)
xo.assert_allclose(tw_no_slice_straight['x', 'end'], tw_curved['x', 'end'],
                     atol=1e-14)
xo.assert_allclose(tw_no_slice_straight['y', 'end'], tw_curved['y', 'end'],
                     atol=1e-14)

for nn in ['start', 'end']:
    # Compare no_slice survey vs curved survey
    xo.assert_allclose(sv_no_slice_curved_start['X', nn], sv_curved['X', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_start['Y', nn], sv_curved['Y', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_start['Z', nn], sv_curved['Z', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_start['theta', nn], sv_curved['theta', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_start['phi', nn], sv_curved['phi', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_start['psi', nn], sv_curved['psi', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_start['s', nn], sv_curved['s', nn], atol=5e-11)

    xo.assert_allclose(sv_no_slice_curved_end['X', nn], sv_curved['X', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_end['Y', nn], sv_curved['Y', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_end['Z', nn], sv_curved['Z', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_end['theta', nn], sv_curved['theta', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_end['phi', nn], sv_curved['phi', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_end['psi', nn], sv_curved['psi', nn], atol=5e-11)
    xo.assert_allclose(sv_no_slice_curved_end['s', nn], sv_curved['s', nn], atol=5e-11)

import matplotlib.pyplot as plt
plt.close('all')
sv_straight.plot(projection='ZY')
plt.plot(sv_curved.Z, sv_curved.Y, '.-', color='r', alpha=0.7)
plt.plot(tw_straight.Z, tw_straight.Y, 'x-', color='g', alpha=0.7)
plt.suptitle('Straight body')
plt.axis('auto')

sv_curved.plot(projection='ZY')
plt.suptitle('Curved body')

plt.show()