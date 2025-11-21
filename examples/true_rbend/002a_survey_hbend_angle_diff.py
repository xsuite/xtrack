import xtrack as xt
import numpy as np
import xobjects as xo

env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

edge_model = 'full'

line = env.new_line(length=5, components=[
    env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
            rbend_angle_diff=0.3,
            model='bend-kick-bend',
            rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
            anchor='start', at=1.)])
line['mb'].compensate_sagitta = False
line['mb'].rbend_shift = line['mb']._x0_in
line.insert('start', xt.Marker(), at=0)
line.append('end', xt.Marker())

line_no_slice = line.copy(shallow=True)

line.slice_thick_elements(
       slicing_strategies=[
              xt.Strategy(slicing=xt.Uniform(6, mode='thick'))])

line['mb'].rbend_model = 'straight-body'
sv_straight = line.survey()
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
sv_curved = line.survey()
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


assert np.all(sv_straight['name'] == [
       'start', '||drift_1', 'mb_entry', 'mb..entry_map', 'mb..0',
       'mb..1', 'mb..2', 'mb..3', 'mb..4', 'mb..5', 'mb..exit_map',
       'mb_exit', '||drift_2', 'end', '_end_point'])

# Assert entire columns using np.all
assert np.all(sv_straight['element_type'] == [
    'Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
       'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThinSliceRBendExit', 'Marker', 'Drift', 'Marker', ''])

xo.assert_allclose(
    sv_straight['angle'],
    np.array([
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.3, 0. , 0. ,
       0. , 0. ]),
    atol=1e-12
)

xo.assert_allclose(sv_straight['s'], np.array([
       0.        , 0.        , 1.        , 1.        , 1.        ,
       1.5075795 , 2.01515901, 2.52273851, 3.03031802, 3.53789752,
       4.04547703, 4.04547703, 4.04547703, 5.        , 5.        ]
), atol=1e-5)

xo.assert_allclose(
    sv_straight['rot_s_rad'], 0, atol=1e-14)

sv_straight.cols['X Y Z']
xo.assert_allclose(sv_straight['Y'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['Z'], np.array(
      [0.        , 0.        , 1.        , 1.        , 1.        ,
       1.5       , 2.        , 2.5       , 3.        , 3.5       ,
       4.        , 4.        , 4.        , 4.91189063, 4.91189063]),
       atol=1e-8)
xo.assert_allclose(sv_straight['X'], np.array([
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       -1.38777878e-17, -1.38777878e-17, -1.38777878e-17, -1.38777878e-17,
       -1.38777878e-17, -1.38777878e-17, -1.38777878e-17, -4.53405654e-01,
       -4.53405654e-01, -7.35486481e-01, -7.35486481e-01]),
       atol=1e-8)


sv_straight.cols['theta phi psi']
xo.assert_allclose(sv_straight['phi'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['psi'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['theta'], np.array([
          0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
       -0.3, -0.3, -0.3, -0.3]))


sv_curved.cols['s element_type angle']
assert np.all(sv_curved['name'] == [
       'start', '||drift_1', 'mb_entry', 'mb..entry_map', 'mb..0',
       'mb..1', 'mb..2', 'mb..3', 'mb..4', 'mb..5', 'mb..exit_map',
       'mb_exit', '||drift_2', 'end', '_end_point'])

assert np.all(sv_curved['element_type'] == [
       'Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
       'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThinSliceRBendExit', 'Marker', 'Drift', 'Marker', ''])

xo.assert_allclose(
    sv_curved['angle'],
    np.array([
       0.  , 0.  , 0.  , 0.  , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.  ,
       0.  , 0.  , 0.  , 0.  ]),
    atol=1e-8
)

xo.assert_allclose(sv_curved['s'], np.array([
       0.        , 0.        , 1.        , 1.        , 1.        ,
       1.5075795 , 2.01515901, 2.52273851, 3.03031802, 3.53789752,
       4.04547703, 4.04547703, 4.04547703, 5.        , 5.        ]
), atol=1e-5)

xo.assert_allclose(
    sv_curved['rot_s_rad'], 0, atol=1e-14)

sv_curved.cols['X Y Z']
xo.assert_allclose(sv_curved['Y'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['Z'], np.array([
       0.      , 0.      , 1.      , 1.      , 1.      , 1.507368,
       2.013468, 2.517035, 3.01681 , 3.511544, 4.      , 4.      ,
       4.      , 4.911891, 4.911891]), atol=1e-5)
xo.assert_allclose(sv_curved['X'], np.array(
      [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.01268684, -0.05071567, -0.11399141, -0.20235593, -0.31558835,
       -0.45340565, -0.45340565, -0.45340565, -0.73548648, -0.73548648]),
       atol=1e-8)

sv_curved.cols['theta phi psi']
xo.assert_allclose(sv_curved['phi'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['psi'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['theta'], np.array([
        0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.05, -0.1 , -0.15, -0.2 ,
       -0.25, -0.3 , -0.3 , -0.3 , -0.3 , -0.3 ],
    ), atol=1e-8)

for nn in ['start', 'end']:
    xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=1e-14)

xo.assert_allclose(sv_straight_start['X'], sv_straight['X'], atol=1e-14)
xo.assert_allclose(sv_straight_start['Y'], sv_straight['Y'], atol=1e-14)
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
xo.assert_allclose(sv_curved_start['Y'], sv_curved['Y'], atol=1e-14)
xo.assert_allclose(sv_curved_start['Z'], sv_curved['Z'], atol=1e-14)
xo.assert_allclose(sv_curved_start['theta'], sv_curved['theta'], atol=1e-14)
xo.assert_allclose(sv_curved_start['phi'], sv_curved['phi'], atol=1e-14)
xo.assert_allclose(sv_curved_start['psi'], sv_curved['psi'], atol=1e-14)

xo.assert_allclose(sv_curved_end['X'], sv_curved['X'], atol=1e-14)
xo.assert_allclose(sv_curved_end['Y'], sv_curved['Y'], atol=1e-14)
xo.assert_allclose(sv_curved_end['Z'], sv_curved['Z'], atol=1e-14)
xo.assert_allclose(sv_curved_end['theta'], sv_curved['theta'], atol=1e-14)
xo.assert_allclose(sv_curved_end['phi'], sv_curved['phi'], atol=1e-14)
xo.assert_allclose(sv_curved_end['psi'], sv_curved['psi'], atol=1e-14)
xo.assert_allclose(tw_straight['X', 'start'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'start'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['Z', 'start'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['X', 'start'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['Y', 'start'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['Z', 'start'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['X', 'mb_entry'], tw_curved['X', 'mb_entry'], atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'mb_entry'], tw_curved['Y', 'mb_entry'], atol=1e-14)
xo.assert_allclose(tw_straight['Z', 'mb_entry'], tw_curved['Z', 'mb_entry'], atol=1e-14)
xo.assert_allclose(tw_straight['X', 'mb_exit'], tw_curved['X', 'mb_exit'], atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'mb_exit'], tw_curved['Y', 'mb_exit'], atol=1e-14)
xo.assert_allclose(tw_straight['Z', 'mb_exit'], tw_curved['Z', 'mb_exit'], atol=1e-14)

xo.assert_allclose(tw_straight['x', 'mb_entry'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['y', 'mb_entry'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['x', 'mb_exit'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['y', 'mb_exit'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['x', 'mb_entry'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['y', 'mb_entry'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['x', 'mb_exit'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['y', 'mb_exit'], 0, atol=1e-14)

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
    xo.assert_allclose(sv_no_slice_curved_start['X', nn], sv_curved['X', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_start['Y', nn], sv_curved['Y', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_start['Z', nn], sv_curved['Z', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_start['theta', nn], sv_curved['theta', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_start['phi', nn], sv_curved['phi', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_start['psi', nn], sv_curved['psi', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_start['s', nn], sv_curved['s', nn], atol=1e-14)

    xo.assert_allclose(sv_no_slice_curved_end['X', nn], sv_curved['X', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_end['Y', nn], sv_curved['Y', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_end['Z', nn], sv_curved['Z', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_end['theta', nn], sv_curved['theta', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_end['phi', nn], sv_curved['phi', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_end['psi', nn], sv_curved['psi', nn], atol=1e-14)
    xo.assert_allclose(sv_no_slice_curved_end['s', nn], sv_curved['s', nn], atol=1e-14)

import matplotlib.pyplot as plt
plt.close('all')
sv_straight.plot(projection='ZX')
plt.plot(sv_curved.Z, sv_curved.X, '.-', color='r', alpha=0.7)
plt.suptitle('Straight body')

sv_curved.plot(projection='ZX')
plt.suptitle('Curved body')

plt.show()