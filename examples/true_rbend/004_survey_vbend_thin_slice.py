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
            rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
            at=2.5)])
line.insert('start', xt.Marker(), at=0)
line.append('end', xt.Marker())

line_no_slice = line.copy(shallow=True)

line.slice_thick_elements(
        slicing_strategies=[
            # Slicing with thin elements
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Teapot(10), element_type=xt.RBend),
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

assert np.all(sv_straight['name'] == [
       'start', 'drift_1', 'mb_entry', 'mb..entry_map', 'drift_mb..0',
       'mb..0', 'drift_mb..1', 'mb..1', 'drift_mb..2', 'mb..2',
       'drift_mb..3', 'mb..3', 'drift_mb..4', 'mb..4', 'drift_mb..5..0',
       'mid', 'drift_mb..5..1', 'mb..5', 'drift_mb..6', 'mb..6',
       'drift_mb..7', 'mb..7', 'drift_mb..8', 'mb..8', 'drift_mb..9',
       'mb..9', 'drift_mb..10', 'mb..exit_map', 'mb_exit', 'drift_2',
       'end', '_end_point'])

# Assert entire columns using np.all
assert np.all(sv_straight['element_type'] == [
       'Marker', 'Drift', 'Marker', 'ThinSliceRBendEntry',
       'DriftSliceRBend', 'ThinSliceRBend', 'DriftSliceRBend',
       'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
       'DriftSliceRBend', 'ThinSliceRBend', 'DriftSliceRBend',
       'ThinSliceRBend', 'DriftSliceRBend', 'Marker', 'DriftSliceRBend',
       'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
       'DriftSliceRBend', 'ThinSliceRBend', 'DriftSliceRBend',
       'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
       'DriftSliceRBend', 'ThinSliceRBendExit', 'Marker', 'Drift',
       'Marker', ''])

xo.assert_allclose(
    sv_straight['angle'], np.array([
       0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
       0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
       0.  , 0.  , 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  ]),
    atol=1e-12
)

xo.assert_allclose(sv_straight['s'], np.array([
       0.        , 0.        , 0.9943602 , 0.9943602 , 0.9943602 ,
       1.13123654, 1.13123654, 1.4354062 , 1.4354062 , 1.73957586,
       1.73957586, 2.04374551, 2.04374551, 2.34791517, 2.34791517,
       2.5       , 2.5       , 2.65208483, 2.65208483, 2.95625449,
       2.95625449, 3.26042414, 3.26042414, 3.5645938 , 3.5645938 ,
       3.86876346, 3.86876346, 4.0056398 , 4.0056398 , 4.0056398 ,
       5.        , 5.        ]
), atol=1e-5)

xo.assert_allclose(
    sv_straight['rot_s_rad'],
    np.array([0.        , 0.        , 0.        , 1.57079633, 0.        ,
       1.57079633, 0.        , 1.57079633, 0.        , 1.57079633,
       0.        , 1.57079633, 0.        , 1.57079633, 0.        ,
       0.        , 0.        , 1.57079633, 0.        , 1.57079633,
       0.        , 1.57079633, 0.        , 1.57079633, 0.        ,
       1.57079633, 0.        , 1.57079633, 0.        , 0.        ,
       0.        , 0.        ]),
    atol=1e-8)


xo.assert_allclose(sv_straight['X'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['Z'], np.array([
       -2.48319461, -2.48319461, -1.5       , -1.5       , -1.5       ,
       -1.36363636, -1.36363636, -1.06060606, -1.06060606, -0.75757576,
       -0.75757576, -0.45454545, -0.45454545, -0.15151515, -0.15151515,
        0.        ,  0.        ,  0.15151515,  0.15151515,  0.45454545,
        0.45454545,  0.75757576,  0.75757576,  1.06060606,  1.06060606,
        1.36363636,  1.36363636,  1.5       ,  1.5       ,  1.5       ,
        2.48319461,  2.48319461]),
        atol=1e-8)
xo.assert_allclose(sv_straight['Y'], np.array(
      [-0.26130674, -0.26130674, -0.11271141, -0.11271141, -0.05635571,
       -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
       -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
       -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
       -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
       -0.05635571, -0.05635571, -0.05635571, -0.11271141, -0.11271141,
       -0.26130674, -0.26130674]),
       atol=1e-8)


xo.assert_allclose(sv_straight['theta'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['psi'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['phi'], np.array([
        0.15,  0.15,  0.15,  0.15,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  , -0.15, -0.15, -0.15, -0.15]))

assert np.all(sv_curved['name'] == sv_straight['name'])
assert np.all(sv_curved['element_type'] == sv_straight['element_type'])

xo.assert_allclose(
    sv_curved['angle'],
    np.array([
       0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.  , 0.03, 0.  , 0.03, 0.  ,
       0.03, 0.  , 0.03, 0.  , 0.  , 0.  , 0.03, 0.  , 0.03, 0.  , 0.03,
       0.  , 0.03, 0.  , 0.03, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]),
    atol=1e-8
)

xo.assert_allclose(sv_curved['s'], sv_straight['s'], atol=1e-12)
xo.assert_allclose(sv_curved['rot_s_rad'], sv_straight['rot_s_rad'], atol=1e-12)


xo.assert_allclose(sv_curved['X'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['Z'], np.array([
       -2.48319478, -2.48319478, -1.50000017, -1.50000017, -1.50000017,
       -1.3646608 , -1.3646608 , -1.06267854, -1.06267854, -0.75973993,
       -0.75973993, -0.45611762, -0.45611762, -0.15208483, -0.15208483,
        0.        ,  0.        ,  0.15208483,  0.15208483,  0.45611762,
        0.45611762,  0.75973993,  0.75973993,  1.06267854,  1.06267854,
        1.3646608 ,  1.3646608 ,  1.50000017,  1.50000017,  1.50000017,
        2.48319478,  2.48319478]), atol=1e-8)
xo.assert_allclose(sv_curved['Y'], np.array(
      [-0.26016398, -0.26016398, -0.11156865, -0.11156865, -0.11156865,
       -0.0911141 , -0.0911141 , -0.05470128, -0.05470128, -0.02736295,
       -0.02736295, -0.00912372, -0.00912372,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        , -0.00912372,
       -0.00912372, -0.02736295, -0.02736295, -0.05470128, -0.05470128,
       -0.0911141 , -0.0911141 , -0.11156865, -0.11156865, -0.11156865,
       -0.26016398, -0.26016398]), atol=1e-8)

xo.assert_allclose(sv_curved['theta'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['psi'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['phi'], np.array([
        0.15,  0.15,  0.15,  0.15,  0.15,  0.15,  0.12,  0.12,  0.09,
        0.09,  0.06,  0.06,  0.03,  0.03,  0.  ,  0.  ,  0.  ,  0.  ,
       -0.03, -0.03, -0.06, -0.06, -0.09, -0.09, -0.12, -0.12, -0.15,
       -0.15, -0.15, -0.15, -0.15, -0.15],
    ), atol=1e-8)

for nn in ['start', 'end']:
    xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], rtol=5e-3) # curve to polygon approximation
    xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=1e-6) # curve to polygon approximation
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
xo.assert_allclose(tw_straight['X', 'mid'], 0, atol=1e-14)
xo.assert_allclose(tw_straight['Y', 'mid'], 0, atol=2e-3) # curve to polygon approximation
xo.assert_allclose(tw_straight['Z', 'mid'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['X', 'mid'], 0, atol=1e-14)
xo.assert_allclose(tw_curved['Y', 'mid'], 0, atol=2e-3) # curve to polygon approximation
xo.assert_allclose(tw_curved['Z', 'mid'], 0, atol=1e-14)
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
sv_straight.plot(projection='ZY')
plt.plot(sv_curved.Z, sv_curved.Y, '.-', color='r', alpha=0.7)
plt.suptitle('Straight body')

sv_curved.plot(projection='ZY')
plt.suptitle('Curved body')

plt.show()