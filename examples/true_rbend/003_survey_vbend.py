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
            rbend_model='straight-body', edge_entry_model=edge_model, edge_exit_model=edge_model,
            at=2.5)])

line.cut_at_s(np.linspace(0, line.get_length(), 11))
line.insert('mid', xt.Marker(), at=2.5)
line.insert('start', xt.Marker(), at=0)
line.append('end', xt.Marker())

line['mb'].rbend_model = 'straight-body'
sv_straight = line.survey(element0='mid', Y0=-line['mb'].sagitta/2)
tt_straight = line.get_table(attr=True)
tw_straight = line.twiss(betx=1, bety=1)

line['mb'].rbend_model = 'curved-body'
sv_curved = line.survey(element0='mid')
tt_curved = line.get_table(attr=True)
tw_curved = line.twiss(betx=1, bety=1)

sv_straight.cols['s element_type angle']
# is:
# Table: 20 rows, 4 cols
# name                      s element_type            angle
# start                     0 Marker                          0
# drift_1..0                0 DriftSlice                      0
# drift_1..1              0.5 DriftSlice                      0
# mb_entry            0.99436 Marker                          0
# mb..entry_map       0.99436 ThinSliceRBendEntry          0.15
# mb..0               0.99436 ThickSliceRBend                 0
# mb..1                     1 ThickSliceRBend                 0
# mb..2                   1.5 ThickSliceRBend                 0
# mb..3                     2 ThickSliceRBend                 0
# mid                     2.5 Marker                          0
# mb..4                   2.5 ThickSliceRBend                 0
# mb..5                     3 ThickSliceRBend                 0
# mb..6                   3.5 ThickSliceRBend                 0
# mb..7                     4 ThickSliceRBend                 0
# mb..exit_map        4.00564 ThinSliceRBendExit           0.15
# mb_exit             4.00564 Marker                          0
# drift_2..0          4.00564 DriftSlice                      0
# drift_2..1              4.5 DriftSlice                      0
# end                       5 Marker                          0
# _end_point                5                                 0

assert np.all(sv_straight['name'] == [
       'start', 'drift_1..0', 'drift_1..1', 'mb_entry', 'mb..entry_map',
       'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
       'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', 'drift_2..0',
       'drift_2..1', 'end', '_end_point'
])

# Assert entire columns using np.all
assert np.all(sv_straight['element_type'] == ['Marker', 'DriftSlice', 'DriftSlice', 'Marker',
       'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
       'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThinSliceRBendExit', 'Marker', 'DriftSlice', 'DriftSlice',
       'Marker', ''])

xo.assert_allclose(
    sv_straight['angle'],
    np.array([
        0.  , 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  ]),
    atol=1e-12
)

xo.assert_allclose(sv_straight['s'], np.array([
       0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
       1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
       3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
       5.       , 5.       ]
), atol=1e-5)

xo.assert_allclose(
    sv_straight['rot_s_rad'],
    np.array([
        0.        , 0.        , 0.        , 0.        , 1.57079633,
        1.57079633, 1.57079633, 1.57079633, 1.57079633, 0.        ,
        1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
        0.        , 0.        , 0.        , 0.        , 0.        ]),
    atol=1e-8
)

sv_straight.cols['X Y Z']
# is:
# SurveyTable: 20 rows, 4 cols
# name                      X             Y             Z
# start          -1.25496e-17     -0.261307      -2.48319
# drift_1..0     -1.25496e-17     -0.261307      -2.48319
# drift_1..1     -7.97441e-18     -0.186588      -1.98881
# mb_entry       -3.45079e-18     -0.112711          -1.5
# mb..entry_map  -3.45079e-18     -0.112711          -1.5
# mb..0                     0    -0.0563557          -1.5
# mb..1                     0    -0.0563557      -1.49438
# mb..2                     0    -0.0563557     -0.996254
# mb..3                     0    -0.0563557     -0.498127
# mid                       0    -0.0563557             0
# mb..4                     0    -0.0563557             0
# mb..5                     0    -0.0563557      0.498127
# mb..6                     0    -0.0563557      0.996254
# mb..7                     0    -0.0563557       1.49438
# mb..exit_map              0    -0.0563557           1.5
# mb_exit        -3.45079e-18     -0.112711           1.5
# drift_2..0     -3.45079e-18     -0.112711           1.5
# drift_2..1     -7.97441e-18     -0.186588       1.98881
# end            -1.25496e-17     -0.261307       2.48319

xo.assert_allclose(sv_straight['X'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['Z'], np.array([
       -2.48319461, -2.48319461, -1.98880907, -1.5       , -1.5       ,
       -1.5       , -1.49438132, -0.99625422, -0.49812711,  0.        ,
        0.        ,  0.49812711,  0.99625422,  1.49438132,  1.5       ,
        1.5       ,  1.5       ,  1.98880907,  2.48319461,  2.48319461]),
        atol=1e-8)
xo.assert_allclose(sv_straight['Y'], np.array([
       -0.26130674, -0.26130674, -0.18658768, -0.11271141, -0.11271141,
       -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
       -0.05635571, -0.05635571, -0.05635571, -0.05635571, -0.05635571,
       -0.11271141, -0.11271141, -0.18658768, -0.26130674, -0.26130674]),
       atol=1e-8)


sv_straight.cols['theta phi psi']
# is:
# SurveyTable: 20 rows, 4 cols
# name                  theta           phi           psi
# start           9.25436e-18          0.15  -6.95382e-19
# drift_1..0      9.25436e-18          0.15  -6.95382e-19
# drift_1..1      9.25436e-18          0.15  -6.95382e-19
# mb_entry        9.25436e-18          0.15  -6.95382e-19
# mb..entry_map   9.25436e-18          0.15  -6.95382e-19
# mb..0                     0             0             0
# mb..1                     0             0             0
# mb..2                     0             0             0
# mb..3                     0             0             0
# mid                       0             0             0
# mb..4                     0             0             0
# mb..5                     0             0             0
# mb..6                     0             0             0
# mb..7                     0             0             0
# mb..exit_map              0             0             0
# mb_exit        -9.25436e-18         -0.15  -6.95382e-19
# drift_2..0     -9.25436e-18         -0.15  -6.95382e-19
# drift_2..1     -9.25436e-18         -0.15  -6.95382e-19
# end            -9.25436e-18         -0.15  -6.95382e-19

xo.assert_allclose(sv_straight['theta'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['psi'], 0, atol=1e-14)
xo.assert_allclose(sv_straight['phi'], np.array([
        0.15,  0.15,  0.15,  0.15,  0.15,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.15, -0.15, -0.15,
       -0.15, -0.15]))


sv_curved.cols['s element_type angle']
# is:
# Table: 20 rows, 4 cols
# name                      s element_type            angle
# start                     0 Marker                          0
# drift_1..0                0 DriftSlice                      0
# drift_1..1              0.5 DriftSlice                      0
# mb_entry            0.99436 Marker                          0
# mb..entry_map       0.99436 ThinSliceRBendEntry             0
# mb..0               0.99436 ThickSliceRBend       0.000561868
# mb..1                     1 ThickSliceRBend         0.0498127
# mb..2                   1.5 ThickSliceRBend         0.0498127
# mb..3                     2 ThickSliceRBend         0.0498127
# mid                     2.5 Marker                          0
# mb..4                   2.5 ThickSliceRBend         0.0498127
# mb..5                     3 ThickSliceRBend         0.0498127
# mb..6                   3.5 ThickSliceRBend         0.0498127
# mb..7                     4 ThickSliceRBend       0.000561868
# mb..exit_map        4.00564 ThinSliceRBendExit              0
# mb_exit             4.00564 Marker                          0
# drift_2..0          4.00564 DriftSlice                      0
# drift_2..1              4.5 DriftSlice                      0
# end                       5 Marker                          0
# _end_point                5                                 0

assert np.all(sv_curved['name'] == [
    'start', 'drift_1..0', 'drift_1..1', 'mb_entry', 'mb..entry_map',
    'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
    'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', 'drift_2..0',
    'drift_2..1', 'end', '_end_point'
])

assert np.all(sv_curved['element_type'] == [
    'Marker', 'DriftSlice', 'DriftSlice', 'Marker',
    'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
    'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
    'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
    'ThinSliceRBendExit', 'Marker', 'DriftSlice', 'DriftSlice',
    'Marker', ''])

xo.assert_allclose(
    sv_curved['angle'],
    np.array([
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.00056187, 0.04981271, 0.04981271, 0.04981271, 0.        ,
       0.04981271, 0.04981271, 0.04981271, 0.00056187, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ]),
    atol=1e-8
)

xo.assert_allclose(sv_curved['s'], np.array([
       0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
       1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
       3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
       5.       , 5.       ]
), atol=1e-5)

xo.assert_allclose(
    sv_curved['rot_s_rad'],
    np.array([
        0.        , 0.        , 0.        , 0.        , 1.57079633,
        1.57079633, 1.57079633, 1.57079633, 1.57079633, 0.        ,
        1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
        0.        , 0.        , 0.        , 0.        , 0.        ]),
    atol=1e-8
)

sv_curved.cols['X Y Z']
# is:
# SurveyTable: 20 rows, 4 cols
# name                      X             Y             Z
# start          -1.60004e-17     -0.261307      -2.48319
# drift_1..0     -1.60004e-17     -0.261307      -2.48319
# drift_1..1     -1.14252e-17     -0.186588      -1.98881
# mb_entry       -6.90158e-18     -0.112711          -1.5
# mb..entry_map  -6.90158e-18     -0.112711          -1.5
# mb..0          -6.90158e-18     -0.112711          -1.5
# mb..1          -6.85007e-18      -0.11187      -1.49442
# mb..2          -3.04763e-18    -0.0497715     -0.998347
# mb..3           -7.6238e-19    -0.0124506     -0.499793
# mid                       0             0             0
# mb..4                     0             0             0
# mb..5           -7.6238e-19    -0.0124506      0.499793
# mb..6          -3.04763e-18    -0.0497715      0.998347
# mb..7          -6.85007e-18      -0.11187       1.49442
# mb..exit_map   -6.90158e-18     -0.112711           1.5
# mb_exit        -6.90158e-18     -0.112711           1.5
# drift_2..0     -6.90158e-18     -0.112711           1.5
# drift_2..1     -1.14252e-17     -0.186588       1.98881
# end            -1.60004e-17     -0.261307       2.48319
# _end_point     -1.60004e-17     -0.261307       2.48319

xo.assert_allclose(sv_curved['X'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['Z'], np.array([
    -2.48319461, -2.48319461, -1.98880907, -1.5       , -1.5       ,
    -1.5       , -1.49442329, -0.99834662, -0.49979325,  0.        ,
    0.        ,  0.49979325,  0.99834662,  1.49442329,  1.5       ,
    1.5       ,  1.5       ,  1.98880907,  2.48319461,  2.48319461
]), atol=1e-8)
xo.assert_allclose(sv_curved['Y'], np.array([
    -0.26130674, -0.26130674, -0.18658768, -0.11271141, -0.11271141,
    -0.11271141, -0.11187018, -0.04977152, -0.0124506 ,  0.        ,
     0.        , -0.0124506 , -0.04977152, -0.11187018, -0.11271141,
    -0.11271141, -0.11271141, -0.18658768, -0.26130674, -0.26130674
]), atol=1e-8)

sv_curved.cols['theta phi psi']
# is:
# SurveyTable: 20 rows, 4 cols
# name                  theta           phi           psi
# start           9.25436e-18          0.15  -6.95382e-19
# drift_1..0      9.25436e-18          0.15  -6.95382e-19
# drift_1..1      9.25436e-18          0.15  -6.95382e-19
# mb_entry        9.25436e-18          0.15  -6.95382e-19
# mb..entry_map   9.25436e-18          0.15  -6.95382e-19
# mb..0           9.25436e-18          0.15  -6.95382e-19
# mb..1           9.21918e-18      0.149438  -6.90133e-19
# mb..2           6.12056e-18     0.0996254  -3.05134e-19
# mb..3           3.05267e-18     0.0498127  -7.60467e-20
# mid                       0             0             0
# mb..4                     0             0             0
# mb..5          -3.05267e-18    -0.0498127  -7.60467e-20
# mb..6          -6.12056e-18    -0.0996254  -3.05134e-19
# mb..7          -9.21918e-18     -0.149438  -6.90133e-19
# mb..exit_map   -9.25436e-18         -0.15  -6.95382e-19
# mb_exit        -9.25436e-18         -0.15  -6.95382e-19
# drift_2..0     -9.25436e-18         -0.15  -6.95382e-19
# drift_2..1     -9.25436e-18         -0.15  -6.95382e-19
# end            -9.25436e-18         -0.15  -6.95382e-19
# _end_point     -9.25436e-18         -0.15  -6.95382e-19

xo.assert_allclose(sv_curved['theta'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['psi'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['phi'], np.array([
        0.15      ,  0.15      ,  0.15      ,  0.15      ,  0.15      ,
        0.15      ,  0.14943813,  0.09962542,  0.04981271,  0.        ,
        0.        , -0.04981271, -0.09962542, -0.14943813, -0.15      ,
       -0.15      , -0.15      , -0.15      , -0.15      , -0.15      ],
    ), atol=1e-8)

for nn in ['start', 'end']:
    xo.assert_allclose(sv_straight['X', nn], sv_curved['X', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['Y', nn], sv_curved['Y', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['Z', nn], sv_curved['Z', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['theta', nn], sv_curved['theta', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['phi', nn], sv_curved['phi', nn], atol=1e-14)
    xo.assert_allclose(sv_straight['psi', nn], sv_curved['psi', nn], atol=1e-14)

import matplotlib.pyplot as plt
plt.close('all')
sv_straight.plot(projection='ZY')
plt.plot(sv_curved.Z, sv_curved.Y, '.-', color='r', alpha=0.7)
plt.suptitle('Straight body')

sv_curved.plot(projection='ZY')
plt.suptitle('Curved body')

plt.show()