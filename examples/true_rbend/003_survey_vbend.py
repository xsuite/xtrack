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

tt_straight.cols['s element_type angle_rad']
# is:
# Table: 20 rows, 4 cols
# name                      s element_type            angle_rad
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

assert np.all(tt_straight['name'] == [
       'start', 'drift_1..0', 'drift_1..1', 'mb_entry', 'mb..entry_map',
       'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
       'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', 'drift_2..0',
       'drift_2..1', 'end', '_end_point'
])

# Assert entire columns using np.all
assert np.all(tt_straight['element_type'] == ['Marker', 'DriftSlice', 'DriftSlice', 'Marker',
       'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
       'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
       'ThinSliceRBendExit', 'Marker', 'DriftSlice', 'DriftSlice',
       'Marker', ''])

xo.assert_allclose(
    tt_straight['angle_rad'],
    np.array([
        0.  , 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  , 0.  , 0.  ]),
    atol=1e-12
)

xo.assert_allclose(tt_straight['s'], np.array([
       0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
       1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
       3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
       5.       , 5.       ]
), atol=1e-5)


tt_curved.cols['s element_type angle_rad']
# is:
# Table: 20 rows, 4 cols
# name                      s element_type            angle_rad
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

assert np.all(tt_curved['name'] == [
    'start', 'drift_1..0', 'drift_1..1', 'mb_entry', 'mb..entry_map',
    'mb..0', 'mb..1', 'mb..2', 'mb..3', 'mid', 'mb..4', 'mb..5',
    'mb..6', 'mb..7', 'mb..exit_map', 'mb_exit', 'drift_2..0',
    'drift_2..1', 'end', '_end_point'
])

assert np.all(tt_curved['element_type'] == [
    'Marker', 'DriftSlice', 'DriftSlice', 'Marker',
    'ThinSliceRBendEntry', 'ThickSliceRBend', 'ThickSliceRBend',
    'ThickSliceRBend', 'ThickSliceRBend', 'Marker', 'ThickSliceRBend',
    'ThickSliceRBend', 'ThickSliceRBend', 'ThickSliceRBend',
    'ThinSliceRBendExit', 'Marker', 'DriftSlice', 'DriftSlice',
    'Marker', ''])

xo.assert_allclose(
    tt_curved['angle_rad'],
    np.array([
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.00056187, 0.04981271, 0.04981271, 0.04981271, 0.        ,
       0.04981271, 0.04981271, 0.04981271, 0.00056187, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ]),
    atol=1e-8
)

xo.assert_allclose(tt_curved['s'], np.array([
       0.       , 0.       , 0.5      , 0.9943602, 0.9943602, 0.9943602,
       1.       , 1.5      , 2.       , 2.5      , 2.5      , 3.       ,
       3.5      , 4.       , 4.0056398, 4.0056398, 4.0056398, 4.5      ,
       5.       , 5.       ]
), atol=1e-5)



import matplotlib.pyplot as plt
plt.close('all')
sv_straight.plot(projection='ZY')
plt.plot(sv_curved.Z, sv_curved.Y, color='r', alpha=0.7)
plt.suptitle('Straight body')

sv_curved.plot(projection='ZY')
plt.suptitle('Curved body')

plt.show()