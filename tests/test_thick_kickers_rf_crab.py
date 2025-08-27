import pathlib

import numpy as np

import xobjects as xo
import xtrack as xt

from cpymad.madx import Madx

def test_thick_hkicker_cpymad_loader():

    mad_data = """

    hk: hkicker, l=1, kick=1e-3;

    ss: sequence, l=3;
    hk1: hk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)
    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 6 rows, 122 cols
    # name                   s element_type isthick isreplica ...
    # ss$start               0 Marker         False     False
    # drift_0                0 Drift           True     False
    # hk1                  0.5 Multipole       True     False
    # drift_1              1.5 Drift           True     False
    # ss$end                 3 Marker         False     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['ss$start', 'drift_0', 'hk1', 'drift_1', 'ss$end', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
    assert np.all(tt.element_type == np.array(
        ['Marker', 'Drift', 'Multipole', 'Drift', 'Marker', '']))
    assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))
    assert np.allclose(tt.hkick, np.array([0.   , 0.   , 0.001, 0.   , 0.   , 0.   ]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 10 rows, 122 cols
    # name                   s element_type        isthick isreplica ...
    # ss$start               0 Marker                False     False
    # drift_0                0 Drift                  True     False
    # hk1_entry            0.5 Marker                False     False
    # hk1..0               0.5 ThickSliceMultipole    True     False
    # hk1..1          0.833333 ThickSliceMultipole    True     False
    # hk1..2           1.16667 ThickSliceMultipole    True     False
    # hk1_exit             1.5 Marker                False     False
    # drift_1              1.5 Drift                  True     False
    # ss$end                 3 Marker                False     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'ss$start', 'drift_0', 'hk1_entry', 'hk1..0', 'hk1..1', 'hk1..2',
        'hk1_exit', 'drift_1', 'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        False, True, False, True, True, True, False, True, False, False
    ]))
    assert np.allclose(tt_slice_thick.hkick, np.array([
        0., 0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 14 rows, 122 cols
    # name                     s element_type        isthick isreplica ...
    # ss$start                 0 Marker                False     False
    # drift_0                  0 Drift                  True     False
    # hk1_entry              0.5 Marker                False     False
    # drift_hk1..0           0.5 DriftSliceMultipole    True     False
    # hk1..3                0.75 ThinSliceMultipole    False     False
    # drift_hk1..1          0.75 DriftSliceMultipole    True     False
    # hk1..4                   1 ThinSliceMultipole    False     False
    # drift_hk1..2             1 DriftSliceMultipole    True     False
    # hk1..5                1.25 ThinSliceMultipole    False     False
    # drift_hk1..3          1.25 DriftSliceMultipole    True     False
    # hk1_exit               1.5 Marker                False     False
    # drift_1                1.5 Drift                  True     False
    # ss$end                   3 Marker                False     False
    # _end_point               3                       False     False

    assert np.all(tt_slice_thin.name == np.array([
        'ss$start', 'drift_0', 'hk1_entry', 'drift_hk1..0', 'hk1..3', 'drift_hk1..1',
        'hk1..4', 'drift_hk1..2', 'hk1..5', 'drift_hk1..3', 'hk1_exit', 'drift_1',
        'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        False, True, False, True, False, True, False, True, False, True, False, True, False, False
    ]))
    assert np.allclose(tt_slice_thin.hkick, np.array([
        0., 0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)


def test_thick_hkicker_native_loader():

    mad_data = """

    hk: hkicker, l=1, kick=1e-3;

    ss: sequence, l=3;
    hk1: hk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    env = xt.load(string=mad_data, format='madx')
    line = env.ss
    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 4 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # drift_1                0 Drift           True     False
    # hk1                  0.5 Multipole       True     False
    # drift_2              1.5 Drift           True     False
    # _end_point             3                False     False


    assert np.all(tt.name == np.array([
        'drift_1', 'hk1', 'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt.s, np.array([0, 0.5, 1.5, 3]))
    assert np.all(tt.element_type == np.array([
        'Drift', 'Multipole', 'Drift', ''
    ]))
    assert np.all(tt.isthick == np.array([True, True, True, False]))
    assert np.allclose(tt.hkick, np.array([0., 0.001, 0., 0.]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 8 rows, 130 cols
    # name                   s element_type        isthick isreplica ...
    # drift_1                0 Drift                  True     False
    # hk1_entry            0.5 Marker                False     False
    # hk1..0               0.5 ThickSliceMultipole    True     False
    # hk1..1          0.833333 ThickSliceMultipole    True     False
    # hk1..2           1.16667 ThickSliceMultipole    True     False
    # hk1_exit             1.5 Marker                False     False
    # drift_2              1.5 Drift                  True     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'drift_1', 'hk1_entry', 'hk1..0', 'hk1..1', 'hk1..2', 'hk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        True, False, True, True, True, False, True, False
    ]))
    assert np.allclose(tt_slice_thick.hkick, np.array([
        0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 12 rows, 130 cols
    # name                     s element_type        isthick isreplica ...
    # drift_1                  0 Drift                  True     False
    # hk1_entry              0.5 Marker                False     False
    # drift_hk1..0           0.5 DriftSliceMultipole    True     False
    # hk1..3                0.75 ThinSliceMultipole    False     False
    # drift_hk1..1          0.75 DriftSliceMultipole    True     False
    # hk1..4                   1 ThinSliceMultipole    False     False
    # drift_hk1..2             1 DriftSliceMultipole    True     False
    # hk1..5                1.25 ThinSliceMultipole    False     False
    # drift_hk1..3          1.25 DriftSliceMultipole    True     False
    # hk1_exit               1.5 Marker                False     False
    # drift_2                1.5 Drift                  True     False
    # _end_point               3                       False     False

    assert np.all(tt_slice_thin.name == np.array([
        'drift_1', 'hk1_entry', 'drift_hk1..0', 'hk1..3', 'drift_hk1..1',
        'hk1..4', 'drift_hk1..2', 'hk1..5', 'drift_hk1..3', 'hk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        True, False, True, False, True, False, True, False, True, False, True, False
    ]))
    assert np.allclose(tt_slice_thin.hkick, np.array([
        0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)


    line_opt = line_slice_thin.copy(shallow=True)
    line_opt.build_tracker()
    line_opt.optimize_for_tracking()
    tw_opt = line_opt.twiss(betx=1, bety=1)
    tt_opt = line_opt.get_table(attr=True)
    # is
    # Table: 8 rows, 130 cols
    # name                     s element_type   isthick isreplica parent_name ...
    # drift_1                  0 Drift             True     False None
    # hk1..3                0.75 SimpleThinBend   False     False None
    # drift_hk1..1          0.75 Drift             True     False None
    # hk1..4                   1 SimpleThinBend   False     False None
    # drift_hk1..2             1 Drift             True     False None
    # hk1..5                1.25 SimpleThinBend   False     False None
    # drift_hk1..3          1.25 Drift             True     False None
    # _end_point               3                  False     False None

    assert np.all(tt_opt.name == np.array([
        'drift_1', 'hk1..3', 'drift_hk1..1', 'hk1..4', 'drift_hk1..2',
        'hk1..5', 'drift_hk1..3', '_end_point'
    ]))
    xo.assert_allclose(tt_opt.s, np.array([
        0, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_opt.element_type == np.array([
        'Drift', 'SimpleThinBend', 'Drift', 'SimpleThinBend', 'Drift',
        'SimpleThinBend', 'Drift', ''
    ]))
    assert np.all(tt_opt.isthick == np.array([
        True, False, True, False, True, False, True, False
    ]))
    assert np.all(tt_opt.isreplica == np.array([
        False, False, False, False, False, False, False, False
    ]))
    assert np.all(tt_opt.parent_name == np.array([
        None, None, None, None, None, None, None, None
    ]))
    assert np.allclose(tt_opt.hkick, np.array([
        0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0.
    ]))

    xo.assert_allclose(tw_opt.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

def test_thick_vkicker_cpymad_loader():

    mad_data = """

    vk: vkicker, l=1, kick=1e-3;

    ss: sequence, l=3;
    vk1: vk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)

    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 6 rows, 122 cols
    # name                   s element_type isthick isreplica ...
    # ss$start               0 Marker         False     False
    # drift_0                0 Drift           True     False
    # vk1                  0.5 Multipole       True     False
    # drift_1              1.5 Drift           True     False
    # ss$end                 3 Marker         False     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['ss$start', 'drift_0', 'vk1', 'drift_1', 'ss$end', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
    assert np.all(tt.element_type == np.array(
        ['Marker', 'Drift', 'Multipole', 'Drift', 'Marker', '']))
    assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))
    assert np.allclose(tt.vkick, np.array([0.   , 0.   , 0.001, 0.   , 0.   , 0.   ]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 10 rows, 122 cols
    # name                   s element_type        isthick isreplica ...
    # ss$start               0 Marker                False     False
    # drift_0                0 Drift                  True     False
    # vk1_entry            0.5 Marker                False     False
    # vk1..0               0.5 ThickSliceMultipole    True     False
    # vk1..1          0.833333 ThickSliceMultipole    True     False
    # vk1..2           1.16667 ThickSliceMultipole    True     False
    # vk1_exit             1.5 Marker                False     False
    # drift_1              1.5 Drift                  True     False
    # ss$end                 3 Marker                False     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'ss$start', 'drift_0', 'vk1_entry', 'vk1..0', 'vk1..1', 'vk1..2',
        'vk1_exit', 'drift_1', 'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        False, True, False, True, True, True, False, True, False, False
    ]))
    xo.assert_allclose(tt_slice_thick.vkick, np.array([
        0., 0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 14 rows, 122 cols
    # name                     s element_type        isthick isreplica ...
    # ss$start                 0 Marker                False     False
    # drift_0                  0 Drift                  True     False
    # vk1_entry              0.5 Marker                False     False
    # drift_vk1..0           0.5 DriftSliceMultipole    True     False
    # vk1..3                0.75 ThinSliceMultipole    False     False
    # drift_vk1..1          0.75 DriftSliceMultipole    True     False
    # vk1..4                   1 ThinSliceMultipole    False     False
    # drift_vk1..2             1 DriftSliceMultipole    True     False
    # vk1..5                1.25 ThinSliceMultipole    False     False
    # drift_vk1..3          1.25 DriftSliceMultipole    True     False
    # vk1_exit               1.5 Marker                False     False
    # drift_1                1.5 Drift                  True     False
    # ss$end                   3 Marker                False     False
    # _end_point               3                       False     False

    assert np.all(tt_slice_thin.name == np.array([
        'ss$start', 'drift_0', 'vk1_entry', 'drift_vk1..0', 'vk1..3', 'drift_vk1..1',
        'vk1..4', 'drift_vk1..2', 'vk1..5', 'drift_vk1..3', 'vk1_exit', 'drift_1',
        'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        False, True, False, True, False, True, False, True, False, True, False, True, False, False
    ]))
    xo.assert_allclose(tt_slice_thin.vkick, np.array([
        0., 0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

def test_thick_vkicker_native_loader():

    mad_data = """

    vk: vkicker, l=1, kick=1e-3;

    ss: sequence, l=3;
    vk1: vk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    env = xt.load(string=mad_data, format='madx')
    line = env.ss

    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 4 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # drift_1                0 Drift           True     False
    # vk1                  0.5 Multipole       True     False
    # drift_2              1.5 Drift           True     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['drift_1', 'vk1', 'drift_2', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0.5, 1.5, 3]))
    assert np.all(tt.element_type == np.array(
        ['Drift', 'Multipole', 'Drift', '']))
    assert np.all(tt.isthick == np.array([True, True, True, False]))
    assert np.allclose(tt.vkick, np.array([0., 0.001, 0., 0.]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 8 rows, 130 cols
    # name                   s element_type        isthick isreplica ...
    # drift_1                0 Drift                  True     False
    # vk1_entry            0.5 Marker                False     False
    # vk1..0               0.5 ThickSliceMultipole    True     False
    # vk1..1          0.833333 ThickSliceMultipole    True     False
    # vk1..2           1.16667 ThickSliceMultipole    True     False
    # vk1_exit             1.5 Marker                False     False
    # drift_2              1.5 Drift                  True     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'drift_1', 'vk1_entry', 'vk1..0', 'vk1..1', 'vk1..2', 'vk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        True, False, True, True, True, False, True, False
    ]))
    xo.assert_allclose(tt_slice_thick.vkick, np.array([
        0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 12 rows, 130 cols
    # name                     s element_type        isthick isreplica ...
    # drift_1                  0 Drift                  True     False
    # vk1_entry              0.5 Marker                False     False
    # drift_vk1..0           0.5 DriftSliceMultipole    True     False
    # vk1..3                0.75 ThinSliceMultipole    False     False
    # drift_vk1..1          0.75 DriftSliceMultipole    True     False
    # vk1..4                   1 ThinSliceMultipole    False     False
    # drift_vk1..2             1 DriftSliceMultipole    True     False
    # vk1..5                1.25 ThinSliceMultipole    False     False
    # drift_vk1..3          1.25 DriftSliceMultipole    True     False
    # vk1_exit               1.5 Marker                False     False
    # drift_2                1.5 Drift                  True     False
    # _end_point               3                       False     False
    assert np.all(tt_slice_thin.name == np.array([
        'drift_1', 'vk1_entry', 'drift_vk1..0', 'vk1..3', 'drift_vk1..1',
        'vk1..4', 'drift_vk1..2', 'vk1..5', 'drift_vk1..3', 'vk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        True, False, True, False, True, False, True, False, True, False, True, False
    ]))
    xo.assert_allclose(tt_slice_thin.vkick, np.array([
        0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_opt = line_slice_thin.copy(shallow=True)
    line_opt.build_tracker()
    line_opt.optimize_for_tracking()
    tw_opt = line_opt.twiss(betx=1, bety=1)
    tt_opt = line_opt.get_table(attr=True)
    # is
    # Table: 8 rows, 130 cols
    # name                     s element_type isthick isreplica parent_name ...
    # drift_1                  0 Drift           True     False None
    # vk1..3                0.75 Multipole      False     False None
    # drift_vk1..1          0.75 Drift           True     False None
    # vk1..4                   1 Multipole      False     False None
    # drift_vk1..2             1 Drift           True     False None
    # vk1..5                1.25 Multipole      False     False None
    # drift_vk1..3          1.25 Drift           True     False None
    # _end_point               3                False     False None

    assert np.all(tt_opt.name == np.array([
        'drift_1', 'vk1..3', 'drift_vk1..1', 'vk1..4', 'drift_vk1..2',
        'vk1..5', 'drift_vk1..3', '_end_point'
    ]))
    xo.assert_allclose(tt_opt.s, np.array([
        0, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_opt.element_type == np.array([
        'Drift', 'Multipole', 'Drift', 'Multipole', 'Drift',
        'Multipole', 'Drift', ''
    ]))
    assert np.all(tt_opt.isthick == np.array([
        True, False, True, False, True, False, True, False
    ]))
    assert np.all(tt_opt.isreplica == np.array([
        False, False, False, False, False, False, False, False
    ]))
    assert np.all(tt_opt.parent_name == np.array([
        None, None, None, None, None, None, None, None
    ]))
    xo.assert_allclose(tt_opt.vkick, np.array([
        0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0.
    ]))
    xo.assert_allclose(tw_opt.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

def test_thick_kicker_cpymad_loader():

    mad_data = """

    kk: kicker, l=1, hkick=1e-3, vkick=2e-3;

    ss: sequence, l=3;
    kk1: kk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)

    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 6 rows, 122 cols
    # name                   s element_type isthick isreplica ...
    # ss$start               0 Marker         False     False
    # drift_0                0 Drift           True     False
    # kk1                  0.5 Multipole       True     False
    # drift_1              1.5 Drift           True     False
    # ss$end                 3 Marker         False     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['ss$start', 'drift_0', 'kk1', 'drift_1', 'ss$end', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
    assert np.all(tt.element_type == np.array(
        ['Marker', 'Drift', 'Multipole', 'Drift', 'Marker', '']))
    assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))
    xo.assert_allclose(tt.vkick, np.array([0.   , 0.   , 0.002, 0.   , 0.   , 0.   ]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 10 rows, 122 cols
    # name                   s element_type        isthick isreplica ...
    # ss$start               0 Marker                False     False
    # drift_0                0 Drift                  True     False
    # kk1_entry            0.5 Marker                False     False
    # kk1..0               0.5 ThickSliceMultipole    True     False
    # kk1..1          0.833333 ThickSliceMultipole    True     False
    # kk1..2           1.16667 ThickSliceMultipole    True     False
    # kk1_exit             1.5 Marker                False     False
    # drift_1              1.5 Drift                  True     False
    # ss$end                 3 Marker                False     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'ss$start', 'drift_0', 'kk1_entry', 'kk1..0', 'kk1..1', 'kk1..2',
        'kk1_exit', 'drift_1', 'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        False, True, False, True, True, True, False, True, False, False
    ]))
    xo.assert_allclose(tt_slice_thick.hkick, np.array([
        0., 0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tt_slice_thick.vkick, np.array([
        0., 0., 0., 0.002/3, 0.002/3, 0.002/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 14 rows, 122 cols
    # name                     s element_type        isthick isreplica ...
    # ss$start                 0 Marker                False     False
    # drift_0                  0 Drift                  True     False
    # kk1_entry              0.5 Marker                False     False
    # drift_kk1..0           0.5 DriftSliceMultipole    True     False
    # kk1..3                0.75 ThinSliceMultipole    False     False
    # drift_kk1..1          0.75 DriftSliceMultipole    True     False
    # kk1..4                   1 ThinSliceMultipole    False     False
    # drift_kk1..2             1 DriftSliceMultipole    True     False
    # kk1..5                1.25 ThinSliceMultipole    False     False
    # drift_kk1..3          1.25 DriftSliceMultipole    True     False
    # kk1_exit               1.5 Marker                False     False
    # drift_1                1.5 Drift                  True     False
    # ss$end                   3 Marker                False     False
    # _end_point               3                       False     False

    assert np.all(tt_slice_thin.name == np.array([
        'ss$start', 'drift_0', 'kk1_entry', 'drift_kk1..0', 'kk1..3', 'drift_kk1..1',
        'kk1..4', 'drift_kk1..2', 'kk1..5', 'drift_kk1..3', 'kk1_exit', 'drift_1',
        'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        False, True, False, True, False, True, False, True, False, True, False, True, False, False
    ]))
    assert np.allclose(tt_slice_thin.hkick, np.array([
        0., 0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0., 0.
    ]))
    assert np.allclose(tt_slice_thin.vkick, np.array([
        0., 0., 0., 0., 0.002/3, 0., 0.002/3, 0., 0.002/3, 0., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)



def test_thick_kicker_native_loader():

    mad_data = """

    kk: kicker, l=1, hkick=1e-3, vkick=2e-3;

    ss: sequence, l=3;
    kk1: kk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    env = xt.load(string=mad_data, format='madx')
    line = env.ss

    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 4 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # drift_1                0 Drift           True     False
    # kk1                  0.5 Multipole       True     False
    # drift_2              1.5 Drift           True     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['drift_1', 'kk1', 'drift_2', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0.5, 1.5, 3]))
    assert np.all(tt.element_type == np.array(
        ['Drift', 'Multipole', 'Drift', '']))
    assert np.all(tt.isthick == np.array([True, True, True, False]))
    xo.assert_allclose(tt.vkick, np.array([0., 0.002, 0., 0.]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 8 rows, 130 cols
    # name                   s element_type        isthick isreplica ...
    # drift_1                0 Drift                  True     False
    # vk1_entry            0.5 Marker                False     False
    # vk1..0               0.5 ThickSliceMultipole    True     False
    # vk1..1          0.833333 ThickSliceMultipole    True     False
    # vk1..2           1.16667 ThickSliceMultipole    True     False
    # vk1_exit             1.5 Marker                False     False
    # drift_2              1.5 Drift                  True     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'drift_1', 'kk1_entry', 'kk1..0', 'kk1..1', 'kk1..2', 'kk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        True, False, True, True, True, False, True, False
    ]))
    xo.assert_allclose(tt_slice_thick.hkick, np.array([
        0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0.
    ]))
    xo.assert_allclose(tt_slice_thick.vkick, np.array([
        0., 0., 0.002/3, 0.002/3, 0.002/3, 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 12 rows, 130 cols
    # name                     s element_type        isthick isreplica ...
    # drift_1                  0 Drift                  True     False
    # kk1_entry              0.5 Marker                False     False
    # drift_kk1..0           0.5 DriftSliceMultipole    True     False
    # kk1..3                0.75 ThinSliceMultipole    False     False
    # drift_kk1..1          0.75 DriftSliceMultipole    True     False
    # kk1..4                   1 ThinSliceMultipole    False     False
    # drift_kk1..2             1 DriftSliceMultipole    True     False
    # kk1..5                1.25 ThinSliceMultipole    False     False
    # drift_kk1..3          1.25 DriftSliceMultipole    True     False
    # kk1_exit               1.5 Marker                False     False
    # drift_2                1.5 Drift                  True     False
    # _end_point               3                       False     False
    assert np.all(tt_slice_thin.name == np.array([
        'drift_1', 'kk1_entry', 'drift_kk1..0', 'kk1..3', 'drift_kk1..1',
        'kk1..4', 'drift_kk1..2', 'kk1..5', 'drift_kk1..3', 'kk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        True, False, True, False, True, False, True, False, True, False, True, False
    ]))
    assert np.allclose(tt_slice_thin.hkick, np.array([
        0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0.
    ]))
    assert np.allclose(tt_slice_thin.vkick, np.array([
        0., 0., 0., 0.002/3, 0., 0.002/3, 0., 0.002/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_opt = line_slice_thin.copy(shallow=True)
    line_opt.build_tracker()
    line_opt.optimize_for_tracking()
    tw_opt = line_opt.twiss(betx=1, bety=1)
    tt_opt = line_opt.get_table(attr=True)
    # is:
    # Table: 8 rows, 130 cols
    # name                     s element_type isthick isreplica parent_name ...
    # drift_0                  0 Drift           True     False None
    # kk1..3                0.75 Multipole      False     False None
    # drift_kk1..1          0.75 Drift           True     False None
    # kk1..4                   1 Multipole      False     False None
    # drift_kk1..2             1 Drift           True     False None
    # kk1..5                1.25 Multipole      False     False None
    # drift_kk1..3          1.25 Drift           True     False None
    # _end_point               3                False     False None

    assert np.all(tt_opt.element_type == np.array([
        'Drift', 'Multipole', 'Drift', 'Multipole', 'Drift',
        'Multipole', 'Drift', ''
    ]))
    assert np.all(tt_opt.isthick == np.array([
        True, False, True, False, True, False, True, False
    ]))
    assert np.all(tt_opt.isreplica == np.array([
        False, False, False, False, False, False, False, False
    ]))
    assert np.all(tt_opt.parent_name == np.array([
        None, None, None, None, None, None, None, None
    ]))
    xo.assert_allclose(tt_opt.hkick, np.array([
        0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0.
    ]))
    xo.assert_allclose(tt_opt.vkick, np.array([
        0., 0.002/3, 0., 0.002/3, 0., 0.002/3, 0., 0.
    ]))
    xo.assert_allclose(tw_opt.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

def test_thick_tkicker_cpymad_loader():

    mad_data = """

    kk: tkicker, l=1, hkick=1e-3, vkick=2e-3;

    ss: sequence, l=3;
    kk1: kk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)

    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 6 rows, 122 cols
    # name                   s element_type isthick isreplica ...
    # ss$start               0 Marker         False     False
    # drift_0                0 Drift           True     False
    # kk1                  0.5 Multipole       True     False
    # drift_1              1.5 Drift           True     False
    # ss$end                 3 Marker         False     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['ss$start', 'drift_0', 'kk1', 'drift_1', 'ss$end', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
    assert np.all(tt.element_type == np.array(
        ['Marker', 'Drift', 'Multipole', 'Drift', 'Marker', '']))
    assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))
    xo.assert_allclose(tt.vkick, np.array([0.   , 0.   , 0.002, 0.   , 0.   , 0.   ]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 10 rows, 122 cols
    # name                   s element_type        isthick isreplica ...
    # ss$start               0 Marker                False     False
    # drift_0                0 Drift                  True     False
    # kk1_entry            0.5 Marker                False     False
    # kk1..0               0.5 ThickSliceMultipole    True     False
    # kk1..1          0.833333 ThickSliceMultipole    True     False
    # kk1..2           1.16667 ThickSliceMultipole    True     False
    # kk1_exit             1.5 Marker                False     False
    # drift_1              1.5 Drift                  True     False
    # ss$end                 3 Marker                False     False
    # _end_point             3                       False     False

    assert np.all(tt_slice_thick.name == np.array([
        'ss$start', 'drift_0', 'kk1_entry', 'kk1..0', 'kk1..1', 'kk1..2',
        'kk1_exit', 'drift_1', 'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        False, True, False, True, True, True, False, True, False, False
    ]))
    xo.assert_allclose(tt_slice_thick.hkick, np.array([
        0., 0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tt_slice_thick.vkick, np.array([
        0., 0., 0., 0.002/3, 0.002/3, 0.002/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 14 rows, 122 cols
    # name                     s element_type        isthick isreplica ...
    # ss$start                 0 Marker                False     False
    # drift_0                  0 Drift                  True     False
    # kk1_entry              0.5 Marker                False     False
    # drift_kk1..0           0.5 DriftSliceMultipole    True     False
    # kk1..3                0.75 ThinSliceMultipole    False     False
    # drift_kk1..1          0.75 DriftSliceMultipole    True     False
    # kk1..4                   1 ThinSliceMultipole    False     False
    # drift_kk1..2             1 DriftSliceMultipole    True     False
    # kk1..5                1.25 ThinSliceMultipole    False     False
    # drift_kk1..3          1.25 DriftSliceMultipole    True     False
    # kk1_exit               1.5 Marker                False     False
    # drift_1                1.5 Drift                  True     False
    # ss$end                   3 Marker                False     False
    # _end_point               3                       False     False

    assert np.all(tt_slice_thin.name == np.array([
        'ss$start', 'drift_0', 'kk1_entry', 'drift_kk1..0', 'kk1..3', 'drift_kk1..1',
        'kk1..4', 'drift_kk1..2', 'kk1..5', 'drift_kk1..3', 'kk1_exit', 'drift_1',
        'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        False, True, False, True, False, True, False, True, False, True, False, True, False, False
    ]))
    assert np.allclose(tt_slice_thin.hkick, np.array([
        0., 0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0., 0.
    ]))
    assert np.allclose(tt_slice_thin.vkick, np.array([
        0., 0., 0., 0., 0.002/3, 0., 0.002/3, 0., 0.002/3, 0., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

def test_thick_tkicker_native_loader():

    mad_data = """

    kk: tkicker, l=1, hkick=1e-3, vkick=2e-3;

    ss: sequence, l=3;
    kk1: kk, at=1;
    endsequence;
    """

    mad_computation = """
    beam;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    env = xt.load(string=mad_data, format='madx')
    line = env.ss

    line.particle_ref = xt.Particles(p0c=1e9)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 4 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # drift_1                0 Drift           True     False
    # kk1                  0.5 Multipole       True     False
    # drift_2              1.5 Drift           True     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['drift_1', 'kk1', 'drift_2', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0.5, 1.5, 3]))
    assert np.all(tt.element_type == np.array(
        ['Drift', 'Multipole', 'Drift', '']))
    assert np.all(tt.isthick == np.array([True, True, True, False]))
    xo.assert_allclose(tt.vkick, np.array([0., 0.002, 0., 0.]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Multipole),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    #is:
    # Table: 8 rows, 130 cols
    # name                   s element_type        isthick isreplica ...
    # drift_1                0 Drift                  True     False
    # kk1_entry            0.5 Marker                False     False
    # kk1..0               0.5 ThickSliceMultipole    True     False
    # kk1..1          0.833333 ThickSliceMultipole    True     False
    # kk1..2           1.16667 ThickSliceMultipole    True     False
    # kk1_exit             1.5 Marker                False     False
    # drift_2              1.5 Drift                  True     False
    # _end_point             3                       False     False
    assert np.all(tt_slice_thick.name == np.array([
        'drift_1', 'kk1_entry', 'kk1..0', 'kk1..1', 'kk1..2',
        'kk1_exit', 'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'ThickSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        True, False, True, True, True, False, True, False
    ]))
    xo.assert_allclose(tt_slice_thick.hkick, np.array([
        0., 0., 0.001/3, 0.001/3, 0.001/3, 0., 0., 0.
    ]))
    xo.assert_allclose(tt_slice_thick.vkick, np.array([
        0., 0., 0.002/3, 0.002/3, 0.002/3, 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Multipole),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 12 rows, 130 cols
    # name                     s element_type        isthick isreplica ...
    # drift_1                  0 Drift                  True     False
    # kk1_entry              0.5 Marker                False     False
    # drift_kk1..0           0.5 DriftSliceMultipole    True     False
    # kk1..3                0.75 ThinSliceMultipole    False     False
    # drift_kk1..1          0.75 DriftSliceMultipole    True     False
    # kk1..4                   1 ThinSliceMultipole    False     False
    # drift_kk1..2             1 DriftSliceMultipole    True     False
    # kk1..5                1.25 ThinSliceMultipole    False     False
    # drift_kk1..3          1.25 DriftSliceMultipole    True     False
    # kk1_exit               1.5 Marker                False     False
    # drift_2                1.5 Drift                  True     False
    # _end_point               3                       False     False
    assert np.all(tt_slice_thin.name == np.array([
        'drift_1', 'kk1_entry', 'drift_kk1..0', 'kk1..3', 'drift_kk1..1',
        'kk1..4', 'drift_kk1..2', 'kk1..5', 'drift_kk1..3', 'kk1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Drift', 'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        True, False, True, False, True, False, True, False, True, False, True, False
    ]))
    assert np.allclose(tt_slice_thin.hkick, np.array([
        0., 0., 0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0., 0., 0.
    ]))
    assert np.allclose(tt_slice_thin.vkick, np.array([
        0., 0., 0., 0.002/3, 0., 0.002/3, 0., 0.002/3, 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

    line_opt = line_slice_thin.copy(shallow=True)
    line_opt.build_tracker()
    line_opt.optimize_for_tracking()
    tw_opt = line_opt.twiss(betx=1, bety=1)
    tt_opt = line_opt.get_table(attr=True)
    # is:
    # Table: 8 rows, 130 cols
    # name                     s element_type isthick isreplica parent_name ...
    # drift_0                  0 Drift           True     False None
    # kk1..3                0.75 Multipole      False     False None
    # drift_kk1..1          0.75 Drift           True     False None
    # kk1..4                   1 Multipole      False     False None
    # drift_kk1..2             1 Drift           True     False None
    # kk1..5                1.25 Multipole      False     False None
    # drift_kk1..3          1.25 Drift           True     False None
    # _end_point               3                False     False None

    assert np.all(tt_opt.element_type == np.array([
        'Drift', 'Multipole', 'Drift', 'Multipole', 'Drift',
        'Multipole', 'Drift', ''
    ]))
    assert np.all(tt_opt.isthick == np.array([
        True, False, True, False, True, False, True, False
    ]))
    assert np.all(tt_opt.isreplica == np.array([
        False, False, False, False, False, False, False, False
    ]))
    assert np.all(tt_opt.parent_name == np.array([
        None, None, None, None, None, None, None, None
    ]))
    xo.assert_allclose(tt_opt.hkick, np.array([
        0., 0.001/3, 0., 0.001/3, 0., 0.001/3, 0., 0.
    ]))
    xo.assert_allclose(tt_opt.vkick, np.array([
        0., 0.002/3, 0., 0.002/3, 0., 0.002/3, 0., 0.
    ]))
    xo.assert_allclose(tw_opt.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)

def test_thick_cavity_cpymad_loader():

    mad_data = """

    cav: rfcavity, l=1, volt=3, freq=400, lag=0.4;

    ss: sequence, l=3;
    cav1: cav, at=1;
    endsequence;
    """

    mad_computation = """
    beam, particle=proton, pc=1e3;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)

    line.particle_ref = xt.Particles(p0c=1e12)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 6 rows, 122 cols
    # name                   s element_type isthick isreplica ...
    # ss$start               0 Marker         False     False
    # drift_0                0 Drift           True     False
    # cav1                 0.5 Cavity          True     False
    # drift_1              1.5 Drift           True     False
    # ss$end                 3 Marker         False     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['ss$start', 'drift_0', 'cav1', 'drift_1', 'ss$end', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
    assert np.all(tt.element_type == np.array(
        ['Marker', 'Drift', 'Cavity', 'Drift', 'Marker', '']))
    xo.assert_allclose(tt.voltage, np.array(
        [      0.,       0., 3000000.,       0.,       0.,       0.]))
    xo.assert_allclose(tt.lag, np.array([0., 0., 144., 0., 0., 0.]))
    xo.assert_allclose(tt.frequency, np.array(
        [0.e+00, 0.e+00, 4.e+08, 0.e+00, 0.e+00, 0.e+00]))
    assert np.all(tt.isthick == np.array(
        [False, True, True, True, False, False]))

    xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-9)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Cavity),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    # is:
    # Table: 10 rows, 122 cols
    # name                   s element_type     isthick isreplica ...
    # ss$start               0 Marker             False     False
    # drift_0                0 Drift               True     False
    # cav1_entry           0.5 Marker             False     False
    # cav1..0              0.5 ThickSliceCavity    True     False
    # cav1..1         0.833333 ThickSliceCavity    True     False
    # cav1..2          1.16667 ThickSliceCavity    True     False
    # cav1_exit            1.5 Marker             False     False
    # drift_1              1.5 Drift               True     False
    # ss$end                 3 Marker             False     False
    # _end_point             3                    False     False


    assert np.all(tt_slice_thick.name == np.array([
        'ss$start', 'drift_0', 'cav1_entry', 'cav1..0', 'cav1..1', 'cav1..2',
        'cav1_exit', 'drift_1', 'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'ThickSliceCavity', 'ThickSliceCavity',
        'ThickSliceCavity', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        False, True, False, True, True, True, False, True, False, False
    ]))
    xo.assert_allclose(tt_slice_thick.voltage, np.array([
            0.,       0.,       0., 1000000., 1000000., 1000000.,
            0.,       0.,       0.,       0.]))
    xo.assert_allclose(tt_slice_thick.lag, np.array(
        [0., 0., 0., 144., 144., 144., 0., 0., 0., 0.]))
    xo.assert_allclose(tt_slice_thick.frequency, np.array([
        0.e+00, 0.e+00, 0.e+00, 4.e+08, 4.e+08, 4.e+08, 0.e+00, 0.e+00, 0.e+00, 0.e+00
    ]))


    xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Cavity),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 14 rows, 124 cols
    # name                      s element_type     isthick isreplica ...
    # ss$start                  0 Marker             False     False
    # drift_0                   0 Drift               True     False
    # cav1_entry              0.5 Marker             False     False
    # drift_cav1..0           0.5 DriftSliceCavity    True     False
    # cav1..3                0.75 ThinSliceCavity    False     False
    # drift_cav1..1          0.75 DriftSliceCavity    True     False
    # cav1..4                   1 ThinSliceCavity    False     False
    # drift_cav1..2             1 DriftSliceCavity    True     False
    # cav1..5                1.25 ThinSliceCavity    False     False
    # drift_cav1..3          1.25 DriftSliceCavity    True     False
    # cav1_exit               1.5 Marker             False     False
    # drift_1                 1.5 Drift               True     False
    # ss$end                    3 Marker             False     False
    # _end_point                3                    False     False

    assert np.all(tt_slice_thin.name == np.array([
        'ss$start', 'drift_0', 'cav1_entry', 'drift_cav1..0', 'cav1..3', 'drift_cav1..1',
        'cav1..4', 'drift_cav1..2', 'cav1..5', 'drift_cav1..3', 'cav1_exit', 'drift_1',
        'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0, 0.5, 0.5, 0.75, 0.75, 1, 1, 1.25, 1.25, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'DriftSliceCavity', 'ThinSliceCavity',
        'DriftSliceCavity', 'ThinSliceCavity', 'DriftSliceCavity', 'ThinSliceCavity',
        'DriftSliceCavity', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        False, True, False, True, False, True, False, True, False, True, False, True, False, False
    ]))
    xo.assert_allclose(tt_slice_thin.voltage, np.array([
                0.,       0.,       0.,       0., 1000000.,       0.,
        1000000.,       0., 1000000.,       0.,       0.,       0.,
                0.,       0.]))
    xo.assert_allclose(tt_slice_thin.frequency, np.array([
        0.e+00, 0.e+00, 0.e+00, 0.e+00, 4.e+08, 0.e+00, 4.e+08, 0.e+00,
        4.e+08, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00
    ]))
    xo.assert_allclose(tt_slice_thin.lag, np.array([
        0., 0., 0., 0., 144., 0., 144., 0., 144., 0., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

def test_thick_cavity_native_loader():

    mad_data = """

    cav: rfcavity, l=1, volt=3, freq=400, lag=0.4;

    ss: sequence, l=3;
    cav1: cav, at=1;
    endsequence;
    """

    mad_computation = """
    beam, particle=proton, pc=1e3;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """

    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    env = xt.load(string=mad_data, format='madx')
    line = env.ss

    line.particle_ref = xt.Particles(p0c=1e12)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 4 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # drift_1                0 Drift           True     False
    # cav1                 0.5 Cavity          True     False
    # drift_2              1.5 Drift           True     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array(
        ['drift_1', 'cav1', 'drift_2', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0.5, 1.5, 3]))
    assert np.all(tt.element_type == np.array(
        ['Drift', 'Cavity', 'Drift', '']))
    xo.assert_allclose(tt.voltage, np.array(
        [0., 3000000., 0., 0.]))
    xo.assert_allclose(tt.lag, np.array([0., 144., 0., 0.]))
    xo.assert_allclose(tt.frequency, np.array(
        [0.e+00, 4.e+08, 0.e+00, 0.e+00]))
    assert np.all(tt.isthick == np.array(
        [True, True, True, False]))

    xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-9)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.Cavity),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    # is:
    # Table: 8 rows, 130 cols
    # name                   s element_type     isthick isreplica ...
    # drift_1                0 Drift               True     False
    # cav1_entry           0.5 Marker             False     False
    # cav1..0              0.5 ThickSliceCavity    True     False
    # cav1..1         0.833333 ThickSliceCavity    True     False
    # cav1..2          1.16667 ThickSliceCavity    True     False
    # cav1_exit            1.5 Marker             False     False
    # drift_2              1.5 Drift               True     False
    # _end_point             3                    False     False

    assert np.all(tt_slice_thick.name == np.array([
        'drift_1', 'cav1_entry', 'cav1..0', 'cav1..1', 'cav1..2',
        'cav1_exit', 'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Drift', 'Marker', 'ThickSliceCavity', 'ThickSliceCavity',
        'ThickSliceCavity', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        True, False, True, True, True, False, True, False
    ]))
    xo.assert_allclose(tt_slice_thick.voltage, np.array([
        0., 0., 1000000., 1000000., 1000000., 0., 0., 0.
    ]))
    xo.assert_allclose(tt_slice_thick.lag, np.array(
        [0., 0., 144., 144., 144., 0., 0., 0.]))
    xo.assert_allclose(tt_slice_thick.frequency, np.array([
        0.e+00, 0.e+00, 4.e+08, 4.e+08, 4.e+08, 0.e+00, 0.e+00, 0.e+00
    ]))

    xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.Cavity),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 12 rows, 130 cols
    # name                      s element_type     isthick isreplica ...
    # drift_1                   0 Drift               True     False
    # cav1_entry              0.5 Marker             False     False
    # drift_cav1..0           0.5 DriftSliceCavity    True     False
    # cav1..3                0.75 ThinSliceCavity    False     False
    # drift_cav1..1          0.75 DriftSliceCavity    True     False
    # cav1..4                   1 ThinSliceCavity    False     False
    # drift_cav1..2             1 DriftSliceCavity    True     False
    # cav1..5                1.25 ThinSliceCavity    False     False
    # drift_cav1..3          1.25 DriftSliceCavity    True     False
    # cav1_exit               1.5 Marker             False     False
    # drift_2                 1.5 Drift               True     False
    # _end_point                3                    False     False
    assert np.all(tt_slice_thin.name == np.array([
        'drift_1', 'cav1_entry', 'drift_cav1..0', 'cav1..3', 'drift_cav1..1',
        'cav1..4', 'drift_cav1..2', 'cav1..5', 'drift_cav1..3', 'cav1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0.5, 0.5, 0.75, 0.75, 1, 1, 1.25, 1.25, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Drift', 'Marker', 'DriftSliceCavity', 'ThinSliceCavity',
        'DriftSliceCavity', 'ThinSliceCavity', 'DriftSliceCavity',
        'ThinSliceCavity', 'DriftSliceCavity', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        True, False, True, False, True, False, True, False, True, False, True, False
    ]))
    xo.assert_allclose(tt_slice_thin.voltage, np.array([
        0., 0., 0., 1000000., 0., 1000000., 0., 1000000., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tt_slice_thin.frequency, np.array([
        0.e+00, 0.e+00, 0.e+00, 4.e+08, 0.e+00, 4.e+08, 0.e+00,
        4.e+08, 0.e+00, 0.e+00, 0.e+00, 0.e+00
    ]))
    xo.assert_allclose(tt_slice_thin.lag, np.array([
        0., 0., 0., 144., 0., 144., 0., 144., 0., 0., 0., 0.
    ]))
    xo.assert_allclose(tw_slice_thin.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

    line_opt = line_slice_thin.copy(shallow=True)
    line_opt.build_tracker()
    line_opt.optimize_for_tracking()
    tw_opt = line_opt.twiss(betx=1, bety=1)
    tt_opt = line_opt.get_table(attr=True)
    # is:
    # Table: 8 rows, 130 cols
    # name                      s element_type isthick isreplica parent_name ...
    # drift_1                   0 Drift           True     False None
    # cav1..3                0.75 Cavity          True     False None
    # drift_cav1..1          0.75 Drift           True     False None
    # cav1..4                   1 Cavity          True     False None
    # drift_cav1..2             1 Drift           True     False None
    # cav1..5                1.25 Cavity          True     False None
    # drift_cav1..3          1.25 Drift           True     False None
    # _end_point                3                False     False None

    assert np.all(tt_opt.name == np.array([
        'drift_1', 'cav1..3', 'drift_cav1..1', 'cav1..4', 'drift_cav1..2',
        'cav1..5', 'drift_cav1..3', '_end_point'
    ]))
    assert np.all(tt_opt.s == np.array([
        0, 0.75, 0.75, 1, 1, 1.25, 1.25, 3
    ]))
    assert np.all(tt_opt.element_type == np.array([
        'Drift', 'Cavity', 'Drift', 'Cavity', 'Drift', 'Cavity', 'Drift', ''
    ]))
    assert np.all(tt_opt.isthick == np.array([
        True, True, True, True, True, True, True, False
    ]))
    assert np.all(tt_opt.isreplica == np.array([
        False, False, False, False, False, False, False, False
    ]))
    assert np.all(tt_opt.parent_name == np.array([
        None, None, None, None, None, None, None, None
    ]))

    xo.assert_allclose(tw_opt.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw_opt.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

def test_crabcavity_thick_cpymad_loader():

    mad_data = """

    cc: crabcavity, l=1, volt=4, freq=400e+06, lag=0.3;

    ss: sequence, l=3;
    cc1: cc, at=1;
    endsequence;
    """

    mad_computation = """
    beam, particle=proton, pc=1e3;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """
    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    line = xt.Line.from_madx_sequence(madx.sequence.ss, deferred_expressions=True)
    line.particle_ref = xt.Particles(p0c=1e12)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 6 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # ss$start               0 Marker         False     False
    # drift_0                0 Drift           True     False
    # cc1                  0.5 CrabCavity      True     False
    # drift_1              1.5 Drift           True     False
    # ss$end                 3 Marker         False     False
    # _end_point             3                False     False


    assert np.all(tt.name == np.array(
        ['ss$start', 'drift_0', 'cc1', 'drift_1', 'ss$end', '_end_point']))
    xo.assert_allclose(tt.s, np.array([0, 0, 0.5, 1.5, 3, 3]))
    assert np.all(tt.element_type == np.array(
        ['Marker', 'Drift', 'CrabCavity', 'Drift', 'Marker', '']))
    assert np.all(tt.isthick == np.array([False, True, True, True, False, False]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-14)
    xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-14)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.CrabCavity),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    # is:
    # Table: 10 rows, 130 cols
    # name                   s element_type         isthick isreplica ...
    # ss$start               0 Marker                 False     False
    # drift_0                0 Drift                   True     False
    # cc1_entry            0.5 Marker                 False     False
    # cc1..0               0.5 ThickSliceCrabCavity    True     False
    # cc1..1          0.833333 ThickSliceCrabCavity    True     False
    # cc1..2           1.16667 ThickSliceCrabCavity    True     False
    # cc1_exit             1.5 Marker                 False     False
    # drift_1              1.5 Drift                   True     False
    # ss$end                 3 Marker                 False     False
    # _end_point             3                        False     False

    assert np.all(tt_slice_thick.name == np.array([
        'ss$start', 'drift_0', 'cc1_entry', 'cc1..0', 'cc1..1', 'cc1..2',
        'cc1_exit', 'drift_1', 'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'ThickSliceCrabCavity', 'ThickSliceCrabCavity',
        'ThickSliceCrabCavity', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        False, True, False, True, True, True, False, True, False, False
    ]))
    # The MAD-X element is implemented with a single kick in the middle, so it is
    # normal to see small differences
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.CrabCavity),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 14 rows, 130 cols
    # name                     s element_type         isthick isreplica ...
    # ss$start                 0 Marker                 False     False
    # drift_0                  0 Drift                   True     False
    # cc1_entry              0.5 Marker                 False     False
    # drift_cc1..0           0.5 DriftSliceCrabCavity    True     False
    # cc1..3                0.75 ThinSliceCrabCavity    False     False
    # drift_cc1..1          0.75 DriftSliceCrabCavity    True     False
    # cc1..4                   1 ThinSliceCrabCavity    False     False
    # drift_cc1..2             1 DriftSliceCrabCavity    True     False
    # cc1..5                1.25 ThinSliceCrabCavity    False     False
    # drift_cc1..3          1.25 DriftSliceCrabCavity    True     False
    # cc1_exit               1.5 Marker                 False     False
    # drift_1                1.5 Drift                   True     False
    # ss$end                   3 Marker                 False     False
    # _end_point               3                        False     False

    assert np.all(tt_slice_thin.name == np.array([
        'ss$start', 'drift_0', 'cc1_entry', 'drift_cc1..0', 'cc1..3', 'drift_cc1..1',
        'cc1..4', 'drift_cc1..2', 'cc1..5', 'drift_cc1..3', 'cc1_exit', 'drift_1',
        'ss$end', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Marker', 'Drift', 'Marker', 'DriftSliceCrabCavity', 'ThinSliceCrabCavity',
        'DriftSliceCrabCavity', 'ThinSliceCrabCavity', 'DriftSliceCrabCavity',
        'ThinSliceCrabCavity', 'DriftSliceCrabCavity', 'Marker', 'Drift', 'Marker', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        False, True, False, True, False, True, False, True, False, True, False, True, False, False
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thin.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)

def test_crabcavity_thick_native_loader():

    mad_data = """

    cc: crabcavity, l=1, volt=4, freq=400e+06, lag=0.3;

    ss: sequence, l=3;
    cc1: cc, at=1;
    endsequence;
    """

    mad_computation = """
    beam, particle=proton, pc=1e3;
    use, sequence=ss;
    twiss, betx=1, bety=1;
    """
    madx = Madx()

    madx.input(mad_data)
    madx.input(mad_computation)

    tw_mad = xt.Table(madx.table.twiss, _copy_cols=True)

    env = xt.load(string=mad_data, format='madx')
    line = env.ss

    line.particle_ref = xt.Particles(p0c=1e12)

    tw = line.twiss(betx=1, bety=1)
    tt = line.get_table(attr=True)
    # is:
    # Table: 4 rows, 130 cols
    # name                   s element_type isthick isreplica ...
    # drift_1                0 Drift           True     False
    # cc1                  0.5 CrabCavity      True     False
    # drift_2              1.5 Drift           True     False
    # _end_point             3                False     False

    assert np.all(tt.name == np.array([
        'drift_1', 'cc1', 'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt.s, np.array([0, 0.5, 1.5, 3]), rtol=0, atol=1e-7)
    assert np.all(tt.element_type == np.array([
        'Drift', 'CrabCavity', 'Drift', ''
    ]))
    assert np.all(tt.isthick == np.array([True, True, True, False]))

    xo.assert_allclose(tw.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)

    line_slice_thick = line.copy(shallow=True)
    line_slice_thick.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thick'), element_type=xt.CrabCavity),
    ])
    tw_slice_thick = line_slice_thick.twiss(betx=1, bety=1)
    tt_slice_thick = line_slice_thick.get_table(attr=True)
    # is:
    # Table: 8 rows, 130 cols
    # name                   s element_type         isthick isreplica ...
    # drift_1                0 Drift                   True     False
    # cc1_entry            0.5 Marker                 False     False
    # cc1..0               0.5 ThickSliceCrabCavity    True     False
    # cc1..1          0.833333 ThickSliceCrabCavity    True     False
    # cc1..2           1.16667 ThickSliceCrabCavity    True     False
    # cc1_exit             1.5 Marker                 False     False
    # drift_2              1.5 Drift                   True     False
    # _end_point             3                        False     False
    assert np.all(tt_slice_thick.name == np.array([
        'drift_1', 'cc1_entry', 'cc1..0', 'cc1..1', 'cc1..2',
        'cc1_exit', 'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thick.s, np.array([
        0, 0.5, 0.5, 0.83333333, 1.16666667, 1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thick.element_type == np.array([
        'Drift', 'Marker', 'ThickSliceCrabCavity', 'ThickSliceCrabCavity',
        'ThickSliceCrabCavity', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thick.isthick == np.array([
        True, False, True, True, True, False, True, False
    ]))
    # The MAD-X element is implemented with a single kick in the middle, so it is
    # normal to see small differences
    xo.assert_allclose(tw_slice_thick.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thick.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thick.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thick.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thick.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)

    line_slice_thin = line.copy(shallow=True)
    line_slice_thin.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(3, mode='thin'), element_type=xt.CrabCavity),
    ])
    tw_slice_thin = line_slice_thin.twiss(betx=1, bety=1)
    tt_slice_thin = line_slice_thin.get_table(attr=True)
    # is:
    # Table: 12 rows, 130 cols
    # name                     s element_type         isthick isreplica ...
    # drift_1                  0 Drift                   True     False
    # cc1_entry              0.5 Marker                 False     False
    # drift_cc1..0           0.5 DriftSliceCrabCavity    True     False
    # cc1..3                0.75 ThinSliceCrabCavity    False     False
    # drift_cc1..1          0.75 DriftSliceCrabCavity    True     False
    # cc1..4                   1 ThinSliceCrabCavity    False     False
    # drift_cc1..2             1 DriftSliceCrabCavity    True     False
    # cc1..5                1.25 ThinSliceCrabCavity    False     False
    # drift_cc1..3          1.25 DriftSliceCrabCavity    True     False
    # cc1_exit               1.5 Marker                 False     False
    # drift_2                1.5 Drift                   True     False
    # _end_point               3                        False     False
    assert np.all(tt_slice_thin.name == np.array([
        'drift_1', 'cc1_entry', 'drift_cc1..0', 'cc1..3', 'drift_cc1..1',
        'cc1..4', 'drift_cc1..2', 'cc1..5', 'drift_cc1..3', 'cc1_exit',
        'drift_2', '_end_point'
    ]))
    xo.assert_allclose(tt_slice_thin.s, np.array([
        0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25,
        1.5, 1.5, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_slice_thin.element_type == np.array([
        'Drift', 'Marker', 'DriftSliceCrabCavity', 'ThinSliceCrabCavity',
        'DriftSliceCrabCavity', 'ThinSliceCrabCavity', 'DriftSliceCrabCavity',
        'ThinSliceCrabCavity', 'DriftSliceCrabCavity', 'Marker', 'Drift', ''
    ]))
    assert np.all(tt_slice_thin.isthick == np.array([
        True, False, True, False, True, False, True, False, True, False, True, False
    ]))
    xo.assert_allclose(tw_slice_thin.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thin.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_slice_thin.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thin.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_slice_thin.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)

    line_opt = line_slice_thin.copy(shallow=True)
    line_opt.build_tracker()
    line_opt.optimize_for_tracking()
    tw_opt = line_opt.twiss(betx=1, bety=1)
    tt_opt = line_opt.get_table(attr=True)
    # is:
    # name                     s element_type isthick isreplica parent_name ...
    # drift_1                  0 Drift           True     False None
    # cc1..3                0.75 CrabCavity      True     False None
    # drift_cc1..1          0.75 Drift           True     False None
    # cc1..4                   1 CrabCavity      True     False None
    # drift_cc1..2             1 Drift           True     False None
    # cc1..5                1.25 CrabCavity      True     False None
    # drift_cc1..3          1.25 Drift           True     False None
    # _end_point               3                False     False None

    assert np.all(tt_opt.name == np.array([
        'drift_1', 'cc1..3', 'drift_cc1..1', 'cc1..4', 'drift_cc1..2',
        'cc1..5', 'drift_cc1..3', '_end_point'
    ]))
    xo.assert_allclose(tt_opt.s, np.array([
        0, 0.75, 0.75, 1, 1, 1.25, 1.25, 3
    ]), rtol=0, atol=1e-7)
    assert np.all(tt_opt.element_type == np.array([
        'Drift', 'CrabCavity', 'Drift', 'CrabCavity', 'Drift',
        'CrabCavity', 'Drift', ''
    ]))
    assert np.all(tt_opt.isthick == np.array([
        True, True, True, True, True, True, True, False
    ]))
    xo.assert_allclose(tw_opt.px[-1], tw_mad.px[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_opt.py[-1], tw_mad.py[-1], rtol=0, atol=1e-11)
    xo.assert_allclose(tw_opt.x[-1], tw_mad.x[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_opt.y[-1], tw_mad.y[-1], rtol=0, atol=1e-10)
    xo.assert_allclose(tw_opt.ptau[-1], tw_mad.pt[-1], rtol=0, atol=1e-5)


test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_crab_dispersion_hlllhc_b1_and_b2():

    mad = Madx()

    mad.input(f"""
    call,file="{str(test_data_folder)}/hllhc15_thick/lhc.seq";
    call,file="{str(test_data_folder)}/hllhc15_thick/hllhc_sequence.madx";
    call,file="{str(test_data_folder)}/hllhc15_thick/macro.madx";
    call,file="{str(test_data_folder)}/hllhc15_thick/enable_crabcavities.madx";
    call,file="{str(test_data_folder)}/hllhc15_thick/opt_round_150_1500.madx";

    seqedit,sequence=lhcb1;flatten;cycle,start=IP3;flatten;endedit;
    seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;

    exec,mk_beam(7000);
    call,file="{str(test_data_folder)}/hllhc15_thick/opt_round_150_1500.madx";
    exec,check_ip(b1);
    exec,check_ip(b2);
    """)

    mad.input('''
    on_crab1 = -190;
    on_crab5 = -190;
    on_x1 = 0;
    on_x5 = 0;
    on_disp = 0;
    vrf400 = 0;
    lagrf400.b1' = 0.5;
    lagrf400.b2' = 0.;

    use, sequence=lhcb1;
    use, sequence=lhcb2;

    save, sequence=lhcb1,lhcb2, file="saved_b1b2.madx";

    ''')

    mad.use('lhcb1')
    tw_b1 = xt.Table(mad.twiss(), _copy_cols=True)
    tw_b1_t_plus = xt.Table(mad.twiss(t=1e-4), _copy_cols=True)
    tw_b1t_minus = xt.Table(mad.twiss(t=-1e-4), _copy_cols=True)

    tw_b1['dx_dz'] = (tw_b1_t_plus['x'] - tw_b1t_minus['x'])/(tw_b1_t_plus['t'] - tw_b1t_minus['t'])
    tw_b1['dy_dz'] = (tw_b1_t_plus['y'] - tw_b1t_minus['y'])/(tw_b1_t_plus['t'] - tw_b1t_minus['t'])

    mad.use('lhcb2')
    tw_b2 = xt.Table(mad.twiss(), _copy_cols=True)
    tw_b2_t_plus = xt.Table(mad.twiss(t=1e-4), _copy_cols=True)
    tw_b2t_minus = xt.Table(mad.twiss(t=-1e-4), _copy_cols=True)

    tw_b2['dx_dz'] = (tw_b2_t_plus['x'] - tw_b2t_minus['x'])/(tw_b2_t_plus['t'] - tw_b2t_minus['t'])
    tw_b2['dy_dz'] = (tw_b2_t_plus['y'] - tw_b2t_minus['y'])/(tw_b2_t_plus['t'] - tw_b2t_minus['t'])

    print('IP1 - MADX - has known issues on B2!!!')
    print(f'B1 - dx_dz: {tw_b1["dx_dz", "ip1:1"]}')
    print(f'B2 - dx_dz: {tw_b2["dx_dz", "ip1:1"]}')
    print('IP5 - MADX - has known issues on B2!!!')
    print(f'B1 - dy_dz: {tw_b1["dy_dz", "ip5:1"]}')
    print(f'B2 - dy_dz: {tw_b2["dy_dz", "ip5:1"]}')

    line_b1 = xt.Line.from_madx_sequence(mad.sequence['lhcb1'], deferred_expressions=True)
    line_b4 = xt.Line.from_madx_sequence(mad.sequence['lhcb2'], deferred_expressions=True)

    line_b1.particle_ref = xt.Particles(p0c=7e12)
    line_b4.particle_ref = xt.Particles(p0c=7e12)

    tw_xs_b1 = line_b1.twiss4d()
    tw_xs_b4 = line_b4.twiss4d()
    tw_xs_b2 = tw_xs_b4.reverse()

    print('IP1 - Xsuite')
    print(f'B1 - dx_zeta: {tw_xs_b1["dx_zeta", "ip1"]}')
    print(f'B2 - dx_zeta: {tw_xs_b2["dx_zeta", "ip1"]}')
    print(f'B4 - dx_zeta: {tw_xs_b4["dx_zeta", "ip1"]}')

    print('IP5 - Xsuite')
    print(f'B1 - dy_zeta: {tw_xs_b1["dy_zeta", "ip5"]}')
    print(f'B2 - dy_zeta: {tw_xs_b2["dy_zeta", "ip5"]}')
    print(f'B4 - dy_zeta: {tw_xs_b4["dy_zeta", "ip5"]}')

    mad_b4 = Madx()
    mad_b4.input(f"""
    mylhcbeam=4;
    call,file="{str(test_data_folder)}/hllhc15_thick/lhcb4.seq";
    call,file="{str(test_data_folder)}/hllhc15_thick/hllhc_sequence.madx";
    call,file="{str(test_data_folder)}/hllhc15_thick/macro.madx";
    call,file="{str(test_data_folder)}/hllhc15_thick/enable_crabcavities.madx";
    call,file="{str(test_data_folder)}/hllhc15_thick/opt_round_150_1500.madx";

    seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;

    exec,mk_beam(7000);
    call,file="{str(test_data_folder)}/hllhc15_thick/opt_round_150_1500.madx";
    exec,check_ip(b2);
    """)

    mad_b4.input('''
    on_crab1 = -190;
    on_crab5 = -190;
    on_x1 = 0;
    on_x5 = 0;
    on_disp = 0;
    vrf400 = 0;
    lagrf400.b1' = 0.5;
    lagrf400.b2' = 0.;

    use, sequence=lhcb2;

    save, sequence=lhcb2, file="saved_b4.madx";

    ''')

    line_b4_mad = xt.Line.from_madx_sequence(mad_b4.sequence['lhcb2'], deferred_expressions=True)
    line_b4_mad.particle_ref = xt.Particles(p0c=7e12)

    tw_b4_mad = line_b4_mad.twiss4d()
    tw_b2_mad = tw_b4_mad.reverse()

    env_b1b2 = xt.load("saved_b1b2.madx")
    env_b1b2.lhcb1.particle_ref = xt.Particles(p0c=7e12)
    env_b1b2.lhcb2.particle_ref = xt.Particles(p0c=7e12)
    tw_b1_b1b2 = env_b1b2.lhcb1.twiss4d()
    tw_b2_b1b2 = env_b1b2.lhcb2.twiss4d()
    tw_b4_b1b2 = tw_b2_b1b2.reverse()

    env_b4 = xt.load("saved_b4.madx")
    env_b4.lhcb2.particle_ref = xt.Particles(p0c=7e12)
    tw_b4_b4 = env_b4.lhcb2.twiss4d()
    tw_b2_b4 = tw_b4_b4.reverse()

    xo.assert_allclose(tw_xs_b1["dx_zeta", "ip1"], -190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b1_b1b2["dx_zeta", "ip1"], -190e-6, rtol=3e-3)

    xo.assert_allclose(tw_xs_b2["dx_zeta", "ip1"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b2_mad["dx_zeta", "ip1"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b2_b1b2["dx_zeta", "ip1"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b2_b4["dx_zeta", "ip1"], 190e-6, rtol=3e-3)

    xo.assert_allclose(tw_xs_b4["dx_zeta", "ip1"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b4_mad["dx_zeta", "ip1"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b4_b1b2["dx_zeta", "ip1"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b4_b4["dx_zeta", "ip1"], 190e-6, rtol=3e-3)

    xo.assert_allclose(tw_xs_b1["dy_zeta", "ip5"], -190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b1_b1b2["dy_zeta", "ip5"], -190e-6, rtol=3e-3)

    xo.assert_allclose(tw_b2_mad["dy_zeta", "ip5"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b2_b1b2["dy_zeta", "ip5"], 190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b2_b4["dy_zeta", "ip5"], 190e-6, rtol=3e-3)

    xo.assert_allclose(tw_xs_b4["dy_zeta", "ip5"], -190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b4_mad["dy_zeta", "ip5"], -190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b4_b1b2["dy_zeta", "ip5"], -190e-6, rtol=3e-3)
    xo.assert_allclose(tw_b4_b4["dy_zeta", "ip5"], -190e-6, rtol=3e-3)
