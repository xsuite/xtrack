import pathlib

import xtrack as xt
import numpy as np
import xobjects as xo
from cpymad.madx import Madx

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_aperture_table():

    aper_blacklist = [
        'vtaf.51632.b_aper', 'vbrta.51633.a_aper', 'vbrta.51633.b_aper',
        'bgiha.51634.a_aper', 'bgiva.51674.a_aper']

    env = xt.load_madx_lattice(test_data_folder /
                               'sps_with_apertures/EYETS 2024-2025.seq')
    env.vars.load_madx(str(test_data_folder / 'sps_with_apertures/lhc_q20.str'))
    line = env.sps
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=26e9)

    tw0 = line.twiss4d()


    mad = Madx()
    mad.input('''
    SPS : SEQUENCE, refer = centre,    L = 7000;
    a: marker, at = 20;
    endsequence;
    ''')
    mad.call(str(test_data_folder /
                 'sps_with_apertures/APERTURE_EYETS 2024-2025.seq'))
    mad.beam()
    mad.use('SPS')
    line_aper = xt.Line.from_madx_sequence(mad.sequence.SPS, install_apertures=True)

    tt_aper = line_aper.get_table().rows['.*_aper']

    insertions = []
    for nn in tt_aper.name:
        if nn in aper_blacklist:
            continue
        env.elements[nn] = line_aper.get(nn).copy()
        insertions.append(env.place(nn, at=tt_aper['s', nn]))

    for ins in insertions:
        if ins.name.endswith('.a_aper'):
            ins.at += 1e-3
        if ins.name.endswith('.b_aper'):
            ins.at -= 1e-3

    line = env.sps
    line.insert(insertions)

    line.build_tracker()
    aper = line.get_aperture_table(dx=1e-3, dy=1e-3,
                                x_range=(-0.1, 0.1), y_range=(-0.1, 0.1))


    ###########
    aper_check = aper.rows['veba.20250.a_aper' : 'vebb.20270.b_aper']

    assert np.all(aper_check.name == np.array(
        ['veba.20250.a_aper', 'drift_333..2', 'mba.20250', 'drift_334..0',
        'veba.20250.b_aper', 'drift_334..1', 'vebb.20270.a_aper',
        'drift_334..2', 'mbb.20270', 'drift_335..0', 'vebb.20270.b_aper']))
    xo.assert_allclose(aper_check.s, np.array([
        1225.8247    , 1225.8247    , 1226.02017417, 1232.28019277,
        1232.4827    , 1232.4827    , 1232.4847    , 1232.4847    ,
        1232.67019277, 1238.93021138, 1239.1227    ]), rtol=0, atol=1e-6)
    xo.assert_allclose(aper_check.x_aper_low, np.array(
        [-0.0765, -0.0765, -0.0765, -0.0765, -0.0765, -0.0765, -0.0645,
        -0.0645, -0.0645, -0.0645, -0.0645]), rtol=0, atol=1e-3)
    xo.assert_allclose(aper_check.x_aper_high, np.array(
        [0.0755, 0.0755, 0.0755, 0.0755, 0.0755, 0.0755, 0.0645, 0.0645,
        0.0645, 0.0645, 0.0645]), rtol=0, atol=1e-3)
    xo.assert_allclose(aper_check.y_aper_low, np.array(
        [-0.0195, -0.0195, -0.0195, -0.0195, -0.0195, -0.0195, -0.0265,
        -0.0265, -0.0265, -0.0265, -0.0265]), rtol=0, atol=1e-3)
    xo.assert_allclose(aper_check.y_aper_high, np.array(
        [0.0195, 0.0195, 0.0195, 0.0195, 0.0195, 0.0195, 0.0265, 0.0265,
        0.0265, 0.0265, 0.0265]), rtol=0, atol=1e-3)

    assert np.all(np.isnan(aper_check.x_aper_low_discrete) == np.array(
        [False,  True,  True,  True, False,  True, False,  True,  True,
            True, False]))
    assert np.all(np.isnan(aper_check.x_aper_high_discrete) == np.array(
        [False,  True,  True,  True, False,  True, False,  True,  True,
            True, False]))
    assert np.all(np.isnan(aper_check.y_aper_low_discrete) == np.array(
        [False,  True,  True,  True, False,  True, False,  True,  True,
            True, False]))
    assert np.all(np.isnan(aper_check.y_aper_high_discrete) == np.array(
        [False,  True,  True,  True, False,  True, False,  True,  True,
            True, False]))

    mask_not_none = ~np.isnan(aper_check.x_aper_low_discrete)
    assert np.all(aper_check.x_aper_low_discrete[mask_not_none]
                    == aper_check.x_aper_low[mask_not_none])
    assert np.all(aper_check.x_aper_high_discrete[mask_not_none]
                    == aper_check.x_aper_high[mask_not_none])
    assert np.all(aper_check.y_aper_low_discrete[mask_not_none]
                    == aper_check.y_aper_low[mask_not_none])
    assert np.all(aper_check.y_aper_high_discrete[mask_not_none]
                    == aper_check.y_aper_high[mask_not_none])

def test_aperture_table_aper_at_same_s():

    env = xt.Environment()

    line = env.new_line(components=[
        env.new('lrect', xt.LimitRect, min_x=-0.05, max_x=0.05, min_y=-0.02, max_y=0.02, at=3.),
        env.new('lellipse', xt.LimitEllipse, a=0.02, b=0.01),
        env.new('m', xt.Marker, at=5.),
        env.place('lellipse')
        ]
    )

    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=26e9)

    tt = line.get_aperture_table()
    #
    # Table: 7 rows, 10 cols
    # name                    s    x_aper_low   x_aper_high x_aper_low_discrete x_aper_high_discrete ...
    # drift_1                 0       -0.0505        0.0495                 nan                  nan
    # lrect                   3       -0.0505        0.0495             -0.0505               0.0495
    # lellipse::0             3       -0.0205        0.0195             -0.0205               0.0195
    # drift_2                 3       -0.0205        0.0195                 nan                  nan
    # m                       5       -0.0205        0.0195                 nan                  nan
    # lellipse::1             5       -0.0205        0.0195             -0.0205               0.0195
    # _end_point              5       -0.0205        0.0195                 nan                  nan

    import numpy as np
    import xobjects as xo
    assert np.all(tt.name == ['drift_1', 'lrect', 'lellipse', 'drift_2', 'm', 'lellipse',
        '_end_point'])
    xo.assert_allclose(tt.s, np.array([0., 3., 3., 3., 5., 5., 5.]), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.x_aper_low, np.array([-0.0505, -0.0505, -0.0205, -0.0205, -0.0205,
        -0.0205, -0.0205]), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.x_aper_high, np.array([0.0495, 0.0495, 0.0195, 0.0195, 0.0195,
        0.0195, 0.0195]), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.x_aper_low_discrete, np.array([np.nan, -0.0505, -0.0205, np.nan,
        np.nan, -0.0205, np.nan]), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.x_aper_high_discrete, np.array([np.nan, 0.0495, 0.0195, np.nan,
        np.nan, 0.0195, np.nan]), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.y_aper_low, np.array(
        [-0.0205, -0.0205, -0.0105, -0.0105, -0.0105, -0.0105, -0.0105],
        dtype=float), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.y_aper_high, np.array(
        [0.0195, 0.0195, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095],
        dtype=float), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.y_aper_low_discrete, np.array(
        [np.nan, -0.0205, -0.0105, np.nan, np.nan, -0.0105, np.nan],
        dtype=float), rtol=0, atol=1e-6)
    xo.assert_allclose(tt.y_aper_high_discrete, np.array(
        [np.nan, 0.0195, 0.0095, np.nan, np.nan, 0.0095, np.nan],
        dtype=float), rtol=0, atol=1e-6)