import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment(
    particle_ref=xt.Particles(p0c=7000e9, x=1e-3, px=1e-3, y=1e-3, py=1e-3))

line0 = env.new_line(
    components=[
        env.new('b0', 'Bend', length=1.0, anchor='start', at=5.0),
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
        env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
        env.new('mk1', 'Marker', at=40),
        env.new('mk2', 'Marker'),
        env.new('mk3', 'Marker'),
        env.place('q0'),
        env.place('b0'),
        env.new('end', 'Marker', at=50.),
    ])
tt0 = line0.get_table()
tt0.show(cols=['name', 's_start', 's_end', 's_center'])

line = line0.copy()
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Teapot(2), name=r'q0.*'),
        xt.Strategy(slicing=xt.Teapot(3), name=r'b0.*'),
    ]
)
tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0_entry::0', 'b0..entry_map', 'drift_b0..0', 'b0..0',
       'drift_b0..1', 'b0..1', 'drift_b0..2', 'b0..2', 'drift_b0..3',
       'b0..exit_map', 'b0_exit::0', 'drift_2', 'ql', 'drift_3',
       'q0_entry::0', 'drift_q0..0', 'q0..0', 'drift_q0..1', 'q0..1',
       'drift_q0..2', 'q0_exit::0', 'drift_4', 'qr', 'drift_5', 'mk1',
       'mk2', 'mk3', 'q0_entry::1', 'drift_q0..3', 'q0..2', 'drift_q0..4',
       'q0..3', 'drift_q0..5', 'q0_exit::1', 'b0_entry::1',
       'b0..entry_map_0', 'drift_b0..4', 'b0..3', 'drift_b0..5', 'b0..4',
       'drift_b0..6', 'b0..5', 'drift_b0..7', 'b0..exit_map_0',
       'b0_exit::1', 'drift_6', 'end', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2.5       ,  5.        ,  5.        ,  5.0625    ,  5.125     ,
        5.3125    ,  5.5       ,  5.6875    ,  5.875     ,  5.9375    ,
        6.        ,  6.        ,  7.5       , 10.        , 15.        ,
       19.        , 19.16666667, 19.33333333, 20.        , 20.66666667,
       20.83333333, 21.        , 25.        , 30.        , 35.5       ,
       40.        , 40.        , 40.        , 40.        , 40.16666667,
       40.33333333, 41.        , 41.66666667, 41.83333333, 42.        ,
       42.        , 42.        , 42.0625    , 42.125     , 42.3125    ,
       42.5       , 42.6875    , 42.875     , 42.9375    , 43.        ,
       43.        , 46.5       , 50.        , 50.        ]),
    rtol=0., atol=1e-8)

line = line0.copy()
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Teapot(2), name=r'q0::0'),
        xt.Strategy(slicing=xt.Teapot(3), name=r'b0::1'),
    ]
)
tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0', 'drift_2', 'ql', 'drift_3', 'q0_entry',
       'drift_q0..0', 'q0..0', 'drift_q0..1', 'q0..1', 'drift_q0..2',
       'q0_exit', 'drift_4', 'qr', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0',
       'b0_entry', 'b0..entry_map', 'drift_b0..0', 'b0..0', 'drift_b0..1',
       'b0..1', 'drift_b0..2', 'b0..2', 'drift_b0..3', 'b0..exit_map',
       'b0_exit', 'drift_6', 'end', '_end_point']))

xo.assert_allclose(tt.s_center, np.array(
    [ 2.5       ,  5.5       ,  7.5       , 10.        , 15.        ,
       19.        , 19.16666667, 19.33333333, 20.        , 20.66666667,
       20.83333333, 21.        , 25.        , 30.        , 35.5       ,
       40.        , 40.        , 40.        , 41.        , 42.        ,
       42.        , 42.0625    , 42.125     , 42.3125    , 42.5       ,
       42.6875    , 42.875     , 42.9375    , 43.        , 43.        ,
       46.5       , 50.        , 50.        ]),
    rtol=0., atol=1e-8)

line = line0.copy()
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name=r'q0.*'),
        xt.Strategy(slicing=xt.Teapot(3, mode='thick'), name=r'b0.*'),
    ],
)

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0_entry::0', 'b0..entry_map', 'b0..0', 'b0..1',
       'b0..2', 'b0..exit_map', 'b0_exit::0', 'drift_2', 'ql', 'drift_3',
       'q0_entry::0', 'q0..0', 'q0..1', 'q0_exit::0', 'drift_4', 'qr',
       'drift_5', 'mk1', 'mk2', 'mk3', 'q0_entry::1', 'q0..2', 'q0..3',
       'q0_exit::1', 'b0_entry::1', 'b0..entry_map_0', 'b0..3', 'b0..4',
       'b0..5', 'b0..exit_map_0', 'b0_exit::1', 'drift_6', 'end',
       '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    [ 2.5       ,  5.        ,  5.        ,  5.08333333,  5.5       ,
        5.91666667,  6.        ,  6.        ,  7.5       , 10.        ,
       15.        , 19.        , 19.5       , 20.5       , 21.        ,
       25.        , 30.        , 35.5       , 40.        , 40.        ,
       40.        , 40.        , 40.5       , 41.5       , 42.        ,
       42.        , 42.        , 42.08333333, 42.5       , 42.91666667,
       43.        , 43.        , 46.5       , 50.        , 50.        ]),
    rtol=0., atol=1e-8)

line = line0.copy()
line.cut_at_s([20.1, 20.2, 41.7, 41.8, 5.5])

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'b0_entry', 'b0..entry_map', 'b0..0', 'b0..1',
       'b0..exit_map', 'b0_exit', 'drift_2', 'ql', 'drift_3',
       'q0_entry::0', 'q0..0', 'q0..1', 'q0..2', 'q0_exit::0', 'drift_4',
       'qr', 'drift_5', 'mk1', 'mk2', 'mk3', 'q0_entry::1', 'q0..3',
       'q0..4', 'q0..5', 'q0_exit::1', 'b0', 'drift_6', 'end',
       '_end_point']))

xo.assert_allclose(tt.s_center, np.array(
    [ 2.5 ,  5.  ,  5.  ,  5.25,  5.75,  6.  ,  6.  ,  7.5 , 10.  ,
       15.  , 19.  , 19.55, 20.15, 20.6 , 21.  , 25.  , 30.  , 35.5 ,
       40.  , 40.  , 40.  , 40.  , 40.85, 41.75, 41.9 , 42.  , 42.5 ,
       46.5 , 50.  , 50.  ]),
    rtol=0., atol=1e-8)
