import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('el', xt.Bend, length=4, k1=0.5, h=0.1, k0=0.8, knl=[0, 0, 0.03]),
])

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(4, mode='thick'), element_type=xt.Bend),
    ])

tt_first_slice = line.get_table(attr=True)

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(4, mode='thin'), element_type=xt.ThickSliceBend),
    ])

tt_second_slice = line.get_table(attr=True)

assert np.all(tt_first_slice.name == np.array(
    ['el_entry', 'el..entry_map', 'el..0', 'el..1', 'el..2', 'el..3',
       'el..exit_map', 'el_exit', '_end_point']))

assert np.all(tt_first_slice.element_type == np.array(
    ['Marker', 'ThinSliceBendEntry', 'ThickSliceBend', 'ThickSliceBend',
       'ThickSliceBend', 'ThickSliceBend', 'ThinSliceBendExit', 'Marker',
       '']))

# Check first table

xo.assert_allclose(tt_first_slice.angle_rad, np.array(
    [0.  , 0.  , 0.05, 0.15, 0.15, 0.05, 0.  , 0.  , 0.  ]), rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_first_slice.angle_rad), 0.4, rtol=0, atol=1e-12)

xo.assert_allclose(tt_first_slice.k0l, np.array(
    [0. , 0. , 0.4, 1.2, 1.2, 0.4, 0. , 0. , 0. ]), rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_first_slice.k0l), 3.2, rtol=0, atol=1e-12)

xo.assert_allclose(tt_first_slice.k1l, np.array(
    [0.  , 0.  , 0.25, 0.75, 0.75, 0.25, 0.  , 0.  , 0.  ]), rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_first_slice.k1l), 2, rtol=0, atol=1e-12)

xo.assert_allclose(tt_first_slice.k2l, np.array(
    [0.     , 0.     , 0.00375, 0.01125, 0.01125, 0.00375, 0. , 0., 0.]),
    rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_first_slice.k2l), 0.03, rtol=0, atol=1e-12)

# Check second table

assert np.all(tt_second_slice.name == np.array(
    ['el_entry', 'el..entry_map', 'el..0_entry', 'drift_el..0..0',
       'el..0..0', 'drift_el..0..1', 'el..0..1', 'drift_el..0..2',
       'el..0..2', 'drift_el..0..3', 'el..0..3', 'drift_el..0..4',
       'el..0_exit', 'el..1_entry', 'drift_el..1..0', 'el..1..0',
       'drift_el..1..1', 'el..1..1', 'drift_el..1..2', 'el..1..2',
       'drift_el..1..3', 'el..1..3', 'drift_el..1..4', 'el..1_exit',
       'el..2_entry', 'drift_el..2..0', 'el..2..0', 'drift_el..2..1',
       'el..2..1', 'drift_el..2..2', 'el..2..2', 'drift_el..2..3',
       'el..2..3', 'drift_el..2..4', 'el..2_exit', 'el..3_entry',
       'drift_el..3..0', 'el..3..0', 'drift_el..3..1', 'el..3..1',
       'drift_el..3..2', 'el..3..2', 'drift_el..3..3', 'el..3..3',
       'drift_el..3..4', 'el..3_exit', 'el..exit_map', 'el_exit',
       '_end_point']))

assert np.all(tt_second_slice.element_type == np.array(
    ['Marker', 'ThinSliceBendEntry', 'Marker', 'DriftSliceBend',
       'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
       'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
       'ThinSliceBend', 'DriftSliceBend', 'Marker', 'Marker',
       'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
       'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
       'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend', 'Marker',
       'Marker', 'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
       'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
       'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend', 'Marker',
       'Marker', 'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
       'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBend',
       'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend', 'Marker',
       'ThinSliceBendExit', 'Marker', ''
    ]))

xo.assert_allclose(tt_second_slice.angle_rad, np.array(
    [0.    , 0.    , 0.    , 0.    , 0.0125, 0.    , 0.0125, 0.    ,
        0.0125, 0.    , 0.0125, 0.    , 0.    , 0.    , 0.    , 0.0375,
        0.    , 0.0375, 0.    , 0.0375, 0.    , 0.0375, 0.    , 0.    ,
        0.    , 0.    , 0.0375, 0.    , 0.0375, 0.    , 0.0375, 0.    ,
        0.0375, 0.    , 0.    , 0.    , 0.    , 0.0125, 0.    , 0.0125,
        0.    , 0.0125, 0.    , 0.0125, 0.    , 0.    , 0.    , 0.    ,
        0.    ]), rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_second_slice.angle_rad), 0.4, rtol=0, atol=1e-12)

xo.assert_allclose(tt_second_slice.k0l, np.array(
    [0. , 0. , 0. , 0. , 0.1, 0. , 0.1, 0. , 0.1, 0. , 0.1, 0. , 0. ,
       0. , 0. , 0.3, 0. , 0.3, 0. , 0.3, 0. , 0.3, 0. , 0. , 0. , 0. ,
       0.3, 0. , 0.3, 0. , 0.3, 0. , 0.3, 0. , 0. , 0. , 0. , 0.1, 0. ,
       0.1, 0. , 0.1, 0. , 0.1, 0. , 0. , 0. , 0. , 0. ]), rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_second_slice.k0l), 3.2, rtol=0, atol=1e-12)

xo.assert_allclose(tt_second_slice.k1l, np.array(
    [0.    , 0.    , 0.    , 0.    , 0.0625, 0.    , 0.0625, 0.    ,
       0.0625, 0.    , 0.0625, 0.    , 0.    , 0.    , 0.    , 0.1875,
       0.    , 0.1875, 0.    , 0.1875, 0.    , 0.1875, 0.    , 0.    ,
       0.    , 0.    , 0.1875, 0.    , 0.1875, 0.    , 0.1875, 0.    ,
       0.1875, 0.    , 0.    , 0.    , 0.    , 0.0625, 0.    , 0.0625,
       0.    , 0.0625, 0.    , 0.0625, 0.    , 0.    , 0.    , 0.    ,
       0.    ]), rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_second_slice.k1l), 2, rtol=0, atol=1e-12)

xo.assert_allclose(tt_second_slice.k2l, np.array(
    [0.       , 0.       , 0.       , 0.       , 0.0009375, 0.       ,
       0.0009375, 0.       , 0.0009375, 0.       , 0.0009375, 0.       ,
       0.       , 0.       , 0.       , 0.0028125, 0.       , 0.0028125,
       0.       , 0.0028125, 0.       , 0.0028125, 0.       , 0.       ,
       0.       , 0.       , 0.0028125, 0.       , 0.0028125, 0.       ,
       0.0028125, 0.       , 0.0028125, 0.       , 0.       , 0.       ,
       0.       , 0.0009375, 0.       , 0.0009375, 0.       , 0.0009375,
       0.       , 0.0009375, 0.       , 0.       , 0.       , 0.       ,
       0.       ]),
    rtol=0, atol=1e-12)
xo.assert_allclose(np.sum(tt_second_slice.k2l), 0.03, rtol=0, atol=1e-12)