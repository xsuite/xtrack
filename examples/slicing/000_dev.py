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