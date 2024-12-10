import xtrack as xt

env = xt.Environment()
line = env.new_line(components=[
    env.new('el', xt.Bend, length=2, k1=0.5),
])

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(4, mode='thick'), element_type=xt.Bend),
    ])

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(4, mode='thin'), element_type=xt.ThickSliceBend),
    ])