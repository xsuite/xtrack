import xtrack as xt

env = xt.Environment()
line = env.new_line(components=[
    env.new('qf', xt.Quadrupole, length=2, k1=0.5),
])

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(4, mode='thick'), element_type=xt.Quadrupole),
    ])

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(4, mode='thin'), element_type=xt.ThickSliceQuadrupole),
    ])