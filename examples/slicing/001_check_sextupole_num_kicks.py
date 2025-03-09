import xtrack as xt

line = xt.Line(elements={
    's1': xt.Sextupole(k2=0.1, length=10.0)
})

line_1slice = line.copy(shallow=True)
line_1slice.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(1), element_type=xt.Sextupole),
    ])

line_3slices = line.copy(shallow=True)
line_3slices.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Sextupole),
    ])

p = xt.Particles(p0c=7000e9, x=1e-2)
p_1slice = p.copy()
p_3slices = p.copy()

line_1slice.track(p_1slice)
line_3slices.track(p_3slices)

