import xtrack as xt
import xobjects as xo
assert_allclose = xo.assert_allclose

slice_mode= 'thin'

elements = {
    'e0': xt.Bend(k0=0.3, h=0.31, length=1),
    'e1': xt.Replica(_parent_name='e0'),
    'e2': xt.Bend(k0=-0.4, h=-0.41, length=1),
    'e3': xt.Replica(_parent_name='e2'),
    'e4': xt.Replica(_parent_name='e3'), # Replica of replica
}
line = xt.Line(elements=elements,
               element_names=list(elements.keys()))
line.build_tracker()

element_no_repl={
    'e0': xt.Bend(k0=0.3, h=0.31, length=1),
    'e1': xt.Bend(k0=0.3, h=0.31, length=1),
    'e2': xt.Bend(k0=-0.4, h=-0.41, length=1),
    'e3': xt.Bend(k0=-0.4, h=-0.41, length=1),
    'e4': xt.Bend(k0=-0.4, h=-0.41, length=1),
}

line_no_repl = xt.Line(elements=element_no_repl,
                       element_names=list(element_no_repl.keys()))
line_no_repl.build_tracker()

p0 = xt.Particles(p0c=1e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p1 = p0.copy()
p2 = p0.copy()

line.track(p1)
line_no_repl.track(p2)

assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(xt.Teapot(2, mode=slice_mode), name='e2|e3|e4')])
line.build_tracker()

line_no_repl.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(xt.Teapot(2, mode=slice_mode), name='e2|e3|e4')])
line_no_repl.build_tracker()

p1 = p0.copy()
p2 = p0.copy()

line.track(p1)
line_no_repl.track(p2)

assert_allclose(p2.x, p1.x, rtol=0, atol=1e-14)
assert_allclose(p2.px, p1.px, rtol=0, atol=1e-14)
assert_allclose(p2.y, p1.y, rtol=0, atol=1e-14)
assert_allclose(p2.py, p1.py, rtol=0, atol=1e-14)
assert_allclose(p2.zeta, p1.zeta, rtol=0, atol=1e-14)
assert_allclose(p2.delta, p1.delta, rtol=0, atol=1e-14)
