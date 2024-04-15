import numpy as np
import xtrack as xt
import xobjects as xo

xo.context_default.kernels.clear()

model = 'full'

# only edge entry
bend_only_e1 = xt.Bend(
    length=0, k0=0.1,
    edge_entry_angle=0.05,
    edge_entry_hgap=0.06,
    edge_entry_fint=0.08,
    edge_exit_active=False)

edge_e1 = xt.DipoleEdge(
    k=0.1, side='entry', e1=0.05, hgap=0.06, fint=0.08)

line = xt.Line(elements=[bend_only_e1])

line['e0'].length = 1 # to force the slicing
line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(1))])
line['e0'].length = 0
line.build_tracker()

assert 'e0..entry_map' in line.element_names
assert 'e0..exit_map' in line.element_names
assert isinstance(line['e0..entry_map'], xt.ThinSliceBendEntry)
assert isinstance(line['e0..exit_map'], xt.ThinSliceBendExit)

edge_e1.model = model
bend_only_e1.edge_entry_model = model

p1 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p2 = p1.copy()
p3 = p1.copy()

bend_only_e1.track(p1)
edge_e1.track(p2)
line.track(p3)

assert_allclose = np.testing.assert_allclose
assert_allclose(p1.x, p2.x, rtol=0, atol=1e-14)
assert_allclose(p1.px, p2.px, rtol=0, atol=1e-14)
assert_allclose(p1.y, p2.y, rtol=0, atol=1e-14)
assert_allclose(p1.py, p2.py, rtol=0, atol=1e-14)
assert_allclose(p1.zeta, p2.zeta, rtol=0, atol=1e-14)
assert_allclose(p1.delta, p2.delta, rtol=0, atol=1e-14)

assert_allclose(p1.x, p3.x, rtol=0, atol=1e-14)
assert_allclose(p1.px, p3.px, rtol=0, atol=1e-14)
assert_allclose(p1.y, p3.y, rtol=0, atol=1e-14)
assert_allclose(p1.py, p3.py, rtol=0, atol=1e-14)
assert_allclose(p1.zeta, p3.zeta, rtol=0, atol=1e-14)
assert_allclose(p1.delta, p3.delta, rtol=0, atol=1e-14)
