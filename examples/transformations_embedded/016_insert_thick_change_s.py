import xtrack as xt
import xobjects as xo
import numpy as np

assert_allclose = xo.assert_allclose

elements = {
    'd1': xt.Drift(length=1),
    'm1': xt.Marker(),
    'd2': xt.Drift(length=1),
}

line=xt.Line(elements=elements,
             element_names=list(elements.keys()))

# Note that the name is reused
line.insert_element(element=xt.Bend(length=1.), name='m1', at_s=0.5)

tt = line.get_table()

assert np.all(tt.name == ['d1..0', 'm1', 'd2..1', '_end_point'])
assert np.all(tt.parent_name == ['d1', None, 'd2', None])
assert_allclose(tt.s, [0. , 0.5, 1.5, 2. ], rtol=0, atol=1e-14)