import xtrack as xt
import xobjects as xo
import numpy as np

line = xt.Line(
    elements=[xt.Drift(length=1.0) for i in range(10)]
)

s_insert = np.array([2.5, 5.5, 7])
l_insert = np.array([1.0, 1.0, 1.0])
ele_insert = [xt.Sextupole(length=l) for l in l_insert]

line._insert_thick_elements_at_s(
    element_names=[f'insertion_{i}' for i in range(len(s_insert))],
    elements=ele_insert,
    at_s=s_insert
)

tt = line.get_table()

assert np.all(tt.name == ['e0', 'e1', 'e2..0', 'insertion_0', 'e3..1', 'e4', 'e5..0',
       'insertion_1', 'e6..1', 'insertion_2', 'e8', 'e9', '_end_point'])
xo.assert_allclose(tt.s, [ 0. ,  1. ,  2. ,  2.5,  3.5,  4. ,  5. ,
                                 5.5,  6.5,  7. ,  8. , 9. , 10. ])

assert np.all(tt.element_type == ['Drift', 'Drift', 'DriftSlice', 'Sextupole', 'DriftSlice', 'Drift',
       'DriftSlice', 'Sextupole', 'DriftSlice', 'Sextupole', 'Drift',
       'Drift', ''])