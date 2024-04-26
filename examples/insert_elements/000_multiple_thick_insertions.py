import xtrack as xt
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