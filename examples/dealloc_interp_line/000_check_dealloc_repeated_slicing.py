import xtrack as xt
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('q', xt.Quadrupole, k1=0.1, length=1.0)
])
line.build_tracker()
print('After build tracker')
print(line._buffer)

n_repetitions = 1000
for _ in range(n_repetitions):
    old_elements = set(line.env.elements)
    line_sliced = line.copy(shallow=True)
    line_sliced.cut_at_s(np.linspace(0, line.get_length(), 11))
    print('After slicing')
    print(line._buffer)

    new_elements = set(line_sliced.env.elements) - old_elements
    del line_sliced
    for nn in new_elements:
        sz = line.env.element_dict[nn]._xobject._size
        oo = line.env.element_dict[nn]._xobject._offset
        line._buffer.free(oo, sz)
        del line.env.element_dict[nn]

    print('After deallocation')
    print(line._buffer)