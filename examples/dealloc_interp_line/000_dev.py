import xtrack as xt
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('q', xt.Quadrupole, k1=0.1, length=1.0)
])
line.build_tracker()

old_elements = set(line.env.elements)
line_sliced = line.copy(shallow=True)
line_sliced.cut_at_s(np.linspace(0, line.get_length(), 11))

new_elements = set(line_sliced.env.elements) - old_elements
del line_sliced
for nn in new_elements:
    sz = line.env.element_dict[nn]._xobject._size
    oo = line.env.element_dict[nn]._xobject._offset
    line._buffer.free(oo, sz)
    print(f'Deallocated {nn} at offset {oo} of size {sz}')
    print(line._buffer)
