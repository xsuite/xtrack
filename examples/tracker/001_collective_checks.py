import numpy as np
import xtrack as xt

line = xt.Line(
    elements=[xt.Drift(length=1) for i in range(8)],
    element_names=[f'e{i}' for i in range(8)]
)
line['e3'].iscollective = True
e3_buffer = line['e3']._buffer
e3 = line['e3']

try:
    line.iscollective
except RuntimeError:
    pass
else:
    raise ValueError('This should have failed')

try:
    line._buffer
except RuntimeError:
    pass
else:
    raise ValueError('This should have failed')

line.build_tracker()

assert line.iscollective == True
assert line['e0']._buffer is line._buffer
assert line['e7']._buffer is line._buffer
assert line['e3']._buffer is not line._buffer
assert line['e3']._buffer is e3_buffer
assert line['e3'] is e3

nc_line = line._get_non_collective_line()

# Check that the original line is untouched
assert line.iscollective == True
assert line['e0']._buffer is line._buffer
assert line['e7']._buffer is line._buffer
assert line['e3']._buffer is not line._buffer
assert line['e3']._buffer is e3_buffer
assert line['e3'] is e3

assert nc_line.iscollective == False
assert nc_line._buffer is line._buffer
assert nc_line['e0']._buffer is line._buffer
assert nc_line['e7']._buffer is line._buffer
assert nc_line['e3']._buffer is line._buffer
assert nc_line['e3'] is not e3
assert nc_line['e0'] is line['e0']
assert nc_line['e7'] is line['e7']

assert np.allclose(nc_line.get_s_elements(), line.get_s_elements(),
                   rtol=0, atol=1e-15)

assert nc_line.tracker is not line.tracker
assert nc_line.tracker._tracker_data is line.tracker._tracker_data
assert line.tracker._track_kernel is nc_line.tracker._track_kernel