import xtrack as xt
import xobjects as xo
from pathlib import Path
import numpy as np


quad = xt.Quadrupole(k1=0.1, length=1)
# quad.rot_s = 20.
# quad.shift_x = 0.1
# quad.shift_y = 0.2


quad_slice = xt.ThinSliceQuadrupole(weight=0.5, _parent=quad, _buffer=quad._buffer)
quad_mult = xt.Multipole(knl=[0, 0.1/2], length=1./2)

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

quad_mult.track(p_ref)
quad_slice.track(p_slice)

line = xt.Line(elements=[quad])
line.build_tracker() # Put everything in the same buffer
line.discard_tracker()

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(1000))])
line.build_tracker()
assert line['e0..995']._parent_name == 'e0'
assert line['e0..995']._parent is line['e0']

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

line.track(p_slice)
quad.track(p_ref)

line.to_json('ttt.json')
line2 = xt.Line.from_json('ttt.json')
assert isinstance(line2['e0..995'], xt.ThinSliceQuadrupole)
assert line2['e0..995']._parent_name == 'e0'
assert line2['e0..995']._parent is None