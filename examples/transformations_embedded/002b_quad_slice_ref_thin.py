import xtrack as xt
import xobjects as xo
from pathlib import Path
import numpy as np


quad = xt.Quadrupole(k1=0.1, length=1)
# quad.rot_s = 20.
# quad.shift_x = 0.1
# quad.shift_y = 0.2


quad_slice = xt.ThinSliceQuadrupole(weight=0.5, parent=quad, _buffer=quad._buffer)
quad_mult = xt.Multipole(knl=[0, 0.1/2], length=1./2)

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

quad_mult.track(p_ref)
quad_slice.track(p_slice)

line = xt.Line(elements=[quad])
line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(5))])