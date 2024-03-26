import xtrack as xt
import xobjects as xo

class QuadThickSlice(xt.BeamElement):
    _xofields = {
        'parent': xo.Ref(xt.Quadrupole),
    }

quad = xt.Quadrupole(k1=0.1, length=1)
quad_slice = QuadThickSlice(parent=quad)