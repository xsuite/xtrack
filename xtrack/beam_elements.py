import xobjects as xo

class Multipole(xo.Struct):
    order = xo.Int64
    length = xo.Float64
    hxl = xo.Float64
    hyl = xo.Float64
    bal = xo.Float64[:]

class Drift(xo.Struct):
    length = xo.Float64

class Cavity(xo.Struct):
    voltage = xo.Float64
    frequency = xo.Float64
    lag = xo.Float64

class XYShift(xo.Struct):
    dx = xo.Float64
    dy = xo.Float64

class SRotation(xo.Struct):
    cos_z = xo.Float64
    sin_z = xo.Float64
