import xobjects as xo

from .dress_element import dress_element


class LimitRect(xo.Struct):
    min_x = xo.Float64
    max_x = xo.Float64
    min_y = xo.Float64
    max_y = xo.Float64
