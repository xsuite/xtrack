import xobjects as xo

from ..dress_element import dress_element
from ..general import _pkg_root

class LimitRectData(xo.Struct):
    min_x = xo.Float64
    max_x = xo.Float64
    min_y = xo.Float64
    max_y = xo.Float64
LimitRectData.extra_sources = [
        _pkg_root.joinpath('beam_elements/apertures_src/limitrect.h')]

class LimitRect(dress_element(LimitRectData)):
    pass
