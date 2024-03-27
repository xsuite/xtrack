import xtrack as xt
import xobjects as xo
from pathlib import Path
import numpy as np

from xtrack.general import _pkg_root
from xtrack.random import RandomUniform, RandomExponential
from xtrack.beam_elements.elements import SynchrotronRadiationRecord

xo.context_default.kernels.clear()

class QuadrupoleThinSlice(xt.BeamElement):
    allow_rot_and_shift = False
    _xofields = {
        'length': xo.Float64,
        'parent': xo.Ref(xt.Quadrupole),
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
        'weight': xo.Float64,
    }

    _depends_on = [RandomUniform, RandomExponential]


    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
        Path('./quad_thin_slice.h'),
    ]

    _internal_record_class = SynchrotronRadiationRecord

    has_backtrack = True


quad = xt.Quadrupole(k1=0.1, length=1)
# quad.rot_s = 20.
# quad.shift_x = 0.1
# quad.shift_y = 0.2


quad_slice = QuadrupoleThinSlice(length=quad.length/2, weight=0.5, parent=quad, _buffer=quad._buffer)
quad_mult = xt.Multipole(knl=[0, 0.1/2], length=1./2)

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

quad_mult.track(p_ref)
quad_slice.track(p_slice)