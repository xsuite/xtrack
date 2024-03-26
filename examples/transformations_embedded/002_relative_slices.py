import xtrack as xt
import xobjects as xo
from pathlib import Path
import numpy as np

from xtrack.general import _pkg_root

xo.context_default.kernels.clear()

class QuadrupoleThickSlice(xt.BeamElement):
    allow_rot_and_shift = False
    _xofields = {
        'length': xo.Float64,
        'parent': xo.Ref(xt.Quadrupole),
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_srotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_quadrupole.h'),
        Path('./quad_thick_slice.h'),
    ]

quad = xt.Quadrupole(k1=0.1, length=1)
quad.rot_s = 20.
quad.shift_x = 0.1
quad.shift_y = 0.2


quad_slice = QuadrupoleThickSlice(length=quad.length/2, parent=quad, _buffer=quad._buffer)

p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p_ref = p0.copy()
p_slice = p0.copy()

quad_slice.track(p_slice)
quad_slice.track(p_slice)

quad.track(p_ref)

assert_allclose = np.testing.assert_allclose
assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-14)
assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-14)
assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-14)
assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-14)
assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-14)
assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-14)