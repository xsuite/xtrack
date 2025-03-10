import xtrack as xt
import xobjects as xo

from ..base_element import BeamElement
from ..general import _pkg_root

class MagnetDrift(BeamElement):
    isthick = True
    has_backtrack = True

    _xofields = {
        'length': xo.Float64,
        'k0': xo.Float64,
        'k1': xo.Float64,
        'h': xo.Float64,
        'drift_model': xo.Int64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/magnet_drift.h'),
    ]


class MagnetKick(BeamElement):
    isthick = False
    has_backtrack = False

    _xofields = {
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'factor_knl_ksl': xo.Float64,
        'kick_weight': xo.Float64,
        'k0': xo.Float64,
        'k1': xo.Float64,
        'k2': xo.Float64,
        'k3': xo.Float64,
        'k0s': xo.Float64,
        'k1s': xo.Float64,
        'k2s': xo.Float64,
        'k3s': xo.Float64,
        'h': xo.Float64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_kick.h'),
        _pkg_root.joinpath('beam_elements/elements_src/magnet_kick.h'),
    ]
