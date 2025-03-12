import xtrack as xt
import xobjects as xo

from xtrack.base_element import BeamElement
from xtrack.beam_elements.elements import DEFAULT_MULTIPOLE_ORDER, _prepare_multipolar_params, SynchrotronRadiationRecord
from xtrack.general import _pkg_root

from ..random import RandomUniformAccurate, RandomExponential, RandomNormal
from ..internal_record import RecordIndex

from typing import List


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
        'angle': xo.Float64,
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


class Magnet(BeamElement):
    isthick = True
    has_backtrack = True

    _xofields = {
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'num_multipole_kicks': xo.Int64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'k0': xo.Float64,
        'k1': xo.Float64,
        'k2': xo.Float64,
        'k3': xo.Float64,
        'k0s': xo.Float64,
        'k1s': xo.Float64,
        'k2s': xo.Float64,
        'k3s': xo.Float64,
        'angle': xo.Float64,
        'h': xo.Float64,
        'k0_from_h': xo.UInt64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _rename = {
        'order': '_order',
        'model': '_model',
        'k0': '_k0',
        'k0_from_h': '_k0_from_h',
        'angle': '_angle',
        'length': '_length',
        'h': '_h',
        'integrator': '_integrator',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_kick.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_radiation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet.h'),
        _pkg_root.joinpath('beam_elements/elements_src/magnet.h'),
    ]


    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    _INDEX_TO_MODEL = {
        0: 'adaptive',
        1: 'full',
        2: 'bend-kick-bend',
        3: 'rot-kick-rot',
        4: 'mat-kick-mat',
        5: 'drift-kick-drift-exact',
        6: 'drift-kick-drift-expanded',
    }
    _MODEL_TO_INDEX = {k: v for v, k in _INDEX_TO_MODEL.items()} | {'expanded': 4}

    _INDEX_TO_INTEGRATOR = {
        0: 'adaptive',
        1: 'teapot',
        2: 'yoshida4',
    }
    _INTEGRATOR_TO_INDEX = {k: v for v, k in _INDEX_TO_INTEGRATOR.items()}


    def __init__(self, order=None, knl: List[float]=None, ksl: List[float]=None, **kwargs):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = _prepare_multipolar_params(knl, ksl, order)
        kwargs.update(multipolar_kwargs)

        self.xoinitialize(**kwargs)

        # Calculate length and h in the event length_straight and/or angle given
        self.set_bend_params(
            kwargs.get('length'),
            kwargs.get('h'),
            kwargs.get('angle'),
        )

        if self.k0_from_h:
            self.k0 = self.h

        if model is not None:
            self.model = model

    def set_bend_params(self, length=None, h=None, angle=None):
        length, h, angle = self.compute_bend_params(
            length, h, angle,
        )

        # None becomes NaN in numpy buffers
        if length is not None:
            self._length = length
        if h is not None:
            self._h = h
        if angle is not None:
            self._angle = angle

        if self.k0_from_h:
            self._k0 = self.h

    @staticmethod
    def compute_bend_params(length=None, h=None, angle=None):
        if not length:
            # If no length, then we cannot meaningfully calculate anything
            return length, h, angle

        if angle is not None:
            computed_h = angle / length

            if h is not None and not np.isclose(h, computed_h, rtol=0, atol=1e-13):
                raise ValueError('Given `h` and `angle` are inconsistent!')

            h = h or computed_h
            return length, h, angle

        if h is not None:
            computed_angle = h * length

            if angle is not None and not np.isclose(angle, computed_angle, rtol=0, atol=1e-13):
                raise ValueError('Given `h` and `angle` are inconsistent!')

            angle = angle or computed_angle
            return length, h, angle

        # Both `h` and `angle` are None
        return length, h, angle

    @property
    def k0(self):
        return self._k0

    @k0.setter
    def k0(self, value):
        if self.k0_from_h and not np.isclose(value, self.h, atol=1e-13):
            self.k0_from_h = False
        self._k0 = value

    @property
    def k0_from_h(self):
        return bool(self._k0_from_h)

    @k0_from_h.setter
    def k0_from_h(self, value):
        if value:
            self._k0 = self.h
        self._k0_from_h = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self.set_bend_params(length=value, angle=self.angle)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        self.set_bend_params(length=self.length, h=value)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self.set_bend_params(length=self.length, angle=value)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)

    @property
    def model(self):
        return self._INDEX_TO_MODEL[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = self._MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    @property
    def integrator(self):
        return self._INDEX_TO_INTEGRATOR[self._integrator]

    @integrator.setter
    def integrator(self, value):
        try:
            self._integrator = self._INTEGRATOR_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid integrator: {value}')
