import numpy as np
from scipy.special import factorial

import xtrack as xt
import xobjects as xo
import numpy as np

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
        'edge_entry_active': xo.Field(xo.Int64, default=1),
        'edge_exit_active': xo.Field(xo.Int64, default=1),
        'edge_entry_model': xo.Int64,
        'edge_exit_model': xo.Int64,
        'edge_entry_angle': xo.Float64,
        'edge_exit_angle': xo.Float64,
        'edge_entry_angle_fdown': xo.Float64,
        'edge_exit_angle_fdown': xo.Float64,
        'edge_entry_fint': xo.Float64,
        'edge_exit_fint': xo.Float64,
        'edge_entry_hgap': xo.Float64,
        'edge_exit_hgap': xo.Float64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
    }

    _rename = {
        'order': '_order',
        'model': '_model',
        'edge_entry_model': '_edge_entry_model',
        'edge_exit_model': '_edge_exit_model',
        'k0': '_k0',
        'k0_from_h': '_k0_from_h',
        'angle': '_angle',
        'length': '_length',
        'h': '_h',
        'integrator': '_integrator',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_yrotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_wedge.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_fringe.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_linear.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_mult_fringe.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_edge.h'),
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
        3: 'uniform',
    }
    _INTEGRATOR_TO_INDEX = {k: v for v, k in _INDEX_TO_INTEGRATOR.items()}


    def __init__(self, order=None, knl: List[float]=None, ksl: List[float]=None, **kwargs):
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = _prepare_multipolar_params(order, knl=knl, ksl=ksl)
        kwargs.update(multipolar_kwargs)

        edge_entry_model = kwargs.pop('edge_entry_model', None)
        edge_exit_model = kwargs.pop('edge_exit_model', None)

        self.xoinitialize(**kwargs)

        # Calculate length and h in the event length_straight and/or angle given
        self.set_bend_params(
            kwargs.get('length'),
            kwargs.get('h'),
            kwargs.get('angle'),
        )

        if self.k0_from_h:
            self.k0 = self.h

        # Trigger properties
        if model is not None:
            self.model = model

        if edge_entry_model is not None:
            self.edge_entry_model = edge_entry_model

        if edge_exit_model is not None:
            self.edge_exit_model = edge_exit_model

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

    @property
    def edge_entry_model(self):
        return {
            0: 'linear',
            1: 'full',
           -1: 'suppressed',
        }[self._edge_entry_model]

    @edge_entry_model.setter
    def edge_entry_model(self, value):
        assert value in ['linear', 'full', 'suppressed']
        self._edge_entry_model = {
            'linear': 0,
            'full': 1,
            'suppressed': -1,
        }[value]

    @property
    def edge_exit_model(self):
        return {
            0: 'linear',
            1: 'full',
           -1: 'suppressed',
        }[self._edge_exit_model]

    @edge_exit_model.setter
    def edge_exit_model(self, value):
        assert value in ['linear', 'full', 'suppressed']
        self._edge_exit_model = {
            'linear': 0,
            'full': 1,
            'suppressed': -1,
        }[value]


class MagnetEdge(BeamElement):
    """Beam element modeling a magnet edge.

    Parameters
    ----------
    model : str
        Model to be used for the edge. It can be 'linear', 'full' or 'suppress'.
        Default is 'linear'.
        account.
    is_exit : bool
        If False, the edge is a entrance edge. If True, the edge is an exit edge.
    kn : list of floats
        List of normal multipolar strengths. If not provided, will be filled
        with zeros according to ``k_order``.
    ks : list of floats
        List of skew multipolar strengths. If not provided, will be filled
        with zeros according to ``k_order``.
    k_order : int
        Order of kn and ks. If not provided, will either be inferred from kn
        and/or ks or set to -1.
    knl : list of floats
        List of integrated normal strengths. If not provided, will be filled
        with zeros according to ``kl_order``.
    ksl : list of floats
        List of integrated skew strengths. If not provided, will be filled
        with zeros according to ``kl_order``.
    kl_order : int
        Order of knl and ksl. If not provided, will either be inferred from
        knl and/or ksl or set to -1.
    length : float
        Length of the magnet. Only necessary if integrated strengths are given.
    half_gap : float
        Equivalent gap in m.
    face_angle : float
        Face angle in rad.
    face_angle_feed_down : float
        Term added to ``face_angle`` only for the linear mode and only in the
        vertical plane to account for non-zero angle in the closed orbit when
        entering the fringe field (feed down effect).
    fringe_integral : float
        Fringe integral.
    """
    isthick = True
    has_backtrack = True

    _xofields = {
        'model': xo.Int64,
        'is_exit': xo.Int64,
        'kn': xo.Float64[:],
        'ks': xo.Float64[:],
        'k_order': xo.Field(xo.Int64, default=-1),
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'kl_order': xo.Field(xo.Int64, default=-1),
        'length': xo.Float64,
        'half_gap': xo.Float64,
        'face_angle': xo.Float64,
        'face_angle_feed_down': xo.Float64,
        'fringe_integral': xo.Float64,
        'delta_taper': xo.Float64,
    }

    _rename = {
        'model': '_model',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/track_yrotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_wedge.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_fringe.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_linear.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_mult_fringe.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_magnet_edge.h'),
        _pkg_root.joinpath('beam_elements/elements_src/magnet_edge.h'),
    ]

    _repr_fields = [
        'model', 'is_exit', 'kn', 'ks', 'k_order', 'knl', 'ksl', 'kl_order',
        'length', 'half_gap', 'face_angle', 'face_angle_feed_down',
        'fringe_integral', 'delta_taper',
    ]

    _INDEX_TO_MODEL = {
        -1: 'suppressed',
        0: 'linear',
        1: 'full',
    }

    _MODEL_TO_INDEX = {v: k for k, v in _INDEX_TO_MODEL.items()}

    def __init__(self, **kwargs):
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)

        k_order = kwargs.pop('k_order', -1)
        kn, ks = kwargs.pop('kn', []), kwargs.pop('ks', [])
        k_multipolar_kwargs = _prepare_multipolar_params(
            k_order, skip_factorial=True, order_name='k_order', kn=kn, ks=ks)
        kwargs.update(k_multipolar_kwargs)

        kl_order = kwargs.pop('kl_order', -1)
        knl, ksl = kwargs.pop('knl', []), kwargs.pop('ksl', [])
        kl_multipolar_kwargs = _prepare_multipolar_params(
            kl_order, skip_factorial=True, order_name='kl_order', knl=knl, ksl=ksl)
        kwargs.update(kl_multipolar_kwargs)

        self.xoinitialize(**kwargs)

        if model is not None:
            self.model = model

    @property
    def model(self):
        return self._INDEX_TO_MODEL[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = self._MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid edge model: {value}')

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)

        if f'_model' in out:
            out.pop(f'_model')
        out['model'] = getattr(self, 'model')

        # See the comment in Multiple.to_dict about knl/ksl/order dumping
        for field in ['knl', 'ksl', 'kn', 'ks']:
            if field in out and np.allclose(out[field], 0, atol=1e-16):
                out.pop(field, None)

        if self.kl_order != -1 and 'knl' not in out and 'ksl' not in out:
            out['kl_order'] = self.order

        if self.k_order != -1 and 'kn' not in out and 'ks' not in out:
            out['k_order'] = self.order

        out['is_exit'] = bool(out['is_exit'])

        return out