from typing import List

import numpy as np
from scipy.special import factorial

import xobjects as xo
from xtrack.base_element import BeamElement
from xtrack.general import _pkg_root
from ..internal_record import RecordIndex
from ..random import RandomUniformAccurate, RandomExponential

DEFAULT_MULTIPOLE_ORDER = 5


_INDEX_TO_MODEL_CURVED = {
    0: 'adaptive',
    1: 'full',
    2: 'bend-kick-bend',
    3: 'rot-kick-rot',
    4: 'mat-kick-mat',
    5: 'drift-kick-drift-exact',
    6: 'drift-kick-drift-expanded',
}
_MODEL_TO_INDEX_CURVED = {k: v for v, k in _INDEX_TO_MODEL_CURVED.items()} | {'expanded': 4}

_INDEX_TO_INTEGRATOR = {
    0: 'adaptive',
    1: 'teapot',
    2: 'yoshida4',
    3: 'uniform',
}
_INTEGRATOR_TO_INDEX = {k: v for v, k in _INDEX_TO_INTEGRATOR.items()}

_INDEX_TO_EDGE_MODEL = {
   -1: 'suppressed',
    0: 'linear',
    1: 'full',
    2: 'dipole-only',
}
_EDGE_MODEL_TO_INDEX = {k: v for v, k in _INDEX_TO_EDGE_MODEL.items()}

_INDEX_TO_MODEL_STRAIGHT = _INDEX_TO_MODEL_CURVED.copy()
_INDEX_TO_MODEL_STRAIGHT.pop(2)
_INDEX_TO_MODEL_STRAIGHT.pop(3)
_MODEL_TO_INDEX_STRAIGHT = {k: v for v, k in _INDEX_TO_MODEL_STRAIGHT.items()}

_NOEXPR_FIELDS = {'model', 'integrator', 'edge_entry_model', 'edge_exit_model',
                  'name_associated_aperture'}

COMMON_MAGNET_SOURCES = [
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
]


class SynchrotronRadiationRecord(xo.HybridClass):
    _xofields = {
        '_index': RecordIndex,
        'photon_energy': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:],
        'particle_delta': xo.Float64[:]
    }


class MagnetDrift(BeamElement):
    """A drift slice in a magnet kick splitting. Mostly used for testing purposes.

    See ``Magnet`` for the description of the parameters.
    """
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
    """A thin kick in a magnet kick splitting. Mostly used for testing purposes.

    See ``Magnet`` for the description of the parameters.
    """
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


class Magnet(BeamElement):
    """General transverse field magnet with curvature and fringe fields.

    A beam element representing a magnet with transverse fields, curvature, and
    edge and fringe-field effects. Optional ``integrator`` and ``model``
    parameters can be used to specify the integration scheme and drift model to
    be used in the kick-splitting scheme. Default value is ``adaptive`` for
    both, which aims to provide best results in the general case (``rot-kick-rot``
    using the polar/exact, drift depending on h, for the model, and ``yoshida4``
    for the integration scheme).

    Parameters
    ----------
    length : float, optional
        Length of the element in meters along the reference trajectory.
    k0 : float, optional
        Strength of the horizontal dipolar component in units of m^-1.
    k1 : float, optional
        Strength of the horizontal quadrupolar component in units of m^-2.
    k2 : float, optional
        Strength of the horizontal sextupolar component in units of m^-3.
    k3 : float, optional
        Strength of the horizontal octupolar component in units of m^-4.
    k0s : float, optional
        Strength of the skew dipolar component in units of m^-1.
    k1s : float, optional
        Strength of the skew quadrupolar component in units of m^-2.
    k2s : float, optional
        Strength of the skew sextupolar component in units of m^-3.
    k3s : float, optional
        Strength of the skew octupolar component in units of m^-4.
    h : float, optional
        Curvature of the reference trajectory in units of m^-1 (= 1 / radius).
        Will imply the value of ``k0`` if ``k0_from_h`` is set.
    k0_from_h : bool, optional
        If true, the value of ``k0`` will be pinned to the value of ``h``.
    order : int, optional
        Maximum order of multipole expansion for this magnet. Defaults to 5.
    knl : list of floats, optional
        Normal multipole integrated strengths. If not provided, defaults to zeroes.
    ksl : list of floats, optional
        Skew multipole integrated strengths. If not provided, defaults to zeroes.
    model : str, optional
        Drift model to be used in the kick-splitting scheme. The options are:

            - ``adaptive``: default option, same as ``rot-kick-rot``.
            - ``full``: kept for backward compatibility, same as ``rot-kick-rot``.
            - ``bend-kick-bend``: use a thick (curved, if ``h`` non-zero) exact
                bend map for ``k0``, ``h``, and handle the other strengths in
                the kicks.
            - ``rot-kick-rot``: use an exact drift map (polar, if ``h`` non-zero)
                and handle all strengths in the kicks.
            - ``mat-kick-mat``: use an expanded combined-function magnet map
                for ``k0``, ``k1``, ``h``, and handle the other strengths in
                the kicks.
            - ``drift-kick-drift-exact``: use an exact drift map with no curvature,
                and handle all strengths in the kicks.
            - ``drift-kick-drift-expanded``: use an expanded drift map with no
                curvature, and handle all strengths in the kicks.

        These will not be applied if the length is zero.
    integrator : str, optional
        Integration scheme to be used. The options are:

            - ``adaptive``: default option, same as ``yoshida4``.
            - ``teapot``: use the Teapot integration scheme.
            - ``yoshida4``: use the Yoshida 4 integration scheme. The number of
                kicks will be implicitly rounded up to the nearest multiple of 7,
                as required by the scheme.
            - ``uniform``: slice uniformly.

        The integration scheme setting will be ignored if the length is zero, or
        if the strength and the curvature settings imply no need for applying
        thin kicks.
    num_multipole_kicks : int, optional
        The number of kicks to be used in thin kick splitting. If zero, and if
        the model selection implies that there are kicks that need to be
        performed, the value will be guessed according to a heuristic: one kick
        in the middle for straight magnets, or ~2 kicks/mrad otherwise.
    edge_entry_active : bool, optional
        Whether to include the edge effect at entry. Enabled by default.
    edge_exit_active : bool, optional
        Whether to include the edge effect at exit. Enabled by default.
    edge_entry_model : str, optional
        Edge model at magnet entry. The options are:

            - ``linear``: use a linear model for the edge.
            - ``full``: include all multipolar terms.
            - ``dipole-only``: ``full`` but includes only the dipolar terms.
            - ``suppressed``: ignore the edge effect.
    edge_exit_model : str, optional
        Edge model at magnet exit. See ``edge_entry_model`` for the options.
    edge_entry_angle : float, optional
        The angle of the entry edge in radians. Default is 0.
    edge_exit_angle : float, optional
        Same as `edge_entry_angle`, but for the exit.
    edge_entry_angle_fdown : float, optional
        Term added to the entry angle only for the ``linear`` mode and only in
        the vertical plane to account for non-zero angle in the closed orbit
        when entering the fringe field (feed down effect). Default is 0.
    edge_exit_angle_fdown : float, optional
        Same as ``edge_entry_angle_fdown``, but for the exit. Default is 0.
    edge_entry_fint: float, optional
        Fringe integral value at entry. Default is 0.
    edge_exit_fint : float, optional
        Same as ``edge_entry_fint``, but for the exit. Default is 0.
    edge_entry_hgap : float, optional
        Equivalent gap at entry in meters. Default is 0.
    edge_exit_hgap : float, optional
        Same as ``edge_entry_hgap``, but for the exit.
    radiation_flag : int, optional
        Flag indicating if synchrotron radiation effects are enabled.
        If zero, no radiation effects are simulated; if 1, the ``mean``
        model is used; if 2, the ``quantum`` model is used and the
        emitted photons are stored in the internal radiation record.
    delta_taper : float, optional
        A value added to delta for the purposes of tapering. Default is 0.
    """
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
        'delta_taper': xo.Float64,
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
        *COMMON_MAGNET_SOURCES,
        _pkg_root.joinpath('beam_elements/elements_src/magnet.h'),
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self, order=None, knl: List[float]=None, ksl: List[float]=None, **kwargs):
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = _prepare_multipolar_params(order, knl=knl, ksl=ksl)
        kwargs.update(multipolar_kwargs)

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
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

        if integrator is not None:
            self.integrator = integrator

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
        return _INDEX_TO_MODEL_CURVED[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _MODEL_TO_INDEX_CURVED[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    @property
    def integrator(self):
        return _INDEX_TO_INTEGRATOR[self._integrator]

    @integrator.setter
    def integrator(self, value):
        try:
            self._integrator = _INTEGRATOR_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid integrator: {value}')

    @property
    def edge_entry_model(self):
        return _INDEX_TO_EDGE_MODEL[self._edge_entry_model]

    @edge_entry_model.setter
    def edge_entry_model(self, value):
        try:
            self._edge_entry_model = _EDGE_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid edge model: {value}')

    @property
    def edge_exit_model(self):
        return _INDEX_TO_EDGE_MODEL[self._edge_exit_model]

    @edge_exit_model.setter
    def edge_exit_model(self, value):
        try:
            self._edge_exit_model = _EDGE_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid edge model: {value}')


class MagnetEdge(BeamElement):
    """Beam element modeling a magnet edge. Mostly used for testing purposes.

    Parameters
    ----------
    model : str
        Model to be used for the edge. See ``Magnet.edge_entry_model`` and
        ``Magnet.edge_exit_model`` for the options.
    is_exit : bool
        If False, the edge is the entrance edge. If True, the edge is an exit edge.
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

    _noexpr_fields = _NOEXPR_FIELDS

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
        return _INDEX_TO_EDGE_MODEL[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _EDGE_MODEL_TO_INDEX[value]
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


def _prepare_multipolar_params(
        order=None,
        skip_factorial=False,
        order_name='order',
        **kwargs,
):
    """Prepare the multipolar parameters for an element with kicks.

    This function takes the multipolar coefficients and the order, and extends/
    computes new arrays of compatible order, padding them with zeros if needed.

    Parameters
    ----------
    order : int, optional
        The multipolar order. If not provided, will be inferred from the other
        parameters.
    order_name : str, optional
        The name of the field in ``kwargs`` that stores the order.
    skip_factorial : bool, optional
        Whether to calculate ``inv_factorial_order``. Skipped by default.
    kwargs : dict
        A dictionary with values that are either array-type fields that contain
        multipolar coefficients, or None.

    Returns
    -------
    dict
        A dictionary containing the order field named appropriately and the
        arrays given in ``kwargs``. The arrays will be extended with zeros (and
        None will spawn zeroed arrays) compatible with the given order. If
        ``order`` is not given, its value will be inferred from the given
        arrays. If ``skip_factorial`` is False, the returned dictionary will
        also contain ``inv_factorial_order``.
    """
    order = order or 0

    lengths = [len(kwarg) if kwarg is not None else 0 for kwarg in kwargs.values()]

    target_len = max((order + 1), *lengths)
    assert target_len >= 0

    new_kwargs = {}
    for kwarg_name, kwarg in kwargs.items():
        new_kwarg = np.zeros(target_len, dtype=np.float64)
        new_kwargs[kwarg_name] = new_kwarg
        if kwarg is None:
            continue
        if hasattr(kwarg, 'get'):
            kwarg = kwarg.get()
        new_kwarg[: len(kwarg)] = np.array(kwarg)

    order = target_len - 1

    new_kwargs[order_name] = order

    if not skip_factorial:
        new_kwargs['inv_factorial_order'] = 1.0 / factorial(order, exact=True)

    return new_kwargs