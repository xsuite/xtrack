# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from warnings import warn

import numpy as np

from numbers import Number

from scipy.special import factorial

import xobjects as xo

from ..base_element import FloatOrTpsa
from ..internal_record import RecordIndex

DEFAULT_MULTIPOLE_ORDER = 5

_INDEX_TO_MODEL_DRIFT = {
    0: 'adaptive',
    1: 'expanded',
    2: 'exact'
}

_MODEL_TO_INDEX_DRIFT = {k: v for v, k in _INDEX_TO_MODEL_DRIFT.items()}

_INDEX_TO_MODEL_CURVED = {
    0: 'adaptive',
    1: 'full',
    2: 'bend-kick-bend',
    3: 'rot-kick-rot',
    4: 'mat-kick-mat',
    5: 'drift-kick-drift-exact',
    6: 'drift-kick-drift-expanded',
    7: 'rot-kick-rot-low-order',
    8: 'rot-kick-rot-high-order',
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

_INDEX_TO_MODEL_RF = _INDEX_TO_MODEL_STRAIGHT.copy()

_INDEX_TO_MODEL_RF.pop(1)

_INDEX_TO_MODEL_RF.pop(4)

_MODEL_TO_INDEX_RF = {k: v for v, k in _INDEX_TO_MODEL_RF.items()}

_NOEXPR_FIELDS = {'model', 'integrator', 'edge_entry_model', 'edge_exit_model',
                  'name_associated_aperture', 'rbend_model'}

_INDEX_TO_RBEND_MODEL = {
    0: 'adaptive',
    1: 'curved-body',
    2: 'straight-body'}

_RBEND_MODEL_TO_INDEX = {k: v for v, k in _INDEX_TO_RBEND_MODEL.items()}

_for_docstring_edge_straight = ('''
    edge_entry_active: bool
        Fringe field at the entrance edge is active if True. Default is False.
    edge_exit_active: bool
        Fringe field at the exit edge is active if True. Default is False.
    ''').strip()

_for_docstring_edge_bend = ('''
    edge_entry_active: bool
        Edge effects at the entrance edge are active if True. Default is True.
    edge_exit_active: bool
        Edge effects at the exit edge are active if True. Default is True.
    edge_entry_model : str
        Model used for the entrance edge. Available models are: "suppressed",
        "linear", "full", "dipole-only". Default is "linear".
    edge_exit_model : str
        Model used for the exit edge. Available models are: "suppressed",
        "linear", "full", "dipole-only". Default is "linear".
    edge_entry_angle : float
        Entrance edge angle in radians. Default is ``0``.
    edge_exit_angle : float
        Exit edge angle in radians. Default is ``0``.
    edge_entry_angle_fdown : float
        Angle of the reference trajectory at the entrance edge. Used only
        when `edge_entry_model` is "linear". Default is ``0``.
    edge_exit_angle_fdown : float
        Angle of the reference trajectory at the exit edge. Used only
        when `edge_exit_model` is "linear". Default is ``0``.
    edge_entry_fint : float
        Fringe field integral at the entrance edge. Used only when
        `edge_entry_model` is "full". Default is ``0``.
    edge_exit_fint : float
        Fringe field integral at the exit edge. Used only when
        `edge_exit_model` is "full". Default is ``0``.
    ''').strip()

_for_docstring_alignment = '''
    shift_x : float
        Horizontal shift of the element in meters. Default is ``0``.
    shift_y : float
        Vertical shift of the element in meters. Default is ``0``.
    shift_s : float
        Longitudinal shift of the element in meters. Default is ``0``.
    rot_s_rad : float
        Rotation around the longitudinal axis in radians. Default is ``0``.
    rot_x_rad : float
        Rotation around the horizontal axis in radians. Default is ``0``.
    rot_y_rad : float
        Rotation around the vertical axis in radians. Default is ``0``.
    rot_s_rad_no_frame : float
        Additional rotation around the longitudinal axis in radians. In this case
        the element field is rotated, but the reference frame at the interfaces
        is not changed. Default is ``0``.
    rot_shift_anchor : float
        Position along the element length where the rotations and shifts are applied.
        Given in meters from the element entrance. Default is ``0``.
'''.strip()

_docstring_general_notes = '''
    Notes
    -----

    Additional information on the definition of element properties and the
    implemented physics and models can be found in the Xsuite physics guide
    (https://xsuite.readthedocs.io/en/latest/physicsguide.html).
'''.strip()

class SynchrotronRadiationRecord(xo.HybridClass):
    _xofields = {
        '_index': RecordIndex,
        'photon_energy': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:],
        'particle_delta': xo.Float64[:]
    }

class _HasIntegrator:

    """
    Mixin class adding properties and methods for beam elements
    with integrator fields.
    """

    _for_docstring = ('''
    integrator : str
        Integrator used for the element. Available integrators are: "adaptive",
        "teapot", "yoshida4", "uniform". Default is "adaptive".
    num_multipole_kicks : int
        Number of multipole kicks to be used. For the yoshida integrator, this
        is rounded up to the nearest number compatible with the integrator scheme.
        Default is ``0``, for which the number of kicks is chosen automatically
        based on the element length and strength.
    ''').strip()

    @property
    def integrator(self):
        return _INDEX_TO_INTEGRATOR[self._integrator]

    @integrator.setter
    def integrator(self, value):
        try:
            self._integrator = _INTEGRATOR_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid integrator: {value}')

    _default_integrator = _INDEX_TO_INTEGRATOR[0]

    @staticmethod
    def get_available_integrators():
        """Get list of available integrators for this element.

        Returns
        -------
        List[str]
            List of available integrators.
        """
        out = [kk for kk in _INTEGRATOR_TO_INDEX.keys()]
        return out

class _HasModelDrift:

    """
    Mixin class adding properties and methods for beam elements
    with drift model fields.
    """

    @property
    def model(self):
        return _INDEX_TO_MODEL_DRIFT[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _MODEL_TO_INDEX_DRIFT[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    _default_model = _INDEX_TO_MODEL_DRIFT[0]

    @staticmethod
    def get_available_models():
        """Get list of available models for this element.

        Returns
        -------
        List[str]
            List of available models.
        """
        out = [kk for kk in _MODEL_TO_INDEX_DRIFT.keys()]
        return out

class _HasModelStraight:

    """
    Mixin class adding properties and methods for beam elements
    with model fields.
    """

    _for_docstring = ('''
    model : str
        Model used for the element. Available models are: "adaptive", "mat-kick-mat",
        "drift-kick-drift-exact", "drift-kick-drift-expanded". Default is "adaptive".
    ''').strip()

    @property
    def model(self):
        return _INDEX_TO_MODEL_STRAIGHT[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _MODEL_TO_INDEX_STRAIGHT[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    _default_model = _INDEX_TO_MODEL_STRAIGHT[0]

    @staticmethod
    def get_available_models():
        """Get list of available models for this element.

        Returns
        -------
        List[str]
            List of available models.
        """
        out = [kk for kk in _MODEL_TO_INDEX_STRAIGHT.keys() if kk != 'full']
        return out

class _HasModelCurved:

    """
    Mixin class adding properties and methods for beam elements
    with curved model fields.
    """

    _for_docstring = ('''
    model : str
        Model used for the element. Available models are: "adaptive",
        "bend-kick-bend", "rot-kick-rot", "mat-kick-mat",
        "drift-kick-drift-exact", "drift-kick-drift-expanded".
        Default is "adaptive".
    ''').strip()

    @property
    def model(self):
        return _INDEX_TO_MODEL_CURVED[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _MODEL_TO_INDEX_CURVED[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    _default_model = _INDEX_TO_MODEL_CURVED[0]

    @staticmethod
    def get_available_models():
        """Get list of available models for this element.

        Returns
        -------
        List[str]
            List of available models.
        """
        out = [kk for kk in _MODEL_TO_INDEX_CURVED.keys()
               if kk not in ('full', 'expanded')]
        return out

class _HasModelRF:

    """
    Mixin class adding properties and methods for beam elements
    with RF model fields.
    """

    @property
    def model(self):
        return _INDEX_TO_MODEL_RF[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _MODEL_TO_INDEX_RF[value]
        except KeyError:
            raise ValueError(f'Invalid RF model: {value}')

    _default_model = _INDEX_TO_MODEL_RF[0]

    @staticmethod
    def get_available_models():
        """Get list of available RF models for this element.
        """
        out = [kk for kk in _MODEL_TO_INDEX_RF.keys() if kk != 'full']
        return out

class _HasKnlKsl:

    """
    Mixin class adding properties and methods for beam elements
    with knl and ksl fields.
    """

    _for_docstring = ('''
    knl : array-like
        Integrated strengths of additional normal multipole components in m^(-order).
    ksl : array-like
        Integrated strengths of additional skew multipole components in m^(-order).
    order : int
        Maximum order of additional multipole components. Default is ``5``.
    ''').strip()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)

    def get_total_knl_ksl(self):
        nn = 4  # minimum length for knl and ksl is 4 (octupole order)
        nn = max(nn, len(self.knl))
        nn = max(nn, len(self.ksl))

        if 'knl_rel' in self._xo_fnames:
            nn = max(nn, len(self.knl_rel))
            nn = max(nn, len(self.ksl_rel))

        knl = np.zeros(nn, dtype=np.float64)
        ksl = np.zeros(nn, dtype=np.float64)
        knl[: len(self.knl)] += self._context.nparray_from_context_array(self.knl)
        ksl[: len(self.ksl)] += self._context.nparray_from_context_array(self.ksl)

        if 'knl_rel' in self._xo_fnames:
            knl[: len(self.knl_rel)] += self._context.nparray_from_context_array(
                self.main_strength * self.knl_rel)
            ksl[: len(self.ksl_rel)] += self._context.nparray_from_context_array(
                self.main_strength * self.ksl_rel)

        if 'k0' in self._xo_fnames:
            if hasattr(self, '_k0'): # To bypass k0 = from_angle
                knl[0] += self._k0 * self.length
            else:
                knl[0] += self.k0 * self.length

        for kk, ii in {'k1': 1, 'k2': 2, 'k3': 3}.items():
            if kk in self._xo_fnames:
                knl[ii] += getattr(self, kk) * self.length

        for kk, ii in {'k0s':0, 'k1s': 1, 'k2s': 2, 'k3s': 3}.items():
            if kk in self._xo_fnames:
                ksl[ii] += getattr(self, kk) * self.length

        return knl, ksl

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)

        if 'knl' in out and np.allclose(out['knl'], 0, atol=1e-16):
            out.pop('knl', None)

        if 'ksl' in out and np.allclose(out['ksl'], 0, atol=1e-16):
            out.pop('ksl', None)

        if self.order != 0 and 'knl' not in out and 'ksl' not in out:
            out['order'] = self.order

        return out

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        order = kwargs.pop('order', None)
        knl = kwargs.pop('knl', None)
        ksl = kwargs.pop('ksl', None)
        pn = kwargs.pop('pn', None) # Phase for RF multipoles
        ps = kwargs.pop('ps', None) # Phase for RF multipoles
        phase_n = kwargs.pop('phase_n', None) # Phase for RF multipoles
        phase_s = kwargs.pop('phase_s', None) # Phase for RF multipoles

        for nn, vv in {
                'pn': pn, 'ps': ps,
                'phase_n': phase_n, 'phase_s': phase_s}.items():
            if vv is not None and nn not in self._xofields:
                raise NameError(f"Invalid keyword argument `{nn}`")

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = self._prepare_multipolar_params(order,
                                            knl=knl, ksl=ksl,
                                            pn=pn, ps=ps,
                                            phase_n=phase_n, phase_s=phase_s)
        kwargs.update(multipolar_kwargs)

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)

        if 'knl_rel' in self._xo_fnames:
            _handle_knl_ksl_rel_kwargs(kwargs)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

    @staticmethod
    def _warn_if_deprecated_phase_is_nonzero(value, name, new_name):

        need_warn = False
        if np.isscalar(value) and value != 0:
            need_warn = True
        elif not np.isscalar(value):
            for v in value:
                if v != 0:
                    need_warn = True
                    break

        if need_warn:
            warn(f'`{name}` (in degrees) is deprecated and will be removed '
                 f'in a future version. Please use `{new_name}` (in radians) '
                 f'instead. Note that if both `{name}` and `{new_name}` are '
                 f'set, the effect is the sum of the two with `{name}` '
                 f'converted to radians.',
                 FutureWarning, stacklevel=2)

    def _prepare_multipolar_params(
        self,
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
            if kwarg_name not in self._xofields:
                continue
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

_ROT_AX_TO_ID = {'x': 0, 'y': 1, 's': 2}

_ROT_ID_TO_AX = {0: 'x', 1: 'y', 2: 's'}

def _handle_knl_ksl_rel_kwargs(kwargs):
    knl_rel = kwargs.pop('knl_rel', [0])
    ksl_rel = kwargs.pop('ksl_rel', [0])
    # pad to have the same length for knl_rel and ksl_rel
    max_len_rel = max(len(knl_rel), len(ksl_rel))
    if len(knl_rel) != len(ksl_rel):
        knl_rel = list(knl_rel) + [0] * (max_len_rel - len(knl_rel))
        ksl_rel = list(ksl_rel) + [0] * (max_len_rel - len(ksl_rel))
    kwargs['knl_rel'] = knl_rel
    kwargs['ksl_rel'] = ksl_rel

class _BendCommon(_HasKnlKsl, _HasIntegrator, _HasModelCurved):
    """Common properties for Bend and RBend: see their respective docstrings."""
    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _skip_in_to_dict = ['inv_factorial_order', 'h', 'k0_from_h', '_tpsa_enabled']

    _common_xofields = {
        'k0': FloatOrTpsa,
        'k1': FloatOrTpsa,
        'k2': FloatOrTpsa,
        'h': xo.Float64,
        'angle': xo.Float64,
        'length': xo.Float64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
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
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'knl_rel': xo.Float64[:],
        'ksl_rel': xo.Float64[:],
        'k0_from_h': xo.Field(xo.UInt64, default=1),
        '_tpsa_enabled': xo.Field(xo.Int8, default=0),
    }

    _common_rename = {
        'order': '_order',
        'model': '_model',
        'integrator': '_integrator',
        'edge_entry_model': '_edge_entry_model',
        'edge_exit_model': '_edge_exit_model',
        'k0': '_k0',
        'k0_from_h': '_k0_from_h',
        'angle': '_angle',
        'length': '_length',
        'h': '_h',
    }

    @property
    def main_strength(self):
        """Integrated strength of the main dipole component k0*length."""
        return self._k0 * self.length

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        if self.length != 0:
            self._h = self.angle / self.length
            if self.k0_from_h:
                self._k0 = self.h

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        if self.length != 0:
            self._h = self.angle / self.length
        else:
            self._h = 0.0

        if self.k0_from_h:
            self._k0 = self.h

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        raise RuntimeError("Setting `h` directly is not allowed. "
                           "Set `length` and `angle` instead.")

    @property
    def k0(self):
        if self.k0_from_h:
            return 'from_h'
        return self._k0

    @k0.setter
    def k0(self, value):
        if isinstance(value, str):
            if value != 'from_h':
                raise ValueError("k0 can only be set to 'from_h' as a string")
            self.k0_from_h = True
        else:
            self.k0_from_h = False
            self._k0 = value

    _default_k0 = 'from_h'

    @property
    def k0_from_h(self):
        return bool(self._k0_from_h)

    @k0_from_h.setter
    def k0_from_h(self, value):
        if value:
            self._k0 = self.h
        elif self.k0_from_h: # was true before
            self._k0 = 0.
        self._k0_from_h = value

    @property
    def edge_entry_model(self):
        return _INDEX_TO_EDGE_MODEL[self._edge_entry_model]

    @edge_entry_model.setter
    def edge_entry_model(self, value):
        try:
            self._edge_entry_model = _EDGE_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    _default_edge_entry_model = _INDEX_TO_EDGE_MODEL[0]

    @property
    def edge_exit_model(self):
        return _INDEX_TO_EDGE_MODEL[self._edge_exit_model]

    @edge_exit_model.setter
    def edge_exit_model(self, value):
        try:
            self._edge_exit_model = _EDGE_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid model: {value}')

    _default_edge_exit_model = _INDEX_TO_EDGE_MODEL[0]

    @property
    def _repr_fields(self):
        return ['length', 'k0', 'k1', 'h', 'k0_from_h', 'model', 'knl', 'ksl',
                'knl_rel', 'ksl_rel',
                'edge_entry_active', 'edge_exit_active', 'edge_entry_model',
                'edge_exit_model', 'edge_entry_angle', 'edge_exit_angle',
                'edge_entry_angle_fdown', 'edge_exit_angle_fdown',
                'edge_entry_fint', 'edge_exit_fint', 'edge_entry_hgap',
                'edge_exit_hgap', 'shift_x', 'shift_y', 'rot_s_rad']

    @property
    def sagitta(self):
        if abs(self.angle) < 1e-10:  # avoid numerical issues
            return 0.0
        else:
            return 1. / self.h * (1 - np.cos(0.5 * self.angle))

    @classmethod
    def from_dict(cls, dct, **kwargs):

        dct = dct.copy()

        # Backward compatibility
        if 'h' in dct:
            if 'angle' not in dct:
                assert 'length' in dct
                dct['angle'] = dct['h'] * dct['length']
            dct.pop('h')

        if 'k0_from_h' in dct and dct['k0_from_h']:
            dct['k0'] = 'from_h'
            dct.pop('k0_from_h')

        return super().from_dict(dct, **kwargs)

def _angle_from_trig(cos=None, sin=None, tan=None):
    """
    Given at least two values of (cos, sin, tan), return the angle in radians.
    Raises ValueError if the values are inconsistent.
    """
    sin_given, cos_given, tan_given = (trig is not None for trig in (sin, cos, tan))

    if sum([sin_given, cos_given, tan_given]) <= 1:
        raise ValueError('At least two of (cos, sin, tan) must be given')

    if sin_given and cos_given:
        tan = tan if tan_given else sin / cos
    elif sin_given and tan_given:
        cos = cos if cos_given else sin / tan
    elif cos_given and tan_given:
        sin = sin if sin_given else cos * tan

    if (not np.isclose(sin**2 + cos**2, 1, atol=1e-13)
            or not np.isclose(sin / cos, tan, atol=1e-13)):
        raise ValueError('Given values of sin, cos, tan are inconsistent '
                         'with each other.')

    angle = np.arctan2(sin, cos)
    return angle, cos, sin, tan

def _unregister_if_preset(ref):
    try:
        ref._manager.unregister(ref)
    except KeyError:
        pass

def _get_expr(knob):
    """Return an xdeps expression for `knob`, or, if unavailable, the value."""
    if knob is None:
        return 0
    if hasattr(knob, '_expr'):
        if knob._expr is not None:
            return knob._expr

        value = knob._get_value()
        if hasattr(value, 'get'):  # On cupy, pyopencl gets ndarray
            value = value.get()
        if hasattr(value, 'item'):  # Extract the scalar
            value = value.item()
        return value
    if isinstance(knob, Number):
        return knob
    if hasattr(knob, 'dtype'):
        if hasattr(knob, 'get'):
            return knob.get()
        return knob
    raise ValueError(f'Cannot get expression for {knob}.')

def _nonzero(val_or_expr):
    if isinstance(val_or_expr, Number):
        return val_or_expr != 0

    return val_or_expr._expr

class ElectronCoolerRecord(xo.HybridClass):
    _xofields = {
        '_index': RecordIndex,
        'Fx': xo.Float64[:],
        'Fy': xo.Float64[:],
        'Fl': xo.Float64[:],
        'particle_id': xo.Float64[:]}

class ThinSliceNotNeededError(Exception):
    pass
