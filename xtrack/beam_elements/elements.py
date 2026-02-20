# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
from typing import List
from warnings import warn

import numpy as np
from numbers import Number
from scipy.special import factorial

import xobjects as xo
import xtrack as xt

from ..base_element import BeamElement
from ..random import RandomUniformAccurate, RandomExponential, RandomNormal

from xtrack.internal_record import RecordIndex

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

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = self._prepare_multipolar_params(order,
                                            knl=knl, ksl=ksl, pn=pn, ps=ps)
        kwargs.update(multipolar_kwargs)

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

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


class ReferenceEnergyIncrease(BeamElement):

    '''Beam element modeling a change of reference energy (acceleration,
    deceleration).

    Parameters
    ----------
    Delta_p0c : float
        Change in reference energy in eV. Default is ``0``.

    '''

    _xofields = {
        'Delta_p0c': xo.Float64}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/referenceenergyincrease.h>',
    ]

    has_backtrack = True
    allow_rot_and_shift = False


class Marker(BeamElement):
    """A marker beam element with no effect on the particles.
    """

    _xofields = {
        '_dummy': xo.Int64}

    behaves_like_drift = True
    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False
    _skip_in_repr = ['_dummy']

    _extra_c_sources = [
        "#include <beam_elements/elements_src/marker.h>",
    ]


class Drift(_HasModelDrift, BeamElement):

    _docstring_start = """Beam element modeling a drift section.

    Parameters
    ----------

    length : float
        Length of the drift section in meters. Default is ``0``.
    model : str
        Model used for the drift element. Available models are: "adaptive",
        "expanded", "exact". Default is "adaptive".

    """

    __doc__ = '\n    '.join([_docstring_start, _docstring_general_notes])

    _xofields = {
        'length': xo.Float64,
        'model': xo.Int64
    }

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift.h>',
    ]

    _rename = {
        'model': '_model',
    }

    _noexpr_fields = {'model'}

    def __init__(self, length=None, model=None, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if length:  # otherwise length cannot be set as a positional argument
            kwargs['length'] = length
        super().__init__(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

    @property
    def _thin_slice_class(self):
        return None

    @property
    def _thick_slice_class(self):
        return xt.DriftSlice

    @property
    def _drift_slice_class(self):
        return xt.DriftSlice


class DriftExact(BeamElement):
    """Beam element modeling an exact drift section.

    Parameters
    ----------

    length : float
        Length of the drift section in meters. Default is ``0``.
    """

    _xofields = {
        'length': xo.Float64
    }

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_exact.h>',
    ]

    def __init__(self, length=None, **kwargs):
        if length:  # otherwise length cannot be set as a positional argument
            kwargs['length'] = length
        super().__init__(**kwargs)

    @property
    def _thin_slice_class(self):
        return None

    @property
    def _thick_slice_class(self):
        return xt.DriftExactSlice

    @property
    def _drift_slice_class(self):
        return xt.DriftExactSlice



class Cavity(_HasModelRF, _HasIntegrator, BeamElement):

    _docstring_start = \
    '''RF cavity element.

    Parameters
    ----------
    length : float
        Length of the RF cavity in meters. Default is ``0``.
    voltage : float
        Voltage of the RF cavity in Volts. Default is ``0``.
    frequency : float
        Frequency of the RF cavity in Hertz. It can be set only if harmonic is zero.
        Default is ``0``.
    harmonic : float
        Harmonic number of the RF cavity. It can be set only if frequency is zero.
        If `harmonic` is non-zero, the frequency is computed from the length of the
        beam_line and the speed of the reference particle (beta0 * clight).
        When `harmonic` is set, the cavity can only be used within a Line and not
        in standalone tracking (i.e. Cavity.track(...) will raise an error).
        Default is ``0``.
    lag : float
        Phase in degrees seen at the arrival time of the reference particle (zeta = 0).
        When `absolute_time` is True `lag` is the phase at time zero. Default is ``0``.
    absolute_time : bool
        If True, the cavity phase is computed from the absolute time of the
        simulation, otherwise the cavity is synchronized with the arrival time of
        the reference particle (zeta=0). Default is False.
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start,
        _HasModelStraight._for_docstring,
        _HasIntegrator._for_docstring.replace(
            'num_multipole_kicks', 'num_kicks').replace('multipole kicks', 'kicks'),
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'length': xo.Float64,
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'harmonic': xo.Float64,
        'lag_taper': xo.Float64,
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/cavity.h>',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'frequency': '_frequency',
        'harmonic': '_harmonic',
    }

    _default_frequency = 0.0
    _default_harmonic = 0.0

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        frequency = kwargs.pop('frequency', None)
        harmonic = kwargs.pop('harmonic', None)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

        if frequency is not None:
            self.frequency = frequency

        if harmonic is not None:
            self.harmonic = harmonic

    def track(self, particles, *args, **kwargs):

        if self.harmonic != 0:
            raise RuntimeError("Cavity cannot be used in standalone tracking "
                               "when harmonic is not zero. Please use the "
                               "cavity within a Line or set frequency instead"
                               " of harmonic.")
        return super().track(particles, *args, **kwargs)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if self._harmonic != 0 and value != 0:
            raise ValueError("Cannot set non-zero frequency when harmonic is not zero.")
        self._frequency = value

    @property
    def harmonic(self):
        return self._harmonic

    @harmonic.setter
    def harmonic(self, value):
        if self._frequency != 0 and value != 0:
            raise ValueError("Cannot set non-zero harmonic when frequency is not zero.")
        self._harmonic = value

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceCavity


class CrabCavity(_HasModelRF, _HasIntegrator, BeamElement):
    _docstring_start = \
    '''Crab cavity element.

    Parameters
    ----------
    length : float
        Length of the RF cavity in meters. Default is ``0``.
    crab_voltage : float
        Voltage associated to the horizontal RF deflection in Volts. Default is ``0``.
    frequency : float
        Frequency of the cavity in Hertz. It can be set only if harmonic is zero.
        Default is ``0``.
    lag : float
        Phase in degrees seen at the arrival time of the reference particle (zeta = 0).
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start,
        _HasModelStraight._for_docstring,
        _HasIntegrator._for_docstring.replace(
            'num_multipole_kicks', 'num_kicks').replace('multipole kicks', 'kicks'),
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'length': xo.Float64,
        'crab_voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'lag_taper': xo.Float64,
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/crab_cavity.h>',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceCrabCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceCrabCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceCrabCavity



class XYShift(BeamElement):
    '''
    Beam element modeling an transverse shift of the reference system, by applying
    the following transformation to the particle coordinates:

        x_new = x_old - dx
        y_new = y_old - dy

    Parameters
    ----------
    dx : float
        Horizontal shift in meters. Default is ``0``.
    dy : float
        Vertical shift in meters. Default is ``0``.

    '''
    _xofields = {
        'dx': xo.Float64,
        'dy': xo.Float64,
        }

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/xyshift.h>',
    ]


class Elens(BeamElement):
    '''Beam element modeling a hollow electron lens.

    Parameters
    ----------
    inner_radius : float
        Inner radius of the electron lens in meters. Default is ``0``.
    outer_radius : float
        Outer radius of the electron lens in meters. Default is ``0``.
    current : float
        Current of the electron lens in Ampere. Default is ``0``.
    elens_length : float
        Length of the electron lens in meters. Default is ``0``.
    voltage : float
        Voltage of the electron lens in Volts. Default is ``0``.
    residual_kick_x : float
        Residual kick in the horizontal plane in radians. Default is ``0``.
    residual_kick_y : float
        Residual kick in the vertical plane in radians. Default is ``0``.
    coefficients_polynomial : array
        Array of coefficients of the polynomial. Default is ``[0]``.
    polynomial_order : int
        Order of the polynomial. Default is ``0``.

    '''

    _xofields={
        'current': xo.Float64,
        'inner_radius': xo.Field(xo.Float64, default=1.),
        'outer_radius': xo.Field(xo.Float64, default=1.),
        'elens_length': xo.Float64,
        'voltage': xo.Float64,
        'residual_kick_x': xo.Float64,
        'residual_kick_y': xo.Float64,
        'coefficients_polynomial': xo.Field(xo.Float64[:], default=[0]),
        'polynomial_order': xo.Float64,
    }

    has_backtrack = True

    _extra_c_sources = [
        '#include <beam_elements/elements_src/elens.h>',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)
        polynomial_order = len(self.coefficients_polynomial) - 1
        self.polynomial_order = polynomial_order


class NonLinearLens(BeamElement):
    '''
    Beam element modeling a non-linear lens with elliptic potential.
    See the corresponding element in MAD-X documentation.

    Parameters
    ----------
    knll : float
        Integrated strength of lens (m). The strength is parametrized so that
        the quadrupole term of the multipole expansion is k1=2*knll/cnll^2.
    cnll : float
        Focusing strength (m).
        The dimensional parameter of lens (m).
        The singularities of the potential are located at x=-cnll, +cnll and y=0.
    '''

    _xofields={
        'knll': xo.Float64,
        'cnll': xo.Float64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/nonlinearlens.h>',
    ]


class Wire(BeamElement):

    '''Beam element modeling a wire (used for long range beam-beam compensation).

    Parameters
    ----------

    L_phy : float
        Physical length of the wire in meters. Default is ``0``.
    L_int : float
        Interaction length of the wire in meters. Default is ``0``.
    current : float
        Current of the wire in Ampere. Default is ``0``.
    xma : float
        Horizontal position of the wire in meters. Default is ``0``.
    yma : float
        Vertical position of the wire in meters. Default is ``0``.
    post_subtract_px : float
        Horizontal post-subtraction kick in radians. Default is ``0``.
    post_subtract_py : float
        Vertical post-subtraction kick in radians. Default is ``0``.
    '''

    _xofields={
               'L_phy'  : xo.Float64,
               'L_int'  : xo.Float64,
               'current': xo.Float64,
               'xma'    : xo.Float64,
               'yma'    : xo.Float64,

               'post_subtract_px': xo.Float64,
               'post_subtract_py': xo.Float64,
              }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/wire.h>',
    ]


class SRotation(BeamElement):
    """
    Beam element modeling a rotation of the reference system around the s-axis.
    The sign convention is such that:

            px_out = px_in * cos(angle) - py_in * sin(angle)


    Parameters
    ----------
    angle : float
        Rotation angle in degrees. Default is 0.
    """

    _xofields = {
        'cos_z': xo.Float64,
        'sin_z': xo.Float64,
    }

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/srotation.h>',
    ]

    _store_in_to_dict = ['angle']
    _skip_in_to_dict = ['sin_z', 'cos_s']

    def __init__(self, angle=None, cos_z=None, sin_z=None, **kwargs):
        """
        If either angle or a sufficient number of trig values are given,
        calculate the missing values from the others. If more than necessary
        parameters are given, their consistency will be checked.
        """

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if angle is None and (cos_z is not None or sin_z is not None):
            anglerad, cos_angle, sin_angle, _ = _angle_from_trig(cos_z, sin_z)
        elif angle is not None:
            anglerad = angle / 180 * np.pi
        else:
            anglerad = 0.0

        if cos_z is None:
            cos_z = np.cos(anglerad)
        elif not np.isclose(cos_z, np.cos(anglerad), atol=1e-13):
            raise ValueError(f'cos_z does not match angle: {cos_z} vs {anglerad}')

        if sin_z is None:
            sin_z = np.sin(anglerad)
        elif not np.isclose(sin_z, np.sin(anglerad), atol=1e-13):
            raise ValueError('sin_z does not match angle')

        super().__init__(cos_z=cos_z, sin_z=sin_z, **kwargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_z, self.cos_z) * (180.0 / np.pi)

    @angle.setter
    def angle(self, value):
        anglerad = value / 180 * np.pi
        self.cos_z = np.cos(anglerad)
        self.sin_z = np.sin(anglerad)


class XRotation(BeamElement):
    """
    Beam element modeling a rotation of the reference system around the x-axis.
    The sign convention is such that:

          py_out = py_in * cos(angle) + pz_in * sin(angle)

    Parameters
    ----------
    angle : float
        Rotation angle in degrees. Default is 0.
    """

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/xrotation.h>',
    ]

    _store_in_to_dict = ['angle']
    _skip_in_to_dict = ['sin_angle', 'cos_angle', 'tan_angle']

    def __init__(
            self,
            angle=None,
            cos_angle=None,
            sin_angle=None,
            tan_angle=None,
            **kwargs,
    ):
        """
        If either angle or a sufficient number of trig values are given,
        calculate the missing values from the others. If more than necessary
        parameters are given, their consistency will be checked.
        """
        # Note MAD-X node_value('other_bv ') is ignored

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        at_least_one_trig = sum(trig is not None for trig
                                in (cos_angle, sin_angle, tan_angle)) > 0

        if angle is None and at_least_one_trig:
            params = _angle_from_trig(cos_angle, sin_angle, tan_angle)
            anglerad, cos_angle, sin_angle, tan_angle = params
        elif angle is not None:
            anglerad = angle / 180 * np.pi
        else:
            anglerad = 0.0

        if cos_angle is None:
            cos_angle = np.cos(anglerad)
        elif not np.isclose(cos_angle, np.cos(anglerad), atol=1e-13):
            raise ValueError('cos_angle does not match angle')

        if sin_angle is None:
            sin_angle = np.sin(anglerad)
        elif not np.isclose(sin_angle, np.sin(anglerad), atol=1e-13):
            raise ValueError('sin_angle does not match angle')

        if tan_angle is None:
            tan_angle = np.tan(anglerad)
        elif not np.isclose(tan_angle, np.tan(anglerad), atol=1e-13):
            raise ValueError('tan_angle does not match angle')

        super().__init__(
            cos_angle=cos_angle, sin_angle=sin_angle, tan_angle=tan_angle,
            **kwargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_angle,self.cos_angle) * (180.0 / np.pi)

    @angle.setter
    def angle(self, value):
        anglerad = value / 180 * np.pi
        self.cos_angle = np.cos(anglerad)
        self.sin_angle = np.sin(anglerad)
        self.tan_angle = np.tan(anglerad)


class YRotation(BeamElement):
    """
    Beam element modeling a rotation of the reference system around the y-axis.
    The sign convention is such that:

            px_out = px_in * cos(angle) - pz_in * sin(angle)

    Parameters
    ----------
    angle : float
        Rotation angle in degrees. Default is 0.
    """

    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/yrotation.h>',
    ]

    _store_in_to_dict = ['angle']
    _skip_in_to_dict = ['sin_angle', 'cos_angle', 'tan_angle']

    def __init__(
            self,
            angle=None,
            cos_angle=None,
            sin_angle=None,
            tan_angle=None,
            **kwargs,
    ):
        """
        If either angle or a sufficient number of trig values are given,
        calculate the missing values from the others. If more than necessary
        parameters are given, their consistency will be checked.
        """

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        at_least_one_trig = sum(
            trig is not None for trig
                in (cos_angle, sin_angle, tan_angle)
        ) > 0

        if angle is None and at_least_one_trig:
            params = _angle_from_trig(cos_angle, sin_angle, tan_angle)
            anglerad, cos_angle, sin_angle, tan_angle = params
        elif angle is not None:
            anglerad = angle / 180 * np.pi
        else:
            anglerad = 0.0

        if cos_angle is None:
            cos_angle = np.cos(anglerad)
        elif not np.isclose(cos_angle, np.cos(anglerad), atol=1e-13):
            raise ValueError('cos_angle does not match angle')

        if sin_angle is None:
            sin_angle = np.sin(anglerad)
        elif not np.isclose(sin_angle, np.sin(anglerad), atol=1e-13, rtol=0):
            raise ValueError('sin_angle does not match angle')

        if tan_angle is None:
            tan_angle = np.tan(anglerad)
        elif not np.isclose(tan_angle, np.tan(anglerad), atol=1e-13):
            raise ValueError('tan_angle does not match angle')

        super().__init__(
            cos_angle=cos_angle, sin_angle=sin_angle, tan_angle=tan_angle,
            **kwargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_angle, self.cos_angle) * (180.0 / np.pi)

    @angle.setter
    def angle(self, value):
        anglerad = value / 180 * np.pi
        self.cos_angle = np.cos(anglerad)
        self.sin_angle = np.sin(anglerad)
        self.tan_angle = np.tan(anglerad)


class ZetaShift(BeamElement):
    '''Beam element modeling a time delay.

    Parameters
    ----------

    dzeta : float
        Time shift dzeta in meters. Default is ``0``.

    '''

    _xofields={
        'dzeta': xo.Float64,
        }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/zetashift.h>',
    ]

    _store_in_to_dict = ['dzeta']

class Misalignment(BeamElement):
    """Beam element modeling a misalignment of a strait or curved element.

    Parameters
    ----------
    dx : float
        Misalignment in x in m.
    dy : float
        Misalignment in y in m.
    ds : float
        Misalignment in s in m.
    theta : float
        Rotation around y, yaw, positive s to x, in radians.
    phi : float
        Rotation around x, pitch, positive s to y, in radians.
    psi : float
        Rotation around s, roll, positive y to x, in radians.
    anchor : float
        Location of the misalignment as an offset in m from the element entry.
    length : float
        Length of the misaligned element in m.
    angle : float
        Angle by which the element bends the reference frame in the x-s plane.
        Direction follows the convention of the bend element, i.e. positive
        value bends x to s (opposite of phi), in radians.
    h : float
        Curvature of the element in 1/m, to be specified only for thin slices,
        i.e. when element length is zero (and therefore angle is also zero), but
        which represent slices of a curved element: in such a case curvature
        matters for the cases when ``anchor`` is not zero.
    tilt : float
        Angle (in radians) by which the element body is tilted (rolled) around
        the s-axis. Direction follows the convention of psi.
    is_exit : bool
        If False, this element brings the reference frame to the entrance of the
        misaligned element, if True, it brings the reference frame back to the
        non-misaligned frame from the exit of the element in the misaligned frame.
    """
    _xofields = {
        'dx': xo.Float64,
        'dy': xo.Float64,
        'ds': xo.Float64,
        'theta': xo.Float64,
        'phi': xo.Float64,
        'psi': xo.Float64,
        'anchor': xo.Float64,
        'length': xo.Float64,
        'angle': xo.Float64,
        'h': xo.Float64,
        'tilt': xo.Float64,
        'is_exit': xo.Int64,
    }
    has_backtrack = False
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/misalignment.h>',
    ]


class Multipole(_HasKnlKsl, _HasModelStraight, _HasIntegrator, BeamElement):

    _docstring_start = \
    """Beam element modeling a magnetic multipole.

    Parameters
    ----------

    knl : array
        Integrated strength of the normal components in units of m^-n.
    ksl : array
        Integrated strength of the skew components in units of m^-n.
    order : int
        Order of the multipole. By default it is inferred from the length of
        knl and ksl.
    hxl : float
        Rotation angle in radians applied to the reference trajectory in the
        horizontal plane. Default is ``0``.
    length : float
        Length of the originating thick multipole. Default is ``0``.
    isthick : bool
        Whether the multipole is to be treated as thick (True) or thin (False).
        Default is ``False``.
    """

    __doc__ = '\n    '.join([_docstring_start.strip(),
                             _HasModelCurved._for_docstring,
                             _HasIntegrator._for_docstring,
                             _for_docstring_edge_straight,
                             _for_docstring_alignment, '\n',
                             _docstring_general_notes, '\n\n'])

    #isthick can be changed dynamically for this element

    has_backtrack = True

    _xofields={
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'length': xo.Float64,
        'hxl': xo.Float64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'isthick': xo.Int64,
        'num_multipole_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        }

    _rename = {
        'order': '_order',
        'isthick': '_isthick',
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _extra_c_sources = [
        '#include <beam_elements/elements_src/multipole.h>',
    ]

    _internal_record_class = SynchrotronRadiationRecord

    @property
    def allow_loss_refinement(self):
        '''
        Loss refinement is allowed only for thick multipoles with non-zero length.
        '''
        # Allow refinement only when thick (to keep old behavior when thin and
        # have consistency with other thick elements otherwise)
        return self.isthick and self.length != 0

    def __init__(self, **kwargs):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        order = kwargs.pop('order', None)
        knl = kwargs.pop('knl', None)
        ksl = kwargs.pop('ksl', None)

        multipolar_kwargs = self._prepare_multipolar_params(order, knl=knl, ksl=ksl)
        kwargs.update(multipolar_kwargs)

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        isthick = kwargs.pop('isthick', None)

        if "bal" in kwargs.keys():
            raise ValueError("`bal` not supported anymore")

        if 'hyl' in kwargs.keys():
            assert kwargs['hyl'] == 0.0, 'hyl is not supported anymore'

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

        if isthick is not None:
            self.isthick = isthick

    @property
    def hyl(self):
        raise ValueError("hyl is not anymore supported")

    @hyl.setter
    def hyl(self, value):
        raise ValueError("hyl is not anymore supported")

    @property
    def isthick(self):
        return bool(self._isthick > 0)

    @isthick.setter
    def isthick(self, value):
        self._isthick = int(bool(value))

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceMultipole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceMultipole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceMultipole

class SimpleThinQuadrupole(BeamElement):
    """An specialized version of Multipole to model a thin quadrupole
    (knl[0], ksl, hxl, are all zero).

    Parameters
    ----------
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length 2.

    """

    _xofields={
        'knl': xo.Field(xo.Float64[2], default=[0, 0]),
    }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/simplethinquadrupole.h>',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        knl = kwargs.get('knl')
        if knl is not None and len(knl) != 2:
                raise ValueError("For a quadrupole, len(knl) must be 2.")

        super().__init__(**kwargs)

    @property
    def hxl(self): return 0.0

    @property
    def length(self): return 0.0

    @property
    def radiation_flag(self): return 0.0

    @property
    def order(self): return 1

    @property
    def inv_factorial_order(self): return 1.0

    @property
    def ksl(self): return self._buffer.context.linked_array_type.from_array(
        np.array([0., 0.]),
        mode='readonly',
        container=self,
    )


class _BendCommon(_HasKnlKsl, _HasIntegrator, _HasModelCurved):
    """Common properties for Bend and RBend: see their respective docstrings."""
    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _skip_in_to_dict = ['inv_factorial_order', 'h', 'k0_from_h']

    _common_xofields = {
        'k0': xo.Float64,
        'k1': xo.Float64,
        'k2': xo.Float64,
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
        'k0_from_h': xo.Field(xo.UInt64, default=1),
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


class Bend(_BendCommon, BeamElement):

    _docstring_start = \
    """Bending magnet element, sector-bend type.

    Parameters
    ----------
    length : float
        Length of the element in meters along the reference trajectory.
    angle : float
        Angle of the bend in radians. This is the angle by which the reference
        trajectory is bent in the horizontal plane.
    k0 : float, optional
        Strength of the horizontal dipolar component in units of m^-1.
        It can be set to the string value 'from_h', in which case `k0` is
        computed from the curvature defined by `angle` and `length`
        (i.e. `k0 = h = angle/length`) and `k0_from_h` is set to True.
    k1 : float, optional
        Strength of the quadrupolar component in units of m^-2.
    k2 : float, optional
        Strength of the sextupolar component in units of m^-3.
    k0_from_h : bool, optional
        If True, `k0` is computed from the curvature defined by `angle` and
        `length` (i.e. `k0 = h = angle/length`). Default is True. The flag
        becomes false when `k0` is set directly to a numeric value.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
            _HasModelCurved._for_docstring, _HasIntegrator._for_docstring,
            _for_docstring_edge_bend, _for_docstring_alignment, '\n',
            _docstring_general_notes, '\n\n'])

    allow_loss_refinement = True

    _xofields = _BendCommon._common_xofields
    _rename = _BendCommon._common_rename

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    _noexpr_fields = _NOEXPR_FIELDS

    _extra_c_sources = [
        '#include <beam_elements/elements_src/bend.h>',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if 'h' in kwargs:
            raise ValueError("Setting `h` directly is not allowed. "
                                "Set `length` and `angle` instead.")

        to_be_set_with_properties = []
        for nn in ['length', 'angle', 'k0_from_h', 'edge_entry_model',
                   'edge_exit_model', 'k0']:
            if nn in kwargs:
                to_be_set_with_properties.append((nn, kwargs.pop(nn)))

        _HasKnlKsl.__init__(self, **kwargs)

        for nn, val in to_be_set_with_properties:
            setattr(self, nn, val)

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceBend

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceBend

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceBend

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceBendEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceBendExit

    @property
    def _repr_fields(self):
        return super()._repr_fields


class RBend(_BendCommon, BeamElement):
    _docstring_start = \
    """Rectangular bending magnet element.

    Parameters
    ----------
    length_straith : float
        Length of the element in meters along the axis of the magnet (straight line
        between entry and exit points). This is different from the length of the
        reference trajectory, i.e. the increase of the `s` coordinate through the
        element, which is computed internally and can be inspected via the
        `length` property.
    angle : float
        Angle of the bend in radians. This is the angle by which the reference
        trajectory is bent in the horizontal plane.
    k0 : float
        Strength of the horizontal dipolar component in units of m^-1.
        It can be set to the string value 'from_h', in which case `k0` is
        computed from the curvature defined by `angle` and `length`
        (i.e. `k0 = h = angle/length`) and `k0_from_h` is set to True.
    k1 : float
        Strength of the quadrupolar component in units of m^-2.
    k2 : float
        Strength of the sextupolar component in units of m^-3.
    k0_from_h : bool
        If True, `k0` is computed from the curvature defined by `angle` and
        `length` (i.e. `k0 = h = angle/length`). Default is True. The flag
        becomes false when `k0` is set directly to a numeric value.
    rbend_model : str
        Model used for the rectangular bend. Possible values are:
        "adaptive', "curved-body", "straight-body". Default is "adaptive',
        which falls back to "curved-body".
    rbend_angle_diff : float
        Difference in radians between the angle of the reference trajectory
        with respect to the magnet axis at the entrance and exit of the magnet.
        See drawing on Xsuite Physics Guide. Default is 0.0.
    rbend_shift : float
        Shift of the magnet body, in meters, defined as the displacement
        of the reference trajectory with respect to the magnet axis at the center
        of the magnet. This parameter has effect only when `rbend_model` is
        "straight-body". Default is 0.0.
    rbend_compensate_sagitta : bool
        If True, the magnet body is shifted by half of the trajectory sagitta,
        defined as (1 / h) * (1 - cos(angle / 2)). The shift is added to `rbend_shift`.
        This parameter has effect only when `rbend_model` is "straight-body".
        Default is True.
    """

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
            _HasModelCurved._for_docstring, _HasIntegrator._for_docstring,
            _for_docstring_edge_bend, _for_docstring_alignment, '\n',
            _docstring_general_notes, '\n\n'])

    _xofields = {
        **_BendCommon._common_xofields,
        'length_straight': xo.Float64,
        'rbend_model': xo.Int64,
        'rbend_compensate_sagitta': xo.Field(xo.Int64, default=True),
        'rbend_shift': xo.Float64,
        'rbend_angle_diff': xo.Float64,
    }

    allow_loss_refinement = True

    _rename = {
        **_BendCommon._common_rename,
        'length_straight': '_length_straight',
        'rbend_model': '_rbend_model',
        'rbend_angle_diff': '_rbend_angle_diff',
        'rbend_compensate_sagitta': '_rbend_compensate_sagitta',
    }

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include <beam_elements/elements_src/rbend.h>',
    ]

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if 'h' in kwargs:
            raise ValueError("Setting `h` directly is not allowed. "
                                "Set `length` and `angle` instead.")

        if 'length' in kwargs:
            raise ValueError("Setting `length` directly is not allowed for RBend. "
                             "Set `length_straight` instead.")

        to_be_set_with_properties = []
        for nn in ['length_straight', 'angle', 'k0_from_h', 'edge_entry_model',
                   'edge_exit_model', 'rbend_angle_diff', 'rbend_model', 'k0']:
            if nn in kwargs:
                to_be_set_with_properties.append((nn, kwargs.pop(nn)))

        _HasKnlKsl.__init__(self, **kwargs) # Handles knl, ksl, order, model, integrator

        for nn, val in to_be_set_with_properties:
            setattr(self, nn, val)

    @classmethod
    def from_dict(cls, dct, **kwargs):

        dct = dct.copy()

        if 'length' in dct:
            assert 'length_straight' in dct
            dct.pop('length')

        return super().from_dict(dct, **kwargs)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        raise RuntimeError("Setting `length` directly is not allowed for RBend. "
                           "Set `length_straight` instead.")

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self._update_rbend_h_length_k0()

    @property
    def rbend_angle_diff(self):
        return self._rbend_angle_diff

    @rbend_angle_diff.setter
    def rbend_angle_diff(self, value):
        self._rbend_angle_diff = value
        self._update_rbend_h_length_k0()

    @property
    def length_straight(self):
        return self._length_straight

    @length_straight.setter
    def length_straight(self, value):
        self._length_straight = value
        self._update_rbend_h_length_k0()

    def _update_rbend_h_length_k0(self):
        _angle = self._angle
        _length_straight = self._length_straight
        _rbend_angle_diff = self._rbend_angle_diff

        theta_in = 0.5 * _angle - _rbend_angle_diff / 2
        theta_out = 0.5 * _angle + _rbend_angle_diff / 2
        if abs(_angle) < 1e-10:
            length = _length_straight
            h = 0
        elif abs(_length_straight) < 1e-10:
            length = 0.0
            h = 0
        else:
            h = (np.sin(theta_in) + np.sin(theta_out)) / _length_straight
            length = _angle / h

        self._h = h
        self._angle = _angle
        self._length = length
        if self.k0_from_h:
            self._k0 = self._h

    @property
    def rbend_model(self):
        return _INDEX_TO_RBEND_MODEL[self._rbend_model]

    @rbend_model.setter
    def rbend_model(self, value):
        try:
            self._rbend_model = _RBEND_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid rbend_model: {value}')

    @property
    def rbend_compensate_sagitta(self):
        return bool(self._rbend_compensate_sagitta)

    @rbend_compensate_sagitta.setter
    def rbend_compensate_sagitta(self, value):
        self._rbend_compensate_sagitta = int(bool(value))

    @property
    def hxl(self): return self.h * self.length

    @property
    def _angle_in(self):
        return 0.5 * self.angle - self._rbend_angle_diff / 2

    @property
    def _angle_out(self):
        return 0.5 * self.angle + self._rbend_angle_diff / 2

    @property
    def _x0_mid(self):
        out = -self.rbend_shift
        if abs(self.angle) > 1e-10 and self.rbend_compensate_sagitta:
            out += 0.5 / self.h * (1 - np.cos(self.angle / 2))
        return out

    @property
    def _x0_in(self):
        out = self._x0_mid
        if abs(self.angle) > 1e-10:
            px0_in = np.sin(self._angle_in)
            px0_mid = px0_in - self.h * self.length_straight / 2
            sqrt_mid = np.sqrt(1 - px0_mid * px0_mid)
            cos_theta_in = np.cos(self._angle_in)
            out -= 1 / self.h * (sqrt_mid - cos_theta_in)
        return out

    @property
    def _x0_out(self):
        out = self._x0_mid
        if abs(self.angle) > 1e-10:
            px0_out = np.sin(self._angle_out)
            px0_mid = px0_out - self.h * self.length_straight / 2
            sqrt_mid = np.sqrt(1 - px0_mid * px0_mid)
            cos_theta_out = np.cos(self._angle_out)
            out += 1 / self.h * (cos_theta_out - sqrt_mid)
        return out

    @property
    def radiation_flag(self): return 0.0

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceRBend

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceRBend

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceRBend

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceRBendEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceRBendExit

    @property
    def _repr_fields(self):
        return ['length_straight', 'angle'] + super()._repr_fields

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)

        for kk in {'angle', 'length_straight'}:
            if f'_{kk}' in out:
                out.pop(f'_{kk}')
            out[kk] = getattr(self, kk)

        return out


class Sextupole(_HasKnlKsl, _HasIntegrator, _HasModelStraight, BeamElement):

    _docstring_start = \
    """
    Sextupole element.

    Parameters
    ----------
    k2 : float
        Strength of the sextupole component in m^-3.
    k2s : float
        Strength of the skew sextupole component in m^-3.
    length : float
        Length of the element in meters.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
               _HasModelStraight._for_docstring, _HasIntegrator._for_docstring,
               _for_docstring_edge_straight, _for_docstring_alignment, '\n',
               _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields={
        'k2': xo.Float64,
        'k2s': xo.Float64,
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'edge_entry_active': xo.Field(xo.UInt64, default=False),
        'edge_exit_active': xo.Field(xo.UInt64, default=False),
        'num_multipole_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include <beam_elements/elements_src/sextupole.h>',
    ]

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceSextupole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceSextupole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceSextupole

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceSextupoleEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceSextupoleExit


class Octupole(_HasKnlKsl, _HasIntegrator, _HasModelStraight, BeamElement):

    _docstring_start = \
    """
    Octupole element.

    Parameters
    ----------
    k3 : float
        Strength of the octupole component in m^-4.
    k3s : float
        Strength of the skew octupole component in m^-4.
    length : float
        Length of the element in meters.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
               _HasModelStraight._for_docstring, _HasIntegrator._for_docstring,
               _for_docstring_edge_straight, _for_docstring_alignment, '\n',
               _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields={
        'k3': xo.Float64,
        'k3s': xo.Float64,
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'edge_entry_active': xo.Field(xo.UInt64, default=False),
        'edge_exit_active': xo.Field(xo.UInt64, default=False),
        'num_multipole_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include <beam_elements/elements_src/octupole.h>',
    ]

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceOctupole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceOctupole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceOctupole

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceOctupoleEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceOctupoleExit


class Quadrupole(_HasKnlKsl, _HasIntegrator, _HasModelStraight, BeamElement):

    _docstring_start = \
    """
    Quadrupole element.

    Parameters
    ----------
    k1 : float
        Strength of the quadrupole component in m^-2.
    k1s : float
        Strength of the skew quadrupole component in m^-2.
    length : float
        Length of the element in meters.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
               _HasModelStraight._for_docstring, _HasIntegrator._for_docstring,
               _for_docstring_edge_straight, _for_docstring_alignment, '\n',
               _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'k1': xo.Float64,
        'k1s': xo.Float64,
        'length': xo.Float64,
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'edge_entry_active': xo.Field(xo.UInt64, default=False),
        'edge_exit_active': xo.Field(xo.UInt64, default=False),
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _extra_c_sources = [
        '#include <beam_elements/elements_src/quadrupole.h>',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    @property
    def radiation_flag(self): return 0.0

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceQuadrupole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceQuadrupole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceQuadrupole

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceQuadrupoleEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceQuadrupoleExit


class UniformSolenoid(_HasKnlKsl, _HasIntegrator, BeamElement):

    _docstring_start = \
    """
    Uniform solenoid element with hard-edge fringe field. The axis of the
    solenoid is assumed parallel to the `s` axis. Radiation and spin
    precession are take place only in the solenoid body (no radiation and
    precession in the fringe field).

    Parameters
    ----------
    ks : float
        Strength of the solenoid component (defined as B_s / reference_rigidity)
    length : float
        Length of the element in meters.
    x0 : float, optional
        Horizontal offset of the solenoid center in meters. Defaults to 0.
    y0 : float, optional
        Vertical offset of the solenoid center in meters. Defaults to 0.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
            _HasIntegrator._for_docstring, _for_docstring_edge_straight,
            _for_docstring_alignment, '\n', _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields={
        'ks': xo.Float64,
        'length': xo.Float64,
        'x0': xo.Float64,
        'y0': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'edge_entry_active': xo.Field(xo.UInt64, default=True),
        'edge_exit_active': xo.Field(xo.UInt64, default=True),
        'num_multipole_kicks': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include <beam_elements/elements_src/slnd.h>',
    ]

    def __init__(self, **kwargs):

        if 'model' in kwargs:
            raise ValueError("`model` is not supported for UniformSolenoid.")

        _HasKnlKsl.__init__(self, **kwargs)

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceUniformSolenoid

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceUniformSolenoidEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceUniformSolenoidExit

class VariableSolenoid(_HasKnlKsl, _HasIntegrator, BeamElement):

    _docstring_start = \
    """
    Solenoid with linearly varying lingitudinal field. The transverse fields
    arising form the derivative of the longitudinal fields are taken into account
    in particle dynamics, radiation, spin precession.

    Parameters
    ----------
    ks_profile : array-like of 2 floats
        Solenoid strength at entry and exit of the element (defined as
        B_s / reference_rigidity).
    length : float
        Length of the element in meters along the reference trajectory.
    x0 : float, optional
        Horizontal offset of the solenoid center in meters. Defaults to 0.
    y0 : float, optional
        Vertical offset of the solenoid center in meters. Defaults to 0.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
        _HasIntegrator._for_docstring, _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True

    _xofields={
        'ks_profile': xo.Float64[2],
        'length': xo.Float64,
        'x0': xo.Float64,
        'y0': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'num_multipole_kicks': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include <beam_elements/elements_src/variable_solenoid.h>',
    ]

    def __init__(self, **kwargs):

        if 'model' in kwargs:
            raise ValueError("`model` is not supported for UniformSolenoid.")

        _HasKnlKsl.__init__(self, **kwargs)

class TempRF(_HasKnlKsl, _HasModelRF, _HasIntegrator, BeamElement):

    isthick = True
    has_backtrack = True

    _xofields = {
        'frequency': xo.Float64,
        'voltage': xo.Float64,
        'lag': xo.Float64,
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'pn': xo.Float64[:],
        'ps': xo.Float64[:],
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
    }

    _noexpr_fields = _NOEXPR_FIELDS

    _extra_c_sources = [
        '#include <beam_elements/elements_src/temprf.h>',
    ]


class Solenoid(_HasKnlKsl, BeamElement):
    """Solenoid element.

    Parameters
    ----------
    length : float
        Length of the element in meters.
    ks : float
        Strength of the solenoid component in rad / m.
    ksi : float
        Integrated strength of the solenoid component in rad. Only to be
        specified when the element is thin, i.e. when `length` is 0.
    order : int, optional
        Maximum order of multipole expansion for this magnet. Defaults to 5.
    knl : list of floats, optional
        Normal multipole integrated strengths. If not provided, defaults to zeroes.
    ksl : list of floats, optional
        Skew multipole integrated strengths. If not provided, defaults to zeroes.
    num_multipole_kicks : int, optional
        The number of kicks to be used in thin kick splitting. The default value
        of zero implies a single kick in the middle of the element.
    radiation_flag : int, optional
        Whether to enable radiation. See ``Magnet`` for details.
    mult_rot_x_rad : float, optional
        Rotation around the x-axis of the embedded multipolar field, in radians.
    mult_rot_y_rad : float, optional
        Rotation around the y-axis of the embedded multipolar field, in radians.
    mult_shift_x : float, optional
        Offset of the embedded multipolar field along the x-axis, in metres.
    mult_shift_y : float, optional
        Offset of the embedded multipolar field along the y-axis, in metres.
    mult_shift_s : float, optional
        Offset of the embedded multipolar field along s, in metres.
    """
    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'length': xo.Float64,
        'ks': xo.Float64,
        'ksi': xo.Float64,
        'radiation_flag': xo.Int64,
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'mult_rot_x_rad': xo.Float64,
        'mult_rot_y_rad': xo.Float64,
        'mult_shift_x': xo.Float64,
        'mult_shift_y': xo.Float64,
        'mult_shift_s': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/legacy_solenoid.h>',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self, order=None, knl: List[float] = None, ksl: List[float] = None, **kwargs):
        warn(
            'The `Solenoid` element is deprecated. Use `VariableSolenoid` or `UniformSolenoid` instead.',
            FutureWarning
        )

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if kwargs.get('ksi', 0) != 0:
            # Fail when trying to create a thin solenoid, as these are not
            # tested yet
            raise NotImplementedError('Thin solenoids are not implemented yet.')
            # self.isthick = False

        if kwargs.get('ksi') and kwargs.get('length'):
            raise ValueError(
                "The parameter `ksi` can only be specified when `length` == 0."
            )

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = self._prepare_multipolar_params(order, knl=knl, ksl=ksl)
        kwargs.update(multipolar_kwargs)

        self.xoinitialize(**kwargs)


class Magnet(_BendCommon, BeamElement):
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
        'k0_from_h': xo.Field(xo.UInt64, default=1),
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
        '#include <beam_elements/elements_src/magnet.h>',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if 'h' in kwargs:
            raise ValueError("Setting `h` directly is not allowed. "
                                "Set `length` and `angle` instead.")

        to_be_set_with_properties = []
        for nn in ['length', 'angle', 'k0_from_h', 'edge_entry_model',
                   'edge_exit_model', 'k0']:
            if nn in kwargs:
                to_be_set_with_properties.append((nn, kwargs.pop(nn)))

        _HasKnlKsl.__init__(self, **kwargs)

        for nn, val in to_be_set_with_properties:
            setattr(self, nn, val)

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


class MagnetEdge(_HasKnlKsl, BeamElement):
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
        '#include <beam_elements/elements_src/magnet_edge.h>',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

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
        k_multipolar_kwargs = self._prepare_multipolar_params(
            k_order, skip_factorial=True, order_name='k_order', kn=kn, ks=ks)
        kwargs.update(k_multipolar_kwargs)

        kl_order = kwargs.pop('kl_order', -1)
        knl, ksl = kwargs.pop('knl', []), kwargs.pop('ksl', [])
        kl_multipolar_kwargs = self._prepare_multipolar_params(
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


class CombinedFunctionMagnet:

    def __init__(self, *args, **kwargs):
        raise TypeError('`CombinedFunctionMagnet` is supported anymore. '
                        'Use `Bend` instead.')

    @classmethod
    def from_dict(cls, dct):
        return Bend(**dct)


class DipoleFringe(BeamElement):
    """Fringe field element of a dipole.

    Parameters
    ----------
    fint : float
        Fringe field integral in units of m^-1.
    hgap : float
        Half gap in units of m.
    k : float
        Normalized integrated strength of the normal component in units of 1/m.
    """

    _xofields = {
        'fint': xo.Float64,
        'hgap': xo.Float64,
        'k': xo.Float64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/dipole_fringe.h>',
    ]

    def __init__(self, **kwargs):
        raise NotImplementedError


class Wedge(BeamElement):
    """Wedge field element.

    Parameters
    ----------
    angle : float
        Angle of the wedge in radians.
    k : float
        Normalized integrated strength of the normal component in units of 1/m.
    """

    _xofields = {
        'angle': xo.Float64,
        'k': xo.Float64,
        'k1': xo.Float64,
        'quad_wedge_then_dip_wedge': xo.Int64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/wedge.h>',
    ]


class SimpleThinBend(BeamElement):

    """A specialized version of Multipole to model a thin bend (ksl, hyl are all zero).

    Parameters
    ----------
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length 1.
    hxl : float
        Rotation angle of the reference trajectory in the horizontal plane in
        radians. Default is ``0``.
    length : float
        Length of the originating thick bend. Default is ``0``.
    """

    _xofields={
        'knl': xo.Float64[1],
        'hxl': xo.Float64,
        'length': xo.Float64,
    }

    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include <beam_elements/elements_src/simplethinbend.h>',
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return
        knl = kwargs.get('knl')
        if knl is not None and len(knl) != 1:
            raise ValueError("For a simple thin bend, len(knl) must be 1.")

        super().__init__(**kwargs)

    @property
    def radiation_flag(self): return 0.0

    @property
    def order(self): return 0

    @property
    def inv_factorial_order(self): return 1.0

    @property
    def ksl(self): return self._buffer.context.linked_array_type.from_array(
        np.array([0., 0.]),
        mode='readonly',
        container=self,
    )


class RFMultipole(_HasKnlKsl, BeamElement):

    _docstring_start = \
    """Beam element modeling a thin modulated multipole, with strengths
    dependent on the z coordinate:

    Parameters
    ----------
    frequency : float
        Frequency in Hertz. Default is ``0``.
    knl : array
        Integrated strength of the normal rf-multipole components in units of m^-n.
    ksl : array
        Integrated strength of the skew rf-multipole components in units of m^-n.
    order : int
        Order of the multipole. If not provided, it will be inferred from knl and/or ksl.
    pn : array
        Phase of the normal components in degrees.
    ps : array
        Phase of the skew components in degrees.
    voltage : float
        Longitudinal voltage. Default is ``0``.
    lag : float
        Longitudinal phase seen by the reference particle. Default is ``0``.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _for_docstring_alignment, '\n',
                             _docstring_general_notes, '\n\n'])

    _xofields={
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'pn': xo.Float64[:],
        'ps': xo.Float64[:],
        'absolute_time': xo.Int64,
    }

    has_backtrack = True
    allow_loss_refinement = True

    _extra_c_sources = [
        '#include <beam_elements/elements_src/rfmultipole.h>',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
    }


class DipoleEdge(BeamElement):
    """Beam element modeling a dipole edge (see MAD-X manual for detaild description).

    Parameters
    ----------
    k : float
        Strength in 1/m.
    e1 : float
        Face angle in rad.
    hgap : float
        Equivalent gap in m.
    fint : float
        Fringe integral.
    e1_fd : float
        Term added to e1 only for the linear mode and only in the vertical
        plane to account for non-zero angle in the closed orbit when entering
        the fringe field (feed down effect).
    model : str
        Model to be used for the edge. It can be 'linear', 'full' or 'suppress'.
        Default is 'linear'.
    side : str
        Side of the bend on which the edge is located. It can be 'entry' or
        'exit'. Default is 'entry'.
    """

    _xofields = {
        'r21': xo.Float64,
        'r43': xo.Float64,
        'hgap': xo.Float64,
        'k': xo.Float64,
        'e1': xo.Float64,
        'e1_fd': xo.Float64,
        'fint': xo.Float64,
        'model': xo.Int64,
        'side': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/dipoleedge.h>',
    ]

    has_backtrack = True

    _rename = {
        'r21': '_r21',
        'r43': '_r43',
        'hgap': '_hgap',
        'k': '_k',
        'e1': '_e1',
        'e1_fd': '_e1_fd',
        'fint': '_fint',
        'model': '_model',
        'side': '_side',
    }

    def __init__(
        self,
        k=None,
        e1=None,
        e1_fd=None,
        hgap=None,
        fint=None,
        model=None,
        side=None,
        **kwargs
    ):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        # For backward compatibility
        if 'h' in kwargs.keys():
            assert k is None
            k = kwargs.pop('h')
        if '_h' in kwargs.keys():
            kwargs['_k'] = kwargs.pop('_h')

        self.xoinitialize(**kwargs)

        if hgap is not None:
            self._hgap = hgap
        if k is not None:
            self._k = k
        if e1 is not None:
            self._e1 = e1
        if e1_fd is not None:
            self._e1_fd = e1_fd
        if fint is not None:
            self._fint = fint
        if model is not None:
            self.model = model
        if side is not None:
            self.side = side

        self._update_r21_r43()

    def to_dict(self, copy_to_cpu=True):
        scalar_fields = ['k', 'e1', 'e1_fd', 'hgap', 'fint']

        out = {'__class__': type(self).__name__}

        for field_name in scalar_fields:
            value = getattr(self, field_name)
            if not np.isclose(value, 0, atol=1e-16):
                out[field_name] = value

        if self._model != 0:
            out['model'] = self.model

        if self._side != 0:
            out['side'] = self.side

        return out

    def _update_r21_r43(self):
        corr = np.float64(2.0) * self.k * self.hgap * self.fint
        r21 = self.k * np.tan(self.e1)
        e1_v = self.e1 + self.e1_fd
        temp = corr / np.cos(e1_v) * (
            np.float64(1) + np.sin(e1_v) * np.sin(e1_v))
        r43 = -self.k * np.tan(e1_v - temp)
        self._r21 = r21
        self._r43 = r43

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self._update_r21_r43()

    @property
    def e1(self):
        return self._e1

    @e1.setter
    def e1(self, value):
        self._e1 = value
        self._update_r21_r43()

    @property
    def e1_fd(self):
        return self._e1_fd

    @e1_fd.setter
    def e1_fd(self, value):
        self._e1_fd = value
        self._update_r21_r43()

    @property
    def hgap(self):
        return self._hgap

    @hgap.setter
    def hgap(self, value):
        self._hgap = value
        self._update_r21_r43()

    @property
    def fint(self):
        return self._fint

    @fint.setter
    def fint(self, value):
        self._fint = value
        self._update_r21_r43()

    @property
    def r21(self):
        return self._r21

    @property
    def r43(self):
        return self._r43

    @property
    def model(self):
        return {
            0: 'linear',
            1: 'full',
           -1: 'suppressed',
        }[self._model]

    @model.setter
    def model(self, value):
        assert value in ['linear', 'full', 'suppressed']
        self._model = {
            'linear': 0,
            'full': 1,
            'suppressed': -1,
        }[value]

    @property
    def side(self):
        return {
            0: 'entry',
            1: 'exit',
        }[self._side]

    @side.setter
    def side(self, value):
        assert value in ['entry', 'exit']
        self._side = {
            'entry': 0,
            'exit': 1,
        }[value]


class MultipoleEdge(_HasKnlKsl, BeamElement):
    """Beam element modelling a mulipole edge.

    Parameters
    ----------
    kn: float
        Normalized integrated strength of the normal component in units of 1/m.
    ks: float
        Normalized integrated strength of the skew component in units of 1/m.
    is_exit: bool
        Flag to indicate if the edge is at the exit of the element.
    order: int
        Order of the multipole, corresponds to the length of ``kn`` and ``ks``.
    """
    _xofields = {
        'kn': xo.Float64[:],
        'ks': xo.Float64[:],
        'is_exit': xo.Int64,
        'order': xo.Int64,
    }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/multipoleedge.h>',
    ]

    def __init__(self, kn: list=None, ks: list=None, is_exit=False, order=None, _xobject=None, **kwargs):
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        multipole_kwargs = self._prepare_multipolar_params(order,
                                            skip_factorial=True, kn=kn, ks=ks)

        self.xoinitialize(is_exit=is_exit, **kwargs, **multipole_kwargs)


class LineSegmentMap(BeamElement):

    _xofields={
        'length': xo.Float64,

        'qx': xo.Float64,
        'qy': xo.Float64,

        'coeffs_dqx': xo.Float64[:],
        'coeffs_dqy': xo.Float64[:],
        'det_xx': xo.Float64,
        'det_xy': xo.Float64,
        'det_yy': xo.Float64,
        'det_yx': xo.Float64,

        'betx': xo.Float64[2],
        'bety': xo.Float64[2],
        'alfx': xo.Float64[2],
        'alfy': xo.Float64[2],

        'dx': xo.Float64[2],
        'dpx': xo.Float64[2],
        'dy': xo.Float64[2],
        'dpy': xo.Float64[2],

        'x_ref': xo.Float64[2],
        'px_ref': xo.Float64[2],
        'y_ref': xo.Float64[2],
        'py_ref': xo.Float64[2],

        'energy_ref_increment': xo.Float64,
        'energy_increment': xo.Float64,
        'uncorrelated_rad_damping': xo.Int64,
        'correlated_rad_damping': xo.Int64,
        'damping_factors':xo.Float64[6,6],
        'uncorrelated_gauss_noise': xo.Int64,
        'correlated_gauss_noise': xo.Int64,
        'gauss_noise_matrix':xo.Float64[6,6],

        'longitudinal_mode_flag': xo.Int64,
        'qs': xo.Float64,
        'bets': xo.Float64,
        'bucket_length': xo.Float64,
        'momentum_compaction_factor': xo.Float64,
        'slippage_length': xo.Float64,
        'voltage_rf': xo.Float64[:],
        'frequency_rf': xo.Float64[:],
        'lag_rf': xo.Float64[:],
    }

    _depends_on = [RandomNormal]
    isthick = True

    # _rename = {
    #     'cos_s': '_cos_s',
    #     'sin_s': '_sin_s',
    #     'bets': '_bets',
    #     'longitudinal_mode_flag': '_longitudinal_mode_flag',
    # }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/linesegmentmap.h>',
    ]

    def __init__(self, length=0., qx=0, qy=0,
            betx=1., bety=1., alfx=0., alfy=0.,
            dx=0., dpx=0., dy=0., dpy=0.,
            x_ref=0.0, px_ref=0.0, y_ref=0.0, py_ref=0.0,
            longitudinal_mode=None,
            qs=None, bets=None,bucket_length=None,
            momentum_compaction_factor=None,
            slippage_length=None,
            voltage_rf=None, frequency_rf=None, lag_rf=None,
            dqx=0.0, dqy=0.0, ddqx=0.0, ddqy=0.0, dnqx=None, dnqy=None,
            det_xx=0.0, det_xy=0.0, det_yy=0.0, det_yx=0.0,
            energy_increment=0.0, energy_ref_increment=0.0,
            damping_rate_x = 0.0, damping_rate_px = 0.0,
            damping_rate_y = 0.0, damping_rate_py = 0.0,
            damping_rate_zeta = 0.0, damping_rate_pzeta = 0.0,
            gauss_noise_ampl_x=0.0,gauss_noise_ampl_px=0.0,
            gauss_noise_ampl_y=0.0,gauss_noise_ampl_py=0.0,
            gauss_noise_ampl_zeta=0.0,gauss_noise_ampl_pzeta=0.0,
            damping_matrix=None,gauss_noise_matrix=None,
            **nargs):

        '''
        Map representing a simplified segment of a beamline.

        Parameters
        ----------
        length : float
            Length of the segment in meters.
        qx : float
            Horizontal tune or phase advance of the segment.
        qy : float
            Vertical tune or phase advance of the segment.
        betx : tuple of length 2 or float
            Horizontal beta function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        bety : tuple of length 2 or float
            Vertical beta function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        alfx : tuple of length 2 or float
            Horizontal alpha function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        alfy : tuple of length 2 or float
            Vertical alpha function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dx : tuple of length 2 or float
            Horizontal dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dpx : tuple of length 2 or float
            Px dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dy : tuple of length 2 or float
            Vertical dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dpy : tuple of length 2 or float
            Py dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        x_ref : tuple of length 2 or float
            Horizontal position of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        px_ref : tuple of length 2 or float
            Px coordinate of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        y_ref : tuple of length 2 or float
            Vertical position of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        py_ref : tuple of length 2 or float
            Py coordinate of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        longitudinal_mode : str
            Longitudinal mode of the segment. Can be one of ``'linear_fixed_qs'``,
            ``'nonlinear'``, ``'linear_fixed_rf'`` or ``'frozen'``.
        qs : float
            Synchrotron tune of the segment. Only used if ``longitudinal_mode``
            is ``'linear_fixed_qs'``.
        bets : float
            Synchrotron beta function of the segment (positive above transition,
            negative below transition). Only used if ``longitudinal_mode``
            is ``'linear_fixed_qs'``.
        bucket_length : float
            The linear RF force becomes a sawtooth with a fixed point every
            bucket_length [full length in seconds]. Only used if
            ``longitudinal_mode`` is ``'linear_fixed_qs'``.
        momentum_compaction_factor : float
            Momentum compaction factor of the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        slippage_length : float
            Slippage length of the segment. Only used if ``longitudinal_mode``
            is ``'nonlinear'`` or ``'linear_fixed_rf'``. If not given, the
            ``length`` of the segment is used.
        voltage_rf : list of float
            List of voltages of the RF kicks in the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        frequency_rf : list of float
            List of frequencies of the RF kicks in the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        lag_rf : list of float
            List of lag of the RF kicks in the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        dqx : float or list of float
            Horizontal linear chromaticity of the segment.
        dqy : float or list of float
            Vertical linear chromaticity of the segment.
        ddqx: float
            Horizontal second order chromaticity of the segment
        ddqy: float
            Vertical second order chromaticity of the segment
        dnqx: list of float
            List of horizontal chromaticities up to any order. The first element
            of the list is the horizontal tune, the second element is the
            horizontal linear chromaticity, the third element the horizontal
            second order chromaticity and so on. It can be specified only if the
            horizontal tune, and chromaticities are not specified.
        dnqy: list of float
            List of vertical chromaticities up to any order. The first element
            of the list is the vertical tune, the second element is the
            vertical linear chromaticity, the third element the vertical
            second order chromaticity and so on. It can be specified only if the
            vertical tune, and chromaticities are not specified.
        det_xx : float
            Anharmonicity xx coefficient (i.e. dqx / dJx, where Jx is the horizontal
            action). Optional, default is ``0``.
        det_xy : float
            Anharmonicity xy coefficient (i.e. dqx / dJy, where Jy is the vertical
            action). Optional, default is ``0``.
        det_yx : float
            Anharmonicity yx coefficient (i.e. dqy / dJx, where Jx is the horizontal
            action). Optional, default is ``0``.
        det_yy : float
            Anharmonicity yy coefficient (i.e. dqy / dJy, where Jy is the vertical
            action). Optional, default is ``0``.
        energy_increment : float
            Energy increment of the segment in eV.
        energy_ref_increment : float
            Increment of the reference energy in eV.
        damping_rate_x : float
            Damping rate of the horizontal position
            x_n+1 = (1-damping_rate_x)*x_n. Optional, default is ``0``.
        damping_rate_px : float
            Damping rate of the horizontal momentum
            px_n+1 = (1-damping_rate_px)*px_n. Optional, default is ``0``.
        damping_rate_y : float
            Damping rate of the vertical position
            y_n+1 = (1-damping_rate_y)*y_n. Optional, default is ``0``.
        damping_rate_py : float
            Damping rate of the vertical momentum
            px_n+1 = (1-damping_rate_x)*py_n. Optional, default is ``0``.
        damping_rate_z : float
            Damping rate of the longitudinal position
            z_n+1 = (1-damping_rate_z)*z_n. Optional, default is ``0``.
        damping_rate_pzeta : float
            Damping rate on the momentum
            pzeta_n+1 = (1-damping_rate_pzeta)*pzeta_n. Optional, default is ``0``.
        gauss_noise_ampl_x : float
            Amplitude of Gaussian noise on the horizontal position. Optional, default is ``0``.
        gauss_noise_ampl_px : float
            Amplitude of Gaussian noise on the horizontal momentum. Optional, default is ``0``.
        gauss_noise_ampl_y : float
            Amplitude of Gaussian noise on the vertical position. Optional, default is ``0``.
        gauss_noise_ampl_py : float
            Amplitude of Gaussian noise on the vertical momentum. Optional, default is ``0``.
        gauss_noise_ampl_zeta : float
            Amplitude of Gaussian noise on the longitudinal position. Optional, default is ``0``.
        gauss_noise_ampl_pzeta : float
            Amplitude of Gaussian noise on the longitudinal momentum. Optional, default is ``0``.
        damping_matrix : float[6,6]
            Matrix of damping: Each paticles coordinate vector (x,px,y,py,zeta,pzeta) is multiplied
            by the identity + the damping matrix. Incompatible with inputs damping_rate_*.
            Optional, default is ``None``
        gauss_noise_matrix : float[6,6]
            Covariance matrix of the Gaussian noise applied in (x,px,y,py,zeta,pzeta).
            Incompatible with inputs gauss_noise_ampl_*. Optional, default is ``None``
        '''

        if '_xobject' in nargs.keys() and nargs['_xobject'] is not None:
            self._xobject = nargs['_xobject']
            return

        assert longitudinal_mode in [
            'linear_fixed_qs', 'nonlinear', 'linear_fixed_rf', 'frozen', None]

        if dnqx is not None:
            assert qx == 0 and dqx == 0 and ddqx == 0
            qx = dnqx[0]
        else:
            dnqx = [qx]
            if dqx != 0:
                dnqx.append(dqx)
            if ddqx != 0:
                dnqx.append(ddqx)

        if dnqy is not None:
            assert qy == 0 and dqy == 0 and ddqy == 0
            qy = dnqy[0]
        else:
            dnqy = [qy]
            if dqy != 0:
                dnqy.append(dqy)
            if ddqy != 0:
                dnqy.append(ddqy)

        coeffs_dqx = [dnqx[i] / float(factorial(i)) for i in range(len(dnqx))]
        coeffs_dqy = [dnqy[i] / float(factorial(i)) for i in range(len(dnqy))]

        nargs['qx'] = qx
        nargs['qy'] = qy
        nargs['coeffs_dqx'] = coeffs_dqx
        nargs['coeffs_dqy'] = coeffs_dqy
        nargs['det_xx'] = det_xx
        nargs['det_xy'] = det_xy
        nargs['det_yy'] = det_yy
        nargs['det_yx'] = det_yx
        nargs['length'] = length

        if longitudinal_mode is None:
            if qs is not None:
                longitudinal_mode = 'linear_fixed_qs'
            elif voltage_rf is not None:
                longitudinal_mode = 'nonlinear'
            else:
                longitudinal_mode = 'frozen'

        if longitudinal_mode == 'linear_fixed_qs':
            assert qs is not None
            assert bets is not None
            assert momentum_compaction_factor is None
            assert voltage_rf is None
            assert frequency_rf is None
            assert lag_rf is None
            if bucket_length == None:
                bucket_length = -1.0
            nargs['longitudinal_mode_flag'] = 1
            nargs['qs'] = qs
            nargs['bets'] = bets
            nargs['bucket_length'] = bucket_length
            nargs['voltage_rf'] = [0]
            nargs['frequency_rf'] = [0]
            nargs['lag_rf'] = [0]
        elif longitudinal_mode == 'nonlinear' or longitudinal_mode == 'linear_fixed_rf':
            assert voltage_rf is not None
            assert frequency_rf is not None
            assert lag_rf is not None
            assert momentum_compaction_factor is not None
            assert qs is None
            assert bets is None
            assert bucket_length is None

            if slippage_length is None:
                nargs['slippage_length'] = length
            else:
                nargs['slippage_length'] = slippage_length

            if longitudinal_mode == 'nonlinear':
                nargs['longitudinal_mode_flag'] = 2
            elif longitudinal_mode == 'linear_fixed_rf':
                nargs['longitudinal_mode_flag'] = 3

            nargs['voltage_rf'] = voltage_rf
            nargs['frequency_rf'] = frequency_rf
            nargs['lag_rf'] = lag_rf
            nargs['momentum_compaction_factor'] = momentum_compaction_factor
            for nn in ['frequency_rf', 'lag_rf', 'voltage_rf']:
                if np.isscalar(nargs[nn]):
                    nargs[nn] = [nargs[nn]]

            assert (len(nargs['frequency_rf'])
                    == len(nargs['lag_rf'])
                    == len(nargs['voltage_rf']))

            if longitudinal_mode == 'linear_fixed_rf':
                assert len(nargs['frequency_rf']) == 1

        elif longitudinal_mode == 'frozen':
            nargs['longitudinal_mode_flag'] = 0
            nargs['voltage_rf'] = [0]
            nargs['frequency_rf'] = [0]
            nargs['lag_rf'] = [0]
        else:
            raise ValueError('longitudinal_mode must be one of "linear_fixed_qs", "nonlinear" or "frozen"')


        if np.isscalar(betx): betx = [betx, betx]
        else: assert len(betx) == 2

        if np.isscalar(bety): bety = [bety, bety]
        else: assert len(bety) == 2

        if np.isscalar(alfx): alfx = [alfx, alfx]
        else: assert len(alfx) == 2

        if np.isscalar(alfy): alfy = [alfy, alfy]
        else: assert len(alfy) == 2

        if np.isscalar(dx): dx = [dx, dx]
        else: assert len(dx) == 2

        if np.isscalar(dpx): dpx = [dpx, dpx]
        else: assert len(dpx) == 2

        if np.isscalar(dy): dy = [dy, dy]
        else: assert len(dy) == 2

        if np.isscalar(dpy): dpy = [dpy, dpy]
        else: assert len(dpy) == 2

        if np.isscalar(x_ref): x_ref = [x_ref, x_ref]
        else: assert len(x_ref) == 2

        if np.isscalar(px_ref): px_ref = [px_ref, px_ref]
        else: assert len(px_ref) == 2

        if np.isscalar(y_ref): y_ref = [y_ref, y_ref]
        else: assert len(y_ref) == 2

        if np.isscalar(py_ref): py_ref = [py_ref, py_ref]
        else: assert len(py_ref) == 2

        nargs['betx'] = betx
        nargs['bety'] = bety
        nargs['alfx'] = alfx
        nargs['alfy'] = alfy
        nargs['dx'] = dx
        nargs['dpx'] = dpx
        nargs['dy'] = dy
        nargs['dpy'] = dpy
        nargs['x_ref'] = x_ref
        nargs['px_ref'] = px_ref
        nargs['y_ref'] = y_ref
        nargs['py_ref'] = py_ref

        # acceleration with change of reference momentum
        nargs['energy_ref_increment'] = energy_ref_increment
        # acceleration without change of reference momentum
        nargs['energy_increment'] = energy_increment


        assert damping_rate_x >= 0.0
        assert damping_rate_px >= 0.0
        assert damping_rate_y >= 0.0
        assert damping_rate_py >= 0.0
        assert damping_rate_zeta >= 0.0
        assert damping_rate_pzeta >= 0.0
        
        if (damping_rate_x > 0.0 or damping_rate_px > 0.0
                or damping_rate_y > 0.0 or damping_rate_py > 0.0 
                or damping_rate_zeta > 0.0 or damping_rate_pzeta > 0.0):
            assert damping_matrix is None
            nargs['uncorrelated_rad_damping'] = True
            nargs['correlated_rad_damping'] = False
            nargs['damping_factors'] = np.identity(6,dtype=float)
            nargs['damping_factors'][0,0] -= damping_rate_x
            nargs['damping_factors'][1,1] -= damping_rate_px
            nargs['damping_factors'][2,2] -= damping_rate_y
            nargs['damping_factors'][3,3] -= damping_rate_py
            nargs['damping_factors'][4,4] -= damping_rate_zeta
            nargs['damping_factors'][5,5] -= damping_rate_pzeta
        elif damping_matrix is not None:
            assert np.shape(damping_matrix) == (6,6)
            nargs['correlated_rad_damping'] = True
            nargs['uncorrelated_rad_damping'] = False
            nargs['damping_factors'] = np.identity(6,dtype=float)+damping_matrix
        else:
            nargs['uncorrelated_rad_damping'] = False
            nargs['correlated_rad_damping'] = False

        assert gauss_noise_ampl_x >= 0.0
        assert gauss_noise_ampl_px >= 0.0
        assert gauss_noise_ampl_y >= 0.0
        assert gauss_noise_ampl_py >= 0.0
        assert gauss_noise_ampl_zeta >= 0.0
        assert gauss_noise_ampl_pzeta >= 0.0
        if (gauss_noise_ampl_x > 0 or gauss_noise_ampl_px > 0 or
                gauss_noise_ampl_y > 0 or gauss_noise_ampl_py > 0 or
                gauss_noise_ampl_zeta > 0 or gauss_noise_ampl_pzeta > 0):
            assert gauss_noise_matrix is None
            nargs['uncorrelated_gauss_noise'] = True
            nargs['correlated_gauss_noise'] = False
            nargs['gauss_noise_matrix'] = np.zeros((6,6),dtype=float)
            nargs['gauss_noise_matrix'][0,0] = gauss_noise_ampl_x
            nargs['gauss_noise_matrix'][1,1] = gauss_noise_ampl_px
            nargs['gauss_noise_matrix'][2,2] = gauss_noise_ampl_y
            nargs['gauss_noise_matrix'][3,3] = gauss_noise_ampl_py
            nargs['gauss_noise_matrix'][4,4] = gauss_noise_ampl_zeta
            nargs['gauss_noise_matrix'][5,5] = gauss_noise_ampl_pzeta
        elif gauss_noise_matrix is not None:
            nargs['correlated_gauss_noise'] = True
            nargs['uncorrelated_gauss_noise'] = False
            assert np.shape(gauss_noise_matrix) == (6,6)
            (u, s, vh) = np.linalg.svd(gauss_noise_matrix)
            nargs['gauss_noise_matrix'] = u*np.sqrt(s)
        else:
            nargs['uncorrelated_gauss_noise'] = False
            nargs['correlated_gauss_noise'] = False
        super().__init__(**nargs)

    @property
    def longitudinal_mode(self):
        ret = {
            0: 'frozen',
            1: 'linear_fixed_qs',
            2: 'nonlinear',
            3: 'linear_fixed_rf'
        }[self.longitudinal_mode_flag]
        return ret


class FirstOrderTaylorMap(BeamElement):
    """First order Taylor map.

    Parameters
    ----------
    length : float
        length of the element in meters.
    m0 : array_like
        6x1 array of the zero order Taylor map coefficients.
    m1 : array_like
        6x6 array of the first order Taylor map coefficients.
    """

    isthick = True

    _xofields = {
        'length': xo.Float64,
        'm0': xo.Field(xo.Float64[6], default=np.zeros(6, dtype=np.float64)),
        'm1': xo.Field(xo.Float64[6, 6], default=np.eye(6, dtype=np.float64)),
    }

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _extra_c_sources = [
        '#include <beam_elements/elements_src/firstordertaylormap.h>',
    ]

    _internal_record_class = SynchrotronRadiationRecord # not functional,
    # included for compatibility with Multipole


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

class SecondOrderTaylorMap(BeamElement):

    '''
    Implements the second order Taylor map:

       z_out[i] = k[i] + sum_j (R[i,j]*z_in[j]) + sum_jk (T[i,j,k]*z_in[j]*z_in[k])

       where z = (x, px, y, py, zeta, pzeta)

    Parameters
    ----------
    length : float
        length of the element in meters.
    k : array_like
        6x1 array of the zero order Taylor map coefficients.
    R : array_like
        6x6 array of the first order Taylor map coefficients.
    T : array_like
        6x6x6 array of the second order Taylor map coefficients.

    '''

    isthick = True

    _extra_c_sources = [
        '#include <beam_elements/elements_src/second_order_taylor_map.h>',
    ]

    _xofields={
        'k': xo.Float64[6],
        'R': xo.Float64[6,6],
        'T': xo.Float64[6,6,6],
        'length': xo.Float64
    }

    @classmethod
    def from_line(cls, line, start, end, twiss_table=None,
                  **kwargs):

        '''
        Generate a `SecondOrderTaylorMap` from a `Line` object.
        The coefficients are computed with finite differences around the closed
        orbit.

        Parameters
        ----------
        line : Line
            A `Line` object.
        start : str
            Name of the element where the map starts.
        end : str
            Name of the element where the map stops.
        twiss_table : TwissTable, optional
            A `TwissTable` object. If not given, it will be computed.

        Returns
        -------
        SecondOrderTaylorMap
            A `SecondOrderTaylorMap` object.

        '''
        if start == end:
            # start == end will lead to compute_one_turn_matrix_finite_differences() computing a
            # full one-turn response matrix (but here we would rather expect identity)
            raise NotImplementedError('end element must be after start element')

        if twiss_table is None:
            tw = line.twiss(reverse=False)
        else:
            tw = twiss_table

        twinit = tw.get_twiss_init(start)
        twinit_out = tw.get_twiss_init(end)

        RR = line.compute_one_turn_matrix_finite_differences(
            start=start, end=end, particle_on_co=twinit.particle_on_co
            )['R_matrix']
        TT = line.compute_T_matrix(start=start, end=end,
                                    particle_on_co=twinit.particle_on_co)

        x_co_in = np.array([
            twinit.particle_on_co.x[0],
            twinit.particle_on_co.px[0],
            twinit.particle_on_co.y[0],
            twinit.particle_on_co.py[0],
            twinit.particle_on_co.zeta[0],
            twinit.particle_on_co.pzeta[0],
        ])

        x_co_out = np.array([
            twinit_out.particle_on_co.x[0],
            twinit_out.particle_on_co.px[0],
            twinit_out.particle_on_co.y[0],
            twinit_out.particle_on_co.py[0],
            twinit_out.particle_on_co.zeta[0],
            twinit_out.particle_on_co.pzeta[0],
        ])

        # Handle feeddown (express the expansion in z instead of z - z_co)
        R_T_fd = np.einsum('ijk,k->ij', TT, x_co_in)
        K_T_fd = R_T_fd @ x_co_in

        K_hat = x_co_out - RR @ x_co_in + K_T_fd
        RR_hat = RR - 2 * R_T_fd

        smap = cls(R=RR_hat, T=TT, k=K_hat,
                   length=tw['s', end] - tw['s', start],
                   **kwargs)

        return smap

    def scale_coordinates(self, scale_x=1, scale_px=1, scale_y=1, scale_py=1,
                          scale_zeta=1, scale_pzeta=1):

        '''
        Generate a new `SecondOrderTaylorMap` with scaled coordinates.

        Parameters
        ----------
        scale_x : float
            Scaling factor for x.
        scale_px : float
            Scaling factor for px.
        scale_y : float
            Scaling factor for y.
        scale_py : float
            Scaling factor for py.
        scale_zeta : float
            Scaling factor for zeta.
        scale_pzeta : float
            Scaling factor for pzeta.

        Returns
        -------
        SecondOrderTaylorMap
            A new `SecondOrderTaylorMap` with scaled coordinates.

        '''

        out = self.copy()

        scale_factors = np.array(
            [scale_x, scale_px, scale_y, scale_py, scale_zeta, scale_pzeta])

        for ii in range(6):
            out.T[ii, :, :] *= scale_factors[ii]
            out.R[ii, :] *= scale_factors[ii]
            out.k[ii] *= scale_factors[ii]

        for jj in range(6):
            out.T[:, jj, :] *= scale_factors[jj]
            out.R[:, jj] *= scale_factors[jj]

        for kk in range(6):
            out.T[:, :, kk] *= scale_factors[kk]

        return out

class ElectronCoolerRecord(xo.HybridClass):
    _xofields = {
        '_index': RecordIndex,
        'Fx': xo.Float64[:],
        'Fy': xo.Float64[:],
        'Fl': xo.Float64[:],
        'particle_id': xo.Float64[:]}
class ElectronCooler(BeamElement):
    """
    Beam element modeling an electron cooler. In particular, this beam element uses the Parkhomchuk model for electron cooling.
    Every turn each particle receives transverse and longitudinal kicks based on the cooling force provided by the Parkhomchuk model.


    Parameters
        ----------
        current : float, optional
            The current in the electron beam, in amperes.
        length  : float, optional
            The length of the electron cooler, in meters.
        radius_e_beam : float, optional
            The radius of the electron beam, in meters.
        temp_perp : float, optional
            The transverse temperature of the electron beam, in electron volts.
        temp_long : float, optional
            The longitudinal temperature of the electron beam, in electron volts.
        magnetic_field : float, optional
            The magnetic field strength, in tesla.
        offset_x : float, optional
            The horizontal offset of the electron cooler, in meters.
        offset_px : float, optional
            The horizontal angle of the electron cooler, in rad.
        offset_y : float, optional
            The horizontal offset of the electron cooler, in meters.
        offset_py : float, optional
            The vertical angle of the electron cooler, in rad.
        offset_energy : float, optional
            The energy offset of the electrons, in eV.
        magnetic_field_ratio : float, optional
            The ratio of perpendicular component of magnetic field with the
            longitudinal component of the magnetic field. This is a measure
            of the magnetic field quality. With the ideal magnetic field quality 
            being 0.
        space_charge : float, optional
            Whether space charge of electron beam is enabled. 0 is off and 1 is on.

    """

    _xofields = {
        'current'       :  xo.Float64,
        'length'        :  xo.Float64,
        'radius_e_beam' :  xo.Float64,
        'temp_perp'     :  xo.Float64,
        'temp_long'     :  xo.Float64,
        'magnetic_field':  xo.Float64,

        'offset_x'      :  xo.Float64,
        'offset_px'     :  xo.Float64,
        'offset_y'      :  xo.Float64,
        'offset_py'     :  xo.Float64,
        'offset_energy' :  xo.Float64,

        'magnetic_field_ratio' :  xo.Float64,
        'space_charge_factor'  : xo.Float64,
        'record_flag': xo.Int64,
        }

    _extra_c_sources = [
        '#include <beam_elements/elements_src/electroncooler.h>',
    ]

    _internal_record_class = ElectronCoolerRecord

    def __init__(self,  current        = 0,
                        length         = 0,
                        radius_e_beam  = 0,
                        temp_perp      = 0,
                        temp_long      = 0,
                        magnetic_field = 0,

                        offset_x       = 0,
                        offset_px      = 0,
                        offset_y       = 0,
                        offset_py      = 0,
                        offset_energy  = 0,

                        magnetic_field_ratio = 0,
                        space_charge_factor  = 0,
                        record_flag          =0,
                        **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)
        self.current        = current
        self.length         = length
        self.radius_e_beam  = radius_e_beam
        self.temp_perp      = temp_perp
        self.temp_long      = temp_long
        self.magnetic_field = magnetic_field

        self.offset_x       = offset_x
        self.offset_px      = offset_px
        self.offset_y       = offset_y
        self.offset_py      = offset_py
        self.offset_energy  = offset_energy

        self.magnetic_field_ratio = magnetic_field_ratio
        self.space_charge_factor  = space_charge_factor
        self.record_flag          =  record_flag

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        raise NotImplementedError

class ThinSliceNotNeededError(Exception):
    pass

