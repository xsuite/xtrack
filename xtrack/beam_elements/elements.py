# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from numbers import Number
from scipy.special import factorial

import xobjects as xo
import xpart as xp

from ..base_element import BeamElement
from ..random import RandomUniform, RandomExponential, RandomNormal
from ..general import _pkg_root, _print
from ..internal_record import RecordIndex, RecordIdentifier


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
        _pkg_root.joinpath('beam_elements/elements_src/referenceenergyincrease.h')]

    has_backtrack = True


class Marker(BeamElement):
    """A marker beam element with no effect on the particles.
    """

    _xofields = {
        '_dummy': xo.Int64}

    behaves_like_drift = True
    allow_backtrack = True
    has_backtrack = True

    _extra_c_sources = [
        "/*gpufun*/\n"
        "void Marker_track_local_particle(MarkerData el, LocalParticle* part0){}"
    ]


class Drift(BeamElement):
    '''Beam element modeling a drift section.

    Parameters
    ----------

    length : float
        Length of the drift section in meters. Default is ``0``.

    '''

    _xofields = {
        'length': xo.Float64}

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_elem.h'),
        ]

    def make_slice(self, weight):
        return Drift(length=self.length * weight)

    @staticmethod
    def add_slice(weight, container, thick_name, slice_name, _buffer=None):
        container[slice_name] = Drift(_buffer=_buffer)
        container[slice_name].length = _get_expr(container[thick_name].length) * weight


class Henonmap(BeamElement):
    '''Beam element representing a Henon-like map with an arbitrary polynomial kick.

    Parameters
    ----------

    sin_omega_x : float
        Sine of linear angular frequency in the horizontal plane. Default is ``0``.
    cos_omega_x : float
        Cosine of linear angular frequency in the horizontal plane. Default is ``1``.
    sin_omega_y: float
        Sine of linear angular frequency in the vertical plane. Default is ``0``.
    cos_omega_y: float
        Cosine of linear angular frequency in the vertical plane. Default is ``1``.
    twiss_params: array of floats
        An array of the form [alpha_x, beta_x, alpha_y, beta_y] used for coordinate 
        normalisation and denormalisation. Default is ``[0, 1, 0, 1]``.
    fx_coeffs: array of floats
        An array that contains the coefficients of monomials of the form x^n*y*m that 
        represent the nonlinearities of the map in the horizontal plane. It is 
        calculated at initialisation based on the multipole coefficients provided.
    fx_x_exps: array of floats
        An array containing the exponents, n, of x for all monomials of the form 
        x^n*y^m that represent the nonlinearities of the map in the horizontal plane. 
        It is calculated at initialisation based on the multipole coefficients provided.
    fx_y_exps: array of floats
        An array containing the exponents, m, of y for all monomials of the form 
        x^n*y^m that represent the nonlinearities of the map in the horizontal plane. 
        It is calculated at initialisation based on the multipole coefficients provided.
    fy_coeffs: array of floats
        An array that contains the coefficients of monomials of the form x^n*y*m that 
        represent the nonlinearities of the map in the vertical plane. It is 
        calculated at initialisation based on the multipole coefficients provided.
    fy_x_exps: array of floats
        An array containing the exponents, n, of x for all monomials of the form 
        x^n*y^m that represent the nonlinearities of the map in the vertical plane. 
        It is calculated at initialisation based on the multipole coefficients provided.
    fy_y_exps: array of floats
        An array containing the exponents, m, of y for all monomials of the form 
        x^n*y^m that represent the nonlinearities of the map in the vertical plane. 
        It is calculated at initialisation based on the multipole coefficients provided.
    n_fx_coeffs: int
        Length of the arrays fx_coeffs, fx_x_exps, and fx_y_exps.
    n_fy_coeffs: int
        Length of the arrays fy_coeffs, fy_x_exps, and fy_y_exps.
    n_turns: int
        Number of turns to track. Default is ``1``.

    Comments
    --------

    - The user provides omega_x and omega_y, their sin and cos is calculated at 
      initialisation.
    - The user provides "multipole_coeffs", an array of floats, that contains the 
      strength of the multipoles present in the map. Default is ``[0]``.

    '''

    _xofields = {
        'sin_omega_x': xo.Float64,
        'cos_omega_x': xo.Float64,
        'sin_omega_y': xo.Float64,
        'cos_omega_y': xo.Float64,
        'n_turns': xo.Int64,
        'twiss_params': xo.Float64[:],
        'fx_coeffs': xo.Float64[:],
        'fx_x_exps': xo.Int64[:],
        'fx_y_exps': xo.Int64[:],
        'fy_coeffs': xo.Float64[:],
        'fy_x_exps': xo.Int64[:],
        'fy_y_exps': xo.Int64[:],
        'n_fx_coeffs': xo.Int64,
        'n_fy_coeffs': xo.Int64,
        'norm': xo.Int64,
    }

    isthick = False
    behaves_like_drift = False

    _extra_c_sources = [_pkg_root.joinpath('beam_elements/elements_src/henonmap.h')]

    def __init__(self, omega_x = 0.,
                       omega_y = 0.,
                       n_turns = 1, 
                       twiss_params = [0., 1., 0., 1.],
                       multipole_coeffs = [0.],
                       norm = False, 
                       **kwargs):
        if '_xobject' not in kwargs:
            kwargs.setdefault('sin_omega_x', np.sin(omega_x))
            kwargs.setdefault('cos_omega_x', np.cos(omega_x))
            kwargs.setdefault('sin_omega_y', np.sin(omega_y))
            kwargs.setdefault('cos_omega_y', np.cos(omega_y))
            kwargs.setdefault('n_turns', n_turns)
            kwargs.setdefault('twiss_params', twiss_params)

            fx_coeffs = []
            fx_x_exps = []
            fx_y_exps = []
            fy_coeffs = []
            fy_x_exps = []
            fy_y_exps = []
            for n in range(2, len(multipole_coeffs) + 2):
                for k in range(0, n + 1):
                    if (k % 4) == 0:
                        fx_coeffs.append(multipole_coeffs[n - 2] / factorial(k) / factorial(n - k))
                        fx_x_exps.append(n - k)
                        fx_y_exps.append(k)
                    elif (k % 4) == 2:
                        fx_coeffs.append(-1 * multipole_coeffs[n - 2] / factorial(k) / factorial(n - k))
                        fx_x_exps.append(n - k)
                        fx_y_exps.append(k)
                    elif (k % 4) == 1:
                        fy_coeffs.append(multipole_coeffs[n - 2] / factorial(k) / factorial(n - k))
                        fy_x_exps.append(n - k)
                        fy_y_exps.append(k)
                    else:
                        fy_coeffs.append(-1 * multipole_coeffs[n - 2] / factorial(k) / factorial(n - k))
                        fy_x_exps.append(n - k)
                        fy_y_exps.append(k)
            kwargs.setdefault('fx_coeffs', fx_coeffs)
            kwargs.setdefault('fx_x_exps', fx_x_exps)
            kwargs.setdefault('fx_y_exps', fx_y_exps)
            kwargs.setdefault('fy_coeffs', fy_coeffs)
            kwargs.setdefault('fy_x_exps', fy_x_exps)
            kwargs.setdefault('fy_y_exps', fy_y_exps)
            kwargs.setdefault('n_fx_coeffs', len(fx_coeffs))
            kwargs.setdefault('n_fy_coeffs', len(fy_coeffs))

            if norm == True:
                kwargs.setdefault('norm', 1)
            else:
                kwargs.setdefault('norm', 0)

        super().__init__(**kwargs)

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        raise NotImplementedError

class Cavity(BeamElement):
    '''Beam element modeling an RF cavity.

    Parameters
    ----------
    voltage : float
        Voltage of the RF cavity in Volts. Default is ``0``.
    frequency : float
        Frequency of the RF cavity in Hertz. Default is ``0``.
    lag : float
        Phase seen by the reference particle in degrees. Default is ``0``.

    '''

    _xofields = {
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/cavity.h')]

    has_backtrack = True


class XYShift(BeamElement):
    '''Beam element modeling an transverse shift of the reference system.

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

    allow_backtrack = True
    has_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xyshift.h')]


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

# if array is needed we do it like this
#    _xofields={'inner_radius': xo.Float64[:]}
    _xofields={
               'current'                : xo.Float64,
               'inner_radius'           : xo.Float64,
               'outer_radius'           : xo.Float64,
               'elens_length'           : xo.Float64,
               'voltage'                : xo.Float64,
               'residual_kick_x'        : xo.Float64,
               'residual_kick_y'        : xo.Float64,
               'coefficients_polynomial': xo.Float64[:],
               'polynomial_order'       : xo.Float64
              }

    has_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/elens.h')]

    def __init__(self,  inner_radius = 1.,
                        outer_radius = 1.,
                        current      = 0.,
                        elens_length = 0.,
                        voltage      = 0.,
                        residual_kick_x = 0,
                        residual_kick_y = 0,
                        coefficients_polynomial = [0],
                        _xobject = None,
                        **kwargs):

        kwargs["coefficients_polynomial"] = len(coefficients_polynomial)

        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
            super().__init__(**kwargs)
            self.inner_radius    = inner_radius
            self.outer_radius    = outer_radius
            self.current         = current
            self.elens_length    = elens_length
            self.voltage         = voltage
            self.residual_kick_x   = residual_kick_x
            self.residual_kick_y   = residual_kick_y

            self.coefficients_polynomial[:] = self._arr2ctx(coefficients_polynomial)
            polynomial_order = len(coefficients_polynomial)-1
            self.polynomial_order = polynomial_order



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
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/wire.h'),
    ]

    def __init__(self,  L_phy   = 0,
                        L_int   = 0,
                        current = 0,
                        xma     = 0,
                        yma     = 0,
                        post_subtract_px = 0,
                        post_subtract_py = 0,
                        _xobject = None,
                        **kwargs):

        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
            super().__init__(**kwargs)
            self.L_phy   = L_phy
            self.L_int   = L_int
            self.current = current
            self.xma     = xma
            self.yma     = yma
            self.post_subtract_px = post_subtract_px
            self.post_subtract_py = post_subtract_py


class SRotation(BeamElement):
    '''Beam element modeling an rotation of the reference system around the s axis.

    Parameters
    ----------

    angle : float
        Rotation angle in degrees. Default is ``0``.

    '''

    _xofields = {
        'cos_z': xo.Float64,
        'sin_z': xo.Float64,
        }

    allow_backtrack = True
    has_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/srotation.h')]

    _store_in_to_dict = ['angle']

    def __init__(self, angle=None, cos_z=None, sin_z=None, **kwargs):
        """
        If either angle or a sufficient number of trig values are given,
        calculate the missing values from the others. If more than necessary
        parameters are given, their consistency will be checked.
        """
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
    '''Beam element modeling an rotation of the reference system around the x axis.

    Parameters
    ----------

    angle : float
        Rotation angle in degrees. Default is ``0``.

    '''

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

    allow_backtrack = True
    has_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xrotation.h')]

    _store_in_to_dict = ['angle']

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
    '''Beam element modeling an rotation of the reference system around the y axis.

    Parameters
    ----------

    angle : float
        Rotation angle in degrees. Default is ``0``.

    '''

    has_backtrack = True
    allow_backtrack = True

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/yrotation.h')]

    _store_in_to_dict = ['angle']

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
        #Note MAD-X node_value('other_bv ') is ignored
        #     minus sign follows MAD-X convention
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
        anglerad = -anglerad

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
        return -np.arctan2(self.sin_angle, self.cos_angle) * (180.0 / np.pi)

    @angle.setter
    def angle(self, value):
        anglerad = -value / 180 * np.pi
        self.cos_angle = np.cos(anglerad)
        self.sin_angle = np.sin(anglerad)
        self.tan_angle = np.tan(anglerad)


class ZetaShift(BeamElement):
    '''Beam element modeling a time delat.

    Parameters
    ----------

    dzeta : float
        Time shift dzeta in meters. Default is ``0``.

    '''

    _xofields={
        'dzeta': xo.Float64,
        }

    has_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/zetashift.h')]

    _store_in_to_dict = ['dzeta']

    def __init__(self, dzeta = 0, **nargs):
        nargs['dzeta'] = dzeta
        super().__init__(**nargs)


class SynchrotronRadiationRecord(xo.HybridClass):
    _xofields = {
        '_index': RecordIndex,
        'photon_energy': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:],
        'particle_delta': xo.Float64[:]
        }


class Multipole(BeamElement):
    '''Beam element modeling a thin magnetic multipole.

    Parameters
    ----------

    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
    ksl : array
        Normalized integrated strength of the skew components in units of m^-n.
    order : int
        Order of the multipole. Default is ``0``.
    hxl : float
        Rotation angle of the reference trajectory in the horizontal plane in radians. Default is ``0``.
    hyl : float
        Rotation angle of the reference trajectory in the vertical plane in radians. Default is ``0``.
    length : float
        Length of the originating thick multipole. Default is ``0``.

    '''

    _xofields={
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'length': xo.Float64,
        'hxl': xo.Float64,
        'hyl': xo.Float64,
        'radiation_flag': xo.Int64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        }

    _rename = {
        'order': '_order',
    }

    _depends_on = [RandomUniform, RandomExponential]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/multipole.h')]

    _internal_record_class = SynchrotronRadiationRecord

    has_backtrack = True

    def __init__(self, order=None, knl=None, ksl=None, **kwargs):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if order is None:
            order = 0

        if "bal" in kwargs.keys():
            if not "knl" in kwargs.keys() or not "ksl" in kwargs.keys():
                _bal = kwargs['bal']
                idxes = np.array([ii for ii in range(0, len(_bal), 2)])
                knl = [_bal[idx] * factorial(idx // 2, exact=True) for idx in idxes]
                ksl = [_bal[idx + 1] * factorial(idx // 2, exact=True) for idx in idxes]

        len_knl = len(knl) if knl is not None else 0
        len_ksl = len(ksl) if ksl is not None else 0
        n = max((order + 1), max(len_knl, len_ksl))
        assert n > 0

        nknl = np.zeros(n, dtype=np.float64)
        nksl = np.zeros(n, dtype=np.float64)

        if knl is not None:
            nknl[: len(knl)] = np.array(knl)

        if ksl is not None:
            nksl[: len(ksl)] = np.array(ksl)

        order = n - 1

        kwargs["knl"] = nknl
        kwargs["ksl"] = nksl
        kwargs["order"] = order
        kwargs["inv_factorial_order"] = 1.0 / factorial(order, exact=True)

        self.xoinitialize(**kwargs)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)


class SimpleThinQuadrupole(BeamElement):
    """An specialized version of Multipole to model a thin quadrupole
    (knl[0], ksl, hxl, hyl are all zero).

    Parameters
    ----------
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length 2.

    """

    _xofields={
        'knl': xo.Float64[2],
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/simplethinquadrupole.h')]

    def __init__(self, knl=None, **kwargs):
        if knl is None:
            knl = np.zeros(2)

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if len(knl) != 2:
            raise ValueError("For a quadrupole, len(knl) must be 2.")

        kwargs["knl"] = knl
        self.xoinitialize(**kwargs)

    @property
    def hxl(self): return 0.0

    @property
    def hyl(self): return 0.0

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


class CombinedFunctionMagnet(BeamElement):
    isthick = True

    _xofields={
        'k0': xo.Float64,
        'k1': xo.Float64,
        'h': xo.Float64,
        'length': xo.Float64,
        'knl': xo.Float64[5],
        'ksl': xo.Float64[5],
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
    }

    _rename = {
        'order': '_order',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/multipolar_kick.h'),
        _pkg_root.joinpath('beam_elements/elements_src/combinedfunctionmagnet.h'),
    ]

    def __init__(self, **kwargs):
        if kwargs.get('length', 0.0) == 0.0 and not '_xobject' in kwargs:
            raise ValueError("A thick element must have a length.")

        knl = kwargs.get('knl', np.array([]))
        ksl = kwargs.get('ksl', np.array([]))
        order_from_kl = max(len(knl), len(ksl)) - 1
        order = kwargs.get('order', max(4, order_from_kl))

        if order > 4:
            raise NotImplementedError # Untested

        kwargs['knl'] = np.pad(knl, (0, 5 - len(knl)), 'constant')
        kwargs['ksl'] = np.pad(ksl, (0, 5 - len(ksl)), 'constant')

        self.xoinitialize(**kwargs)

        self.order = order

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)

    @property
    def hxl(self): return self.h * self.length

    @property
    def hyl(self): return 0.0

    @property
    def radiation_flag(self): return 0.0

    @staticmethod
    def add_slice(weight, container, thick_name, slice_name, _buffer=None):
        self_or_ref = container[thick_name]

        container[slice_name] = Multipole(knl=np.zeros(5), ksl=np.zeros(5),
                                          _buffer=_buffer)
        ref = container[slice_name]

        ref.knl[0] = (_get_expr(self_or_ref.k0) * _get_expr(self_or_ref.length)
                      + _get_expr(self_or_ref.knl[0])) * weight
        ref.knl[1] = (_get_expr(self_or_ref.k1) * _get_expr(self_or_ref.length)
                      + _get_expr(self_or_ref.knl[1])) * weight

        order = 1
        for ii in range(2, 5):
            ref.knl[ii] = _get_expr(self_or_ref.knl[ii]) * weight

            if _nonzero(ref.knl[ii]):
                order = max(order, ii)

        for ii in range(5):
            ref.ksl[ii] = _get_expr(self_or_ref.ksl[ii]) * weight

            if _nonzero(self_or_ref.ksl[ii]):  # update in the same way for ksl
                order = max(order, ii)

        ref.hxl = _get_expr(self_or_ref.h) * _get_expr(self_or_ref.length) * weight
        ref.length = _get_expr(self_or_ref.length) * weight
        ref.order = order

    @staticmethod
    def delete_element_ref(ref):
        # Remove the array fields
        for field in ['knl', 'ksl']:
            for ii in range(5):
                _unregister_if_preset(getattr(ref, field)[ii])

        # Remove the scalar fields
        for field in [
            'k0', 'k1', 'h', 'length', 'num_multipole_kicks', 'order',
            'inv_factorial_order',
        ]:
            _unregister_if_preset(getattr(ref, field))

        # Remove the ref to the element itself
        _unregister_if_preset(ref)


class Quadrupole(BeamElement):
    isthick = True

    _xofields={
        'k1': xo.Float64,
        'length': xo.Float64,
        'knl': xo.Float64[5],
        'ksl': xo.Float64[5],
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
    }

    _rename = {
        'order': '_order',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/multipolar_kick.h'),
        _pkg_root.joinpath('beam_elements/elements_src/quadrupole.h'),
    ]

    def __init__(self, **kwargs):
        if kwargs.get('length', 0.0) == 0.0 and not '_xobject' in kwargs:
            raise ValueError("A thick element must have a length.")

        knl = kwargs.get('knl', np.array([]))
        ksl = kwargs.get('ksl', np.array([]))
        order_from_kl = max(len(knl), len(ksl)) - 1
        order = kwargs.get('order', max(4, order_from_kl))

        if order > 4:
            raise NotImplementedError # Untested

        kwargs['knl'] = np.pad(knl, (0, 5 - len(knl)), 'constant')
        kwargs['ksl'] = np.pad(ksl, (0, 5 - len(ksl)), 'constant')

        self.xoinitialize(**kwargs)

        self.order = order

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)

    @property
    def hxl(self): return self.h * self.length

    @property
    def hyl(self): return 0.0

    @property
    def radiation_flag(self): return 0.0

    @staticmethod
    def add_slice(weight, container, thick_name, slice_name, _buffer=None):
        self_or_ref = container[thick_name]

        container[slice_name] = Multipole(knl=np.zeros(5), ksl=np.zeros(5),
                                          _buffer=_buffer)
        ref = container[slice_name]

        ref.knl[0] = 0.
        ref.knl[1] = (_get_expr(self_or_ref.k1) * _get_expr(self_or_ref.length)
                      + _get_expr(self_or_ref.knl[1])) * weight

        order = 1
        for ii in range(2, 5):
            ref.knl[ii] = _get_expr(self_or_ref.knl[ii]) * weight

            if _nonzero(ref.knl[ii]):
                order = max(order, ii)

        for ii in range(5):
            ref.ksl[ii] = _get_expr(self_or_ref.ksl[ii]) * weight

            if _nonzero(self_or_ref.ksl[ii]):  # update in the same way for ksl
                order = max(order, ii)

        ref.hxl = 0
        ref.length = _get_expr(self_or_ref.length) * weight
        ref.order = order

    @staticmethod
    def delete_element_ref(ref):
        # Remove the array fields
        for field in ['knl', 'ksl']:
            for ii in range(5):
                _unregister_if_preset(getattr(ref, field)[ii])

        # Remove the scalar fields
        for field in [
            'k1', 'length', 'num_multipole_kicks', 'order', 'inv_factorial_order',
        ]:
            _unregister_if_preset(getattr(ref, field))

        # Remove the ref to the element itself
        _unregister_if_preset(ref)


class Bend(BeamElement):
    isthick = True

    _xofields={
        'k0': xo.Float64,
        'h': xo.Float64,
        'length': xo.Float64,
        'knl': xo.Float64[5],
        'ksl': xo.Float64[5],
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'method': xo.Int64,
    }

    _rename = {
        'order': '_order',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/multipolar_kick.h'),
        _pkg_root.joinpath('beam_elements/elements_src/bend.h'),
    ]

    def __init__(self, **kwargs):
        if kwargs.get('length', 0.0) == 0.0 and not '_xobject' in kwargs:
            raise ValueError("A thick element must have a length.")

        knl = kwargs.get('knl', np.array([]))
        ksl = kwargs.get('ksl', np.array([]))
        order_from_kl = max(len(knl), len(ksl)) - 1
        order = kwargs.get('order', max(order_from_kl, 4))

        if order > 4:
            raise NotImplementedError # Untested

        kwargs['knl'] = np.pad(knl, (0, 5 - len(knl)), 'constant')
        kwargs['ksl'] = np.pad(ksl, (0, 5 - len(ksl)), 'constant')

        kwargs['method'] = kwargs.get('method', 0)

        self.xoinitialize(**kwargs)

        self.order = order

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)

    @property
    def hxl(self): return self.h * self.length

    @property
    def hyl(self): return 0.0

    @property
    def radiation_flag(self): return 0.0

    @staticmethod
    def add_slice(weight, container, thick_name, slice_name, _buffer=None):
        self_or_ref = container[thick_name]

        container[slice_name] = Multipole(knl=np.zeros(5), ksl=np.zeros(5),
                                          _buffer=_buffer)
        ref = container[slice_name]

        ref.knl[0] = (_get_expr(self_or_ref.k0) * _get_expr(self_or_ref.length)
                      + _get_expr(self_or_ref.knl[0])) * weight
        order = 0
        for ii in range(1, 5):
            ref.knl[ii] = _get_expr(self_or_ref.knl[ii]) * weight

            if _nonzero(self_or_ref.knl[ii]):  # order is max ii where knl[ii] is expr or nonzero
                order = ii

        for ii in range(5):
            ref.ksl[ii] = _get_expr(self_or_ref.ksl[ii]) * weight

            if _nonzero(self_or_ref.ksl[ii]):  # update in the same way for ksl
                order = max(order, ii)

        ref.hxl = _get_expr(self_or_ref.h) * _get_expr(self_or_ref.length) * weight
        ref.length = _get_expr(self_or_ref.length) * weight
        ref.order = order

    @staticmethod
    def delete_element_ref(ref):
        # Remove the array fields
        for field in ['knl', 'ksl']:
            for ii in range(5):
                _unregister_if_preset(getattr(ref, field)[ii])

        # Remove the scalar fields
        for field in [
            'k0', 'h', 'length', 'num_multipole_kicks', 'order',
            'inv_factorial_order',
        ]:
            _unregister_if_preset(getattr(ref, field))

        # Remove the ref to the element itself
        _unregister_if_preset(ref[field])


class SimpleThinBend(BeamElement):
    '''A specialized version of Multipole to model a thin bend (ksl, hyl are all zero).
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length 1.
    hxl : float
        Rotation angle of the reference trajectory in the horizontal plane in radians. Default is ``0``.
    length : float
        Length of the originating thick bend. Default is ``0``.
    '''

    _xofields={
        'knl': xo.Float64[1],
        'hxl': xo.Float64,
        'length': xo.Float64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/simplethinbend.h')]

    def __init__(self, knl=None, **kwargs):
        if knl is None:
            knl = np.zeros(1)

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if len(knl) != 1:
            raise ValueError("For a quadrupole, len(knl) must be 1.")

        kwargs["knl"] = knl
        self.xoinitialize(**kwargs)

    @property
    def hyl(self): return 0.0

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


class RFMultipole(BeamElement):
    '''Beam element modeling a thin modulated multipole, with strengths dependent on the z coordinate:

    Parameters
    ----------
    order : int
        Order of the multipole. Default is ``0``.
    knl : array
        Normalized integrated strength of the normal components in units of m^-n.
        Must be of length ``order+1``.
    ksl : array
        Normalized integrated strength of the skew components in units of m^-n.
        Must be of length ``order+1``.
    pn : array
        Phase of the normal components in degrees. Must be of length ``order+1``.
    ps : array
        Phase of the skew components in degrees. Must be of length ``order+1``.
    voltage : float
        Longitudinal voltage. Default is ``0``.
    lag : float
        Longitudinal phase seen by the reference particle. Default is ``0``.
    frequency : float
        Frequency in Hertz. Default is ``0``.

    '''

    _xofields={
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'pn': xo.Float64[:],
        'ps': xo.Float64[:],
    }

    has_backtrack = True

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/rfmultipole.h')]

    def __init__(
        self,
        order=None,
        knl=None,
        ksl=None,
        pn=None,
        ps=None,
        **kwargs
    ):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        assert 'p' not in kwargs, "`p` in RF Multipole is not supported anymore"

        if order is None:
            order = 0

        if "bal" in kwargs.keys():
            if not "knl" in kwargs.keys() or not "ksl" in kwargs.keys():
                _bal = kwargs['bal']
                idxes = np.array([ii for ii in range(0, len(_bal), 2)])
                knl = [_bal[idx] * factorial(idx // 2, exact=True) for idx in idxes]
                ksl = [_bal[idx + 1] * factorial(idx // 2, exact=True) for idx in idxes]


        len_knl = len(knl) if knl is not None else 0
        len_ksl = len(ksl) if ksl is not None else 0
        len_pn = len(pn) if pn is not None else 0
        len_ps = len(ps) if ps is not None else 0
        n = max((order + 1), max(len_knl, len_ksl, len_pn, len_ps))
        assert n > 0

        nknl = np.zeros(n, dtype=np.float64)
        nksl = np.zeros(n, dtype=np.float64)
        npn = np.zeros(n, dtype=np.float64)
        nps = np.zeros(n, dtype=np.float64)

        if knl is not None:
            nknl[: len(knl)] = np.array(knl)

        if ksl is not None:
            nksl[: len(ksl)] = np.array(ksl)

        if pn is not None:
            npn[: len(pn)] = np.array(pn)

        if ps is not None:
            nps[: len(ps)] = np.array(ps)

        order = n - 1

        kwargs["knl"] = nknl
        kwargs["ksl"] = nksl
        kwargs["pn"] = npn
        kwargs["ps"] = nps
        kwargs["order"] = order
        #kwargs["inv_factorial_order"] = 1.0 / factorial(order, exact=True)

        self.xoinitialize(**kwargs)


class DipoleEdge(BeamElement):
    '''Beam element modeling a dipole edge (see MAD-X manual for detaild description).

    Parameters
    ----------
    h : float
        Curvature in 1/m.
    e1 : float
        Face angle in rad.
    hgap : float
        Equivalent gap in m.
    fint : float
        Fringe integral.

    '''

    _xofields = {
            'mode': xo.Int64,
            'r21': xo.Float64,
            'r43': xo.Float64,
            'hgap': xo.Float64,
            'h': xo.Float64,
            'e1': xo.Float64,
            'fint': xo.Float64,
            }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/dipoleedge.h')]

    has_backtrack = True

    _rename = {
        'r21': '_r21',
        'r43': '_r43',
        'hgap': '_hgap',
        'h': '_h',
        'e1': '_e1',
        'fint': '_fint',
    }

    def __init__(
        self,
        r21=None,
        r43=None,
        h=None,
        e1=None,
        hgap=None,
        fint=None,
        mode=None,
        **kwargs
    ):

        self.xoinitialize(**kwargs)
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            return
        if '_r21' in kwargs.keys():
            # has been set with underscored variables
            return

        # To have them initalized
        self.mode = 0
        self._hgap = (hgap or 0)
        self._h = (h or 0)
        self._e1 = (e1 or 0)
        self._fint = (fint or 0)
        self._r21 = (r21 or 0)
        self._r43 = (r43 or 0)

        if mode is not None:
            self.mode = mode
        elif r21 is not None or r43 is not None:
            self.mode = 1
        else:
            self.mode = 0

        if self.mode == 0:
            self._update_r21_r43()

    def _update_r21_r43(self):
        corr = np.float64(2.0) * self.h * self.hgap * self.fint
        r21 = self.h * np.tan(self.e1)
        temp = corr / np.cos(self.e1) * (
            np.float64(1) + np.sin(self.e1) * np.sin(self.e1))
        r43 = -self.h * np.tan(self.e1 - temp)
        self._r21 = r21
        self._r43 = r43
        self.mode = 0

    @property
    def h(self):
        if self.mode == 0:
            return self._h
        else:
            raise AttributeError(
                "`h` is not defined because r21 and r43 were provided directly")

    @h.setter
    def h(self, value):
        self._h = value
        self._update_r21_r43()

    @property
    def e1(self):
        if self.mode == 0:
            return self._e1
        else:
            raise ValueError(
                "`e1` is not defined because r21 and r43 were provided directly")

    @e1.setter
    def e1(self, value):
        self._e1 = value
        self._update_r21_r43()

    @property
    def hgap(self):
        if self.mode == 0:
            return self._hgap
        else:
            raise ValueError(
                "`hgap` is not defined because r21 and r43 were provided directly")

    @hgap.setter
    def hgap(self, value):
        self._hgap = value
        self._update_r21_r43()

    @property
    def fint(self):
        if self.mode == 0:
            return self._fint
        else:
            raise ValueError(
                "`fint` is not defined because r21 and r43 were provided directly")

    @fint.setter
    def fint(self, value):
        self._fint = value
        self._update_r21_r43()

    @property
    def r21(self):
        return self._r21

    @r21.setter
    def r21(self, value):
        self._r21 = value
        self.mode = 1

    @property
    def r43(self):
        return self._r43

    @r43.setter
    def r43(self, value):
        self._r43 = value
        self.mode = 1


class LineSegmentMap(BeamElement):

    _xofields={
        'length': xo.Float64,

        'qx': xo.Float64,
        'qy': xo.Float64,

        'dqx': xo.Float64,
        'dqy': xo.Float64,
        'detx_x': xo.Float64,
        'detx_y': xo.Float64,
        'dety_y': xo.Float64,
        'dety_x': xo.Float64,

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
        'damping_factor_x':xo.Float64,
        'damping_factor_y':xo.Float64,
        'damping_factor_s':xo.Float64,
        'uncorrelated_gauss_noise': xo.Int64,
        'gauss_noise_ampl_x':xo.Float64,
        'gauss_noise_ampl_px':xo.Float64,
        'gauss_noise_ampl_y':xo.Float64,
        'gauss_noise_ampl_py':xo.Float64,
        'gauss_noise_ampl_zeta':xo.Float64,
        'gauss_noise_ampl_delta':xo.Float64,

        'longitudinal_mode_flag': xo.Int64,
        'qs': xo.Float64,
        'bets': xo.Float64,
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
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/linesegmentmap.h')]

    def __init__(self, length=None, qx=0, qy=0,
            betx=1., bety=1., alfx=0., alfy=0.,
            dx=0., dpx=0., dy=0., dpy=0.,
            x_ref=0.0, px_ref=0.0, y_ref=0.0, py_ref=0.0,
            longitudinal_mode=None,
            qs=None, bets=None,
            momentum_compaction_factor=None,
            slippage_length=None,
            voltage_rf=None, frequency_rf=None, lag_rf=None,
            dqx=0.0, dqy=0.0,
            detx_x=0.0, detx_y=0.0, dety_y=0.0, dety_x=0.0,
            energy_increment=0.0, energy_ref_increment=0.0,
            damping_rate_x = 0.0, damping_rate_y = 0.0, damping_rate_s = 0.0,
            equ_emit_x = 0.0, equ_emit_y = 0.0, equ_emit_s = 0.0,
            gauss_noise_ampl_x=0.0,gauss_noise_ampl_px=0.0,
            gauss_noise_ampl_y=0.0,gauss_noise_ampl_py=0.0,
            gauss_noise_ampl_zeta=0.0,gauss_noise_ampl_delta=0.0,
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
        dqx : float
            Horizontal chromaticity of the segment.
        dqy : float
            Vertical chromaticity of the segment.
        detx_x : float
            Anharmonicity xx coefficient. Optional, default is ``0``.
        detx_y : float
            Anharmonicity xy coefficient. Optional, default is ``0``.
        dety_y : float
            Anharmonicity yy coefficient. Optional, default is ``0``.
        energy_increment : float
            Energy increment of the segment in eV.
        energy_ref_increment : float
            Increment of the reference energy in eV.
        damping_rate_x : float
            Horizontal damping rate on the particles motion. Optional, default is ``0``.
        damping_rate_y : float
            Vertical damping rate on the particles motion. Optional, default is ``0``.
        damping_rate_s : float
            Longitudinal damping rate on the particles motion. Optional, default is ``0``.
        equ_emit_x : float
            Horizontal equilibrium emittance. Optional.
        equ_emit_y : float
            Vertical equilibrium emittance. Optional.
        equ_emit_s : float
            Longitudinal equilibrium emittance. Optional.
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
        gauss_noise_ampl_delta : float
            Amplitude of Gaussian noise on the longitudinal momentum. Optional, default is ``0``.

        '''

        assert longitudinal_mode in ['linear_fixed_qs', 'nonlinear', 'linear_fixed_rf', None]

        nargs['qx'] = qx
        nargs['qy'] = qy
        nargs['dqx'] = dqx
        nargs['dqy'] = dqy
        nargs['detx_x'] = detx_x
        nargs['detx_y'] = detx_y
        nargs['dety_y'] = dety_y
        nargs['dety_x'] = dety_x
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
            nargs['longitudinal_mode_flag'] = 1
            nargs['qs'] = qs
            nargs['bets'] = bets
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

            if slippage_length is None:
                assert length is not None
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

        if damping_rate_x < 0.0 or damping_rate_y < 0.0 or damping_rate_s < 0.0:
            raise ValueError('Damping rates cannot be negative')
        if damping_rate_x > 0.0 or damping_rate_y > 0.0 or damping_rate_s > 0.0:
            nargs['uncorrelated_rad_damping'] = True
            nargs['damping_factor_x'] = 1.0-damping_rate_x/2.0
            nargs['damping_factor_y'] = 1.0-damping_rate_y/2.0
            nargs['damping_factor_s'] = 1.0-damping_rate_s/2.0
        else:
            nargs['uncorrelated_rad_damping'] = False

        if equ_emit_x < 0.0 or equ_emit_y < 0.0 or equ_emit_s < 0.0:
            raise ValueError('Equilibrium emittances cannot be negative')
        nargs['uncorrelated_gauss_noise'] = False
        nargs['gauss_noise_ampl_x'] = 0.0
        nargs['gauss_noise_ampl_px'] = 0.0
        nargs['gauss_noise_ampl_y'] = 0.0
        nargs['gauss_noise_ampl_py'] = 0.0
        nargs['gauss_noise_ampl_zeta'] = 0.0
        nargs['gauss_noise_ampl_delta'] = 0.0

        assert equ_emit_x >= 0.0
        assert equ_emit_y >= 0.0
        assert equ_emit_s >= 0.0

        if equ_emit_x > 0.0:
            assert alfx[1] == 0
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_px'] = np.sqrt(equ_emit_x*damping_rate_x/betx[1])
            nargs['gauss_noise_ampl_x'] = betx[0]*nargs['gauss_noise_ampl_px']
        if equ_emit_y > 0.0:
            assert alfy[1] == 0
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_py'] = np.sqrt(equ_emit_y*damping_rate_y/bety[1])
            nargs['gauss_noise_ampl_y'] = bety[0]*nargs['gauss_noise_ampl_py']
        if equ_emit_s > 0.0:
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_delta'] = np.sqrt(equ_emit_s*damping_rate_s/bets)
            nargs['gauss_noise_ampl_zeta'] = bets*nargs['gauss_noise_ampl_delta']

        assert gauss_noise_ampl_x >= 0.0
        assert gauss_noise_ampl_px >= 0.0
        assert gauss_noise_ampl_y >= 0.0
        assert gauss_noise_ampl_py >= 0.0
        assert gauss_noise_ampl_zeta >= 0.0
        assert gauss_noise_ampl_delta >= 0.0

        if gauss_noise_ampl_x > 0.0 or gauss_noise_ampl_px > 0.0 or gauss_noise_ampl_y > 0.0 or gauss_noise_ampl_py > 0.0 or gauss_noise_ampl_zeta > 0.0 or gauss_noise_ampl_delta > 0.0:
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_x'] = np.sqrt(nargs['gauss_noise_ampl_x']**2+gauss_noise_ampl_x**2)
            nargs['gauss_noise_ampl_px'] = np.sqrt(nargs['gauss_noise_ampl_px']**2+gauss_noise_ampl_px**2)
            nargs['gauss_noise_ampl_y'] = np.sqrt(nargs['gauss_noise_ampl_y']**2+gauss_noise_ampl_y**2)
            nargs['gauss_noise_ampl_py'] = np.sqrt(nargs['gauss_noise_ampl_py']**2+gauss_noise_ampl_py**2)
            nargs['gauss_noise_ampl_zeta'] = np.sqrt(nargs['gauss_noise_ampl_zeta']**2+gauss_noise_ampl_zeta**2)
            nargs['gauss_noise_ampl_delta'] = np.sqrt(nargs['gauss_noise_ampl_delta']**2+gauss_noise_ampl_delta**2)

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

    '''
    First order Taylor map.

    Parameters
    ----------
    length : float
        length of the element in meters.
    m0 : array_like
        6x1 array of the zero order Taylor map coefficients.
    m1 : array_like
        6x6 array of the first order Taylor map coefficients.
    radiation_flag : int
        Flag for synchrotron radiation. 0 - no radiation, 1 - radiation on.

    '''

    isthick = True

    _xofields={
        'radiation_flag': xo.Int64,
        'length': xo.Float64,
        'm0': xo.Float64[6],
        'm1': xo.Float64[6,6]}

    _depends_on = [RandomUniform, RandomExponential]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/firstordertaylormap.h')]

    _internal_record_class = SynchrotronRadiationRecord # not functional,
    # included for compatibility with Multipole

    def __init__(self, length = 0.0, m0 = None, m1 = None,radiation_flag=0,**nargs):
        nargs['radiation_flag'] = radiation_flag
        nargs['length'] = length
        if m0 is None:
            nargs['m0'] = np.zeros(6,dtype=np.float64)
        else:
            if len(np.shape(m0)) == 1 and np.shape(m0)[0] == 6:
                nargs['m0'] = m0
            else:
                raise ValueError(f'Wrong shape for m0: {np.shape(m0)}')
        if m1 is None:
            nargs['m1'] = np.eye(6,dtype=np.float64)
        else:
            if len(np.shape(m1)) == 2 and np.shape(m1)[0] == 6 and np.shape(m1)[1] == 6:
                nargs['m1'] = m1
            else:
                raise ValueError(f'Wrong shape for m1: {np.shape(m1)}')
        super().__init__(**nargs)


class LinearTransferMatrix(BeamElement):
    _xofields={
        'no_detuning': xo.Int64,
        'q_x': xo.Float64,
        'q_y': xo.Float64,
        'cos_s': xo.Float64,
        'sin_s': xo.Float64,
        'beta_x_0': xo.Float64,
        'beta_y_0': xo.Float64,
        'beta_ratio_x': xo.Float64,
        'beta_prod_x': xo.Float64,
        'beta_ratio_y': xo.Float64,
        'beta_prod_y': xo.Float64,
        'alpha_x_0': xo.Float64,
        'alpha_x_1': xo.Float64,
        'alpha_y_0': xo.Float64,
        'alpha_y_1': xo.Float64,
        'disp_x_0': xo.Float64,
        'disp_x_1': xo.Float64,
        'disp_y_0': xo.Float64,
        'disp_y_1': xo.Float64,
        'beta_s': xo.Float64,
        'energy_ref_increment': xo.Float64,
        'energy_increment': xo.Float64,
        'chroma_x': xo.Float64,
        'chroma_y': xo.Float64,
        'detx_x': xo.Float64,
        'detx_y': xo.Float64,
        'dety_y': xo.Float64,
        'dety_x': xo.Float64,
        'x_ref_0': xo.Float64,
        'px_ref_0': xo.Float64,
        'y_ref_0': xo.Float64,
        'py_ref_0': xo.Float64,
        'x_ref_1': xo.Float64,
        'px_ref_1': xo.Float64,
        'y_ref_1': xo.Float64,
        'py_ref_1': xo.Float64,
        'length': xo.Float64,
        'uncorrelated_rad_damping': xo.Int64,
        'damping_factor_x':xo.Float64,
        'damping_factor_y':xo.Float64,
        'damping_factor_s':xo.Float64,
        'uncorrelated_gauss_noise': xo.Int64,
        'gauss_noise_ampl_x':xo.Float64,
        'gauss_noise_ampl_px':xo.Float64,
        'gauss_noise_ampl_y':xo.Float64,
        'gauss_noise_ampl_py':xo.Float64,
        'gauss_noise_ampl_zeta':xo.Float64,
        'gauss_noise_ampl_delta':xo.Float64,
        }

    _depends_on = [RandomNormal]
    isthick = True

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/lineartransfermatrix.h')]

    def __init__(self, Q_x=0, Q_y=0,
                     beta_x_0=1.0, beta_x_1=1.0, beta_y_0=1.0, beta_y_1=1.0,
                     alpha_x_0=0.0, alpha_x_1=0.0, alpha_y_0=0.0, alpha_y_1=0.0,
                     disp_x_0=0.0, disp_x_1=0.0, disp_y_0=0.0, disp_y_1=0.0,
                     Q_s=0.0, beta_s=1.0,
                     chroma_x=0.0, chroma_y=0.0,
                     detx_x=0.0, detx_y=0.0, dety_y=0.0, dety_x=0.0,
                     energy_increment=0.0, energy_ref_increment=0.0,
                     x_ref_0 = 0.0, px_ref_0 = 0.0, x_ref_1 = 0.0, px_ref_1 = 0.0,
                     y_ref_0 = 0.0, py_ref_0 = 0.0, y_ref_1 = 0.0, py_ref_1 = 0.0,
                     damping_rate_x = 0.0, damping_rate_y = 0.0, damping_rate_s = 0.0,
                     equ_emit_x = 0.0, equ_emit_y = 0.0, equ_emit_s = 0.0,
                     gauss_noise_ampl_x=0.0,gauss_noise_ampl_px=0.0,gauss_noise_ampl_y=0.0,gauss_noise_ampl_py=0.0,gauss_noise_ampl_zeta=0.0,gauss_noise_ampl_delta=0.0,
                     **nargs):

        _print('Warning: `LinearTransferMatrix` is deprecated and will be removed in the future. '
               'Please use `LineSegmentMap` instead.')
        if (chroma_x==0 and chroma_y==0
            and detx_x==0 and detx_y==0 and dety_y==0 and dety_x==0):

            cos_x = np.cos(2.0*np.pi*Q_x)
            sin_x = np.sin(2.0*np.pi*Q_x)
            cos_y = np.cos(2.0*np.pi*Q_y)
            sin_y = np.sin(2.0*np.pi*Q_y)

            nargs['no_detuning']  =  True
            nargs['q_x'] = sin_x
            nargs['q_y'] = sin_y
            nargs['chroma_x'] = cos_x
            nargs['chroma_y'] = cos_y
            nargs['detx_x'] = 0.
            nargs['detx_y'] = 0.
            nargs['dety_y'] = 0.
            nargs['dety_x'] = 0.
        else:
            nargs['no_detuning']  =  False
            nargs['q_x'] = Q_x
            nargs['q_y'] = Q_y
            nargs['chroma_x'] = chroma_x
            nargs['chroma_y'] = chroma_y
            nargs['detx_x'] = detx_x
            nargs['detx_y'] = detx_y
            nargs['dety_y'] = dety_y
            nargs['dety_x'] = dety_x

        if Q_s is not None:
            nargs['cos_s'] = np.cos(2.0*np.pi*Q_s)
            nargs['sin_s'] = np.sin(2.0*np.pi*Q_s)
        else:
            nargs['cos_s'] = 999
            nargs['sin_s'] = 0.

        nargs['beta_x_0'] = beta_x_0
        nargs['beta_y_0'] = beta_y_0
        nargs['beta_ratio_x'] = np.sqrt(beta_x_1/beta_x_0)
        nargs['beta_prod_x'] = np.sqrt(beta_x_1*beta_x_0)
        nargs['beta_ratio_y'] = np.sqrt(beta_y_1/beta_y_0)
        nargs['beta_prod_y'] = np.sqrt(beta_y_1*beta_y_0)
        nargs['alpha_x_0'] = alpha_x_0
        nargs['alpha_x_1'] = alpha_x_1
        nargs['alpha_y_0'] = alpha_y_0
        nargs['alpha_y_1'] = alpha_y_1
        nargs['disp_x_0'] = disp_x_0
        nargs['disp_x_1'] = disp_x_1
        nargs['disp_y_0'] = disp_y_0
        nargs['disp_y_1'] = disp_y_1
        nargs['beta_s'] = beta_s
        nargs['x_ref_0'] = x_ref_0
        nargs['x_ref_1'] = x_ref_1
        nargs['px_ref_0'] = px_ref_0
        nargs['px_ref_1'] = px_ref_1
        nargs['y_ref_0'] = y_ref_0
        nargs['y_ref_1'] = y_ref_1
        nargs['py_ref_0'] = py_ref_0
        nargs['py_ref_1'] = py_ref_1
        # acceleration with change of reference momentum
        nargs['energy_ref_increment'] = energy_ref_increment
        # acceleration without change of reference momentum
        nargs['energy_increment'] = energy_increment

        if damping_rate_x < 0.0 or damping_rate_y < 0.0 or damping_rate_s < 0.0:
            raise ValueError('Damping rates cannot be negative')
        if damping_rate_x > 0.0 or damping_rate_y > 0.0 or damping_rate_s > 0.0:
            nargs['uncorrelated_rad_damping'] = True
            nargs['damping_factor_x'] = 1.0-damping_rate_x/2.0
            nargs['damping_factor_y'] = 1.0-damping_rate_y/2.0
            nargs['damping_factor_s'] = 1.0-damping_rate_s/2.0
        else:
            nargs['uncorrelated_rad_damping'] = False

        if equ_emit_x < 0.0 or equ_emit_y < 0.0 or equ_emit_s < 0.0:
            raise ValueError('Equilibrium emittances cannot be negative')
        nargs['uncorrelated_gauss_noise'] = False
        nargs['gauss_noise_ampl_x'] = 0.0
        nargs['gauss_noise_ampl_px'] = 0.0
        nargs['gauss_noise_ampl_y'] = 0.0
        nargs['gauss_noise_ampl_py'] = 0.0
        nargs['gauss_noise_ampl_zeta'] = 0.0
        nargs['gauss_noise_ampl_delta'] = 0.0

        assert equ_emit_x >= 0.0
        assert equ_emit_y >= 0.0
        assert equ_emit_s >= 0.0

        if equ_emit_x > 0.0:
            assert alpha_x_1 == 0
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_px'] = np.sqrt(equ_emit_x*damping_rate_x/beta_x_1)
            nargs['gauss_noise_ampl_x'] = beta_x_1*nargs['gauss_noise_ampl_px']
        if equ_emit_y > 0.0:
            assert alpha_y_1 == 0
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_py'] = np.sqrt(equ_emit_y*damping_rate_y/beta_y_1)
            nargs['gauss_noise_ampl_y'] = beta_y_1*nargs['gauss_noise_ampl_py']
        if equ_emit_s > 0.0:
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_delta'] = np.sqrt(equ_emit_s*damping_rate_s/beta_s)
            nargs['gauss_noise_ampl_zeta'] = beta_s*nargs['gauss_noise_ampl_delta']

        assert gauss_noise_ampl_x >= 0.0
        assert gauss_noise_ampl_px >= 0.0
        assert gauss_noise_ampl_y >= 0.0
        assert gauss_noise_ampl_py >= 0.0
        assert gauss_noise_ampl_zeta >= 0.0
        assert gauss_noise_ampl_delta >= 0.0

        if gauss_noise_ampl_x > 0.0 or gauss_noise_ampl_px > 0.0 or gauss_noise_ampl_y > 0.0 or gauss_noise_ampl_py > 0.0 or gauss_noise_ampl_zeta > 0.0 or gauss_noise_ampl_delta > 0.0:
            nargs['uncorrelated_gauss_noise'] = True
            nargs['gauss_noise_ampl_x'] = np.sqrt(nargs['gauss_noise_ampl_x']**2+gauss_noise_ampl_x**2)
            nargs['gauss_noise_ampl_px'] = np.sqrt(nargs['gauss_noise_ampl_px']**2+gauss_noise_ampl_px**2)
            nargs['gauss_noise_ampl_y'] = np.sqrt(nargs['gauss_noise_ampl_y']**2+gauss_noise_ampl_y**2)
            nargs['gauss_noise_ampl_py'] = np.sqrt(nargs['gauss_noise_ampl_py']**2+gauss_noise_ampl_py**2)
            nargs['gauss_noise_ampl_zeta'] = np.sqrt(nargs['gauss_noise_ampl_zeta']**2+gauss_noise_ampl_zeta**2)
            nargs['gauss_noise_ampl_delta'] = np.sqrt(nargs['gauss_noise_ampl_delta']**2+gauss_noise_ampl_delta**2)

        super().__init__(**nargs)

    @property
    def Q_s(self):
        return np.arccos(self.cos_s) / (2*np.pi)

    @property
    def beta_x_1(self):
        return self.beta_prod_x*self.beta_ratio_x

    @property
    def beta_y_1(self):
        return self.beta_prod_y*self.beta_ratio_y


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
    if knob is None:
        return 0
    if hasattr(knob, '_expr'):
        if knob._expr is None:
            value = knob._get_value()
            if hasattr(value, 'get'): # For pyopencl scalars
                value = value.get()
            return value
        return knob._expr
    if isinstance(knob, Number):
        return knob
    raise ValueError(f'Cannot get expression for {knob}.')


def _nonzero(val_or_expr):
    if isinstance(val_or_expr, Number):
        return val_or_expr != 0

    return val_or_expr._expr


def _get_order(array):
    nonzero_indices = np.where(array)
    if not np.any(nonzero_indices):
        return 0
    return np.max(nonzero_indices)
