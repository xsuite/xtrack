# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from numbers import Number
from scipy.special import factorial

import xobjects as xo
import xtrack as xt

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
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_elem.h'),
        ]

    @property
    def _thin_slice_class(self):
        return None

    @property
    def _thick_slice_class(self):
        return xt.DriftSlice

    @property
    def _drift_slice_class(self):
        return xt.DriftSlice


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
        'lag_taper': xo.Float64,
        'absolute_time': xo.Int64,
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

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

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
        _pkg_root.joinpath('beam_elements/elements_src/elens.h')]

    def __init__(self, _xobject=None, **kwargs):
        super().__init__(_xobject=_xobject, **kwargs)
        if _xobject is None:
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
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/nonlinearlens.h'),
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
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/wire.h'),
    ]


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

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/track_srotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/srotation.h')]

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

    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xrotation.h')]

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
    '''Beam element modeling an rotation of the reference system around the y axis.

    Parameters
    ----------

    angle : float
        Rotation angle in degrees. Default is ``0``.

    '''

    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/track_yrotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/yrotation.h')
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
        #Note MAD-X node_value('other_bv ') is ignored
        #     minus sign follows MAD-X convention

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
        _pkg_root.joinpath('beam_elements/elements_src/zetashift.h')]

    _store_in_to_dict = ['dzeta']


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
    length : float
        Length of the originating thick multipole. Default is ``0``.

    '''

    _xofields={
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'length': xo.Float64,
        'hxl': xo.Float64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        }

    _rename = {
        'order': '_order',
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _depends_on = [RandomUniform, RandomExponential]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
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

        if 'hyl' in kwargs.keys():
            assert kwargs['hyl'] == 0.0, 'hyl is not supported anymore'

        len_knl = len(knl) if knl is not None else 0
        len_ksl = len(ksl) if ksl is not None else 0
        n = max((order + 1), max(len_knl, len_ksl))
        assert n > 0

        nknl = np.zeros(n, dtype=np.float64)
        nksl = np.zeros(n, dtype=np.float64)

        if knl is not None:
            if hasattr(knl, 'get'):
                knl = knl.get()
            nknl[: len(knl)] = np.array(knl)

        if ksl is not None:
            if hasattr(ksl, 'get'):
                ksl = ksl.get()
            nksl[: len(ksl)] = np.array(ksl)

        if 'delta_taper' not in kwargs.keys():
            kwargs['delta_taper'] = 0.0

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

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)

        # The constructor essentially overrides order if given knl or ksl
        # imply a higher one to the one given. Otherwise, knl and ksl are
        # resized, which at this stage means that the information about the
        # order (by which we understand the desired size of knl/ksl, which can
        # be different to the actual tracking order, as that can be changed
        # later) is essentially encoded in knl/ksl.
        # We should probably come up with a better way of handling this, but
        # in the meantime let's produce a minimal dict that allows to
        # reconstruct the xobject according to the rules outlined above.

        if 'knl' in out and np.allclose(out['knl'], 0, atol=1e-16):
            out.pop('knl', None)

        if 'ksl' in out and np.allclose(out['ksl'], 0, atol=1e-16):
            out.pop('ksl', None)

        if self.order != 0 and 'knl' not in out and 'ksl' not in out:
            out['order'] = self.order

        return out

    @property
    def hyl(self):
        raise ValueError("hyl is not anymore supported")

    @hyl.setter
    def hyl(self, value):
        raise ValueError("hyl is not anymore supported")


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
        _pkg_root.joinpath('beam_elements/elements_src/simplethinquadrupole.h')]

    def __init__(self, **kwargs):
        knl = kwargs.get('knl')
        if kwargs.get('_xobject') is None and knl is not None:
            if len(knl) != 2:
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


class Bend(BeamElement):
    """
    Implementation of combined function magnet (i.e. a bending magnet with
    a quadrupole component).

    Parameters
    ----------
    k0 : float
        Strength of the horizontal dipolar component in units of m^-1.
    k1 : float
        Strength of the horizontal quadrupolar component in units of m^-2.
    h : float
        Curvature of the reference trajectory in units of m^-1.
    length : float
        Length of the element in units of m.
    knl : array
        Integrated strength of the high-order normal multipolar components
        (knl[0] and knl[1] should not be used).
    ksl : array
        Integrated strength of the high-order skew multipolar components
        (ksl[0] and ksl[1] should not be used).
    num_multipole_kicks : int
        Number of multipole kicks used to model high order multipolar
        components.

    """

    isthick = True
    has_backtrack = True

    _xofields = {
        'k0': xo.Float64,
        'k1': xo.Float64,
        'h': xo.Float64,
        'length': xo.Float64,
        'model': xo.Int64,
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
        'knl': xo.Float64[5],
        'ksl': xo.Float64[5],
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
        'model': '_model',
        'edge_entry_model': '_edge_entry_model',
        'edge_exit_model': '_edge_exit_model',
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_yrotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/wedge_track.h'),
        _pkg_root.joinpath('beam_elements/elements_src/fringe_track.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_linear.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/bend.h'),
    ]

    def __init__(self, **kwargs):

        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)

        knl = kwargs.get('knl', np.array([]))
        ksl = kwargs.get('ksl', np.array([]))
        order_from_kl = max(len(knl), len(ksl)) - 1
        order = kwargs.get('order', max(4, order_from_kl))

        if order > 4:
            raise NotImplementedError # Untested

        kwargs['knl'] = np.pad(knl, (0, 5 - len(knl)), 'constant')
        kwargs['ksl'] = np.pad(ksl, (0, 5 - len(ksl)), 'constant')

        self.xoinitialize(**kwargs)

        if model is not None:
            self.model = model
        self.order = order

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)
        out.pop('_model')
        out['model'] = self.model

        # See the comment in Multiple.to_dict about knl/ksl/order dumping
        if 'knl' in out and np.allclose(out['knl'], 0, atol=1e-16):
            out.pop('knl', None)

        if 'ksl' in out and np.allclose(out['ksl'], 0, atol=1e-16):
            out.pop('ksl', None)

        if self.order != 0 and 'knl' not in out and 'ksl' not in out:
            out['order'] = self.order

        return out

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self.inv_factorial_order = 1.0 / factorial(value, exact=True)

    @property
    def model(self):
        return {
            0: 'adaptive',
            1: 'full',  # same as adaptive (for backward compatibility)
            2: 'bend-kick-bend',
            3: 'rot-kick-rot',
            4: 'expanded'
        }[self._model]

    @model.setter
    def model(self, value):
        assert value in ['adaptive', 'full', 'bend-kick-bend',
                            'rot-kick-rot', 'expanded']
        self._model = {
            'adaptive': 0,
            'full': 1,
            'bend-kick-bend': 2,
            'rot-kick-rot': 3,
            'expanded': 4
        }[value]

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

    @property
    def hxl(self): return self.h * self.length

    @property
    def radiation_flag(self): return 0.0

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
        return ['length', 'k0', 'k1', 'h', 'model', 'knl', 'ksl',
                'edge_entry_active', 'edge_exit_active', 'edge_entry_model',
                'edge_exit_model', 'edge_entry_angle', 'edge_exit_angle',
                'edge_entry_angle_fdown', 'edge_exit_angle_fdown',
                'edge_entry_fint', 'edge_exit_fint', 'edge_entry_hgap',
                'edge_exit_hgap', 'shift_x', 'shift_y', 'rot_s_rad']


class Sextupole(BeamElement):

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
    """

    isthick = True
    has_backtrack = True

    _xofields={
        'k2': xo.Float64,
        'k2s': xo.Float64,
        'length': xo.Float64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/sextupole.h'),
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


class Octupole(BeamElement):

    """
    Octupole element.

    Parameters
    ----------
    k3 : float
        Strength of the octupole component in m^-3.
    k3s : float
        Strength of the skew octupole component in m^-3.
    length : float
        Length of the element in meters.
    """

    isthick = True
    has_backtrack = True

    _xofields={
        'k3': xo.Float64,
        'k3s': xo.Float64,
        'length': xo.Float64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/octupole.h'),
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


class Quadrupole(BeamElement):
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
    """
    isthick = True
    has_backtrack = True

    _xofields = {
        'k1': xo.Float64,
        'k1s': xo.Float64,
        'length': xo.Float64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_srotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_quadrupole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/quadrupole.h'),
    ]

    def __init__(self, **kwargs):
        length = kwargs.get('length', 0)
        if kwargs.get('_xobject') is None and np.isclose(length, 0, atol=1e-13):
            raise ValueError("A thick element must have a non-zero length.")

        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'num_multipole_kicks' in dct:
            assert dct['num_multipole_kicks'] == 0
            dct.pop('num_multipole_kicks')
            dct.pop('knl', None)
            dct.pop('ksl', None)
            dct.pop('order', None)
            dct.pop('inv_factorial_order', None)

        return cls(**dct, **kwargs)

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


class Solenoid(BeamElement):
    """Solenoid element.

    Parameters
    ----------
    length : float
        Length of the element in meters.
    ks : float
        Strength of the solenoid component in rad / m. Only to be specified
        when the element is thin, i.e. when `length` is 0.
    ksi : float
        Integrated strength of the solenoid component in rad.
    """
    isthick = True
    has_backtrack = True

    _xofields = {
        'length': xo.Float64,
        'ks': xo.Float64,
        'ksi': xo.Float64,
        'radiation_flag': xo.Int64,
    }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_solenoid.h'),
        _pkg_root.joinpath('beam_elements/elements_src/solenoid.h'),
    ]

    _depends_on = [RandomUniform, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self, **kwargs):
        """Solenoid element.

        Parameters
        ----------
        length : float
            Length of the element in meters.
        ks : float
            Strength of the solenoid component in rad / m. Only to be specified
            when the element is thin, i.e. when `length` is 0.
        ksi : float
            Integrated strength of the solenoid component in rad.
        """
        if kwargs.get('_xobject') is not None:
            super().__init__(**kwargs)
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

        self.xoinitialize(**kwargs)

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceSolenoid


class CombinedFunctionMagnet:

    def __init__(self, *args, **kwargs):
        raise TypeError('`CombinedFunctionMagnet` is supported anymore. '
                        'Use `Bend` instead.')

    @classmethod
    def from_dict(cls, dct):
        return Bend(**dct)


class Fringe(BeamElement):
    """Fringe field element.

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
        _pkg_root.joinpath('beam_elements/elements_src/fringe_track.h'),
        _pkg_root.joinpath('beam_elements/elements_src/fringe.h'),
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
    }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/track_yrotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/wedge_track.h'),
        _pkg_root.joinpath('beam_elements/elements_src/wedge.h'),
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
        _pkg_root.joinpath('beam_elements/elements_src/simplethinbend.h')]

    def __init__(self, **kwargs):
        knl = kwargs.get('knl')
        if kwargs.get('_xobject') is None and knl is not None:
            if len(knl) != 1:
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


class RFMultipole(BeamElement):
    """Beam element modeling a thin modulated multipole, with strengths
    dependent on the z coordinate:

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
    """

    _xofields={
        'order': xo.Int64,
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

    def __init__(self, **kwargs):
        if 'p' in kwargs:
            raise ValueError("`p` in RF Multipole is not supported anymore")

        if 'bal' in kwargs:
            raise ValueError("`bal` in RF Multipole is not supported anymore")

        order = kwargs.get('order', 0)
        knl = np.array(kwargs.get('knl', [0]))
        ksl = np.array(kwargs.get('ksl', [0]))
        pn = np.array(kwargs.get('pn', [0]))
        ps = np.array(kwargs.get('ps', [0]))
        n = max(order + 1, len(knl), len(ksl), len(pn), len(ps))

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

        self.xoinitialize(**kwargs)


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
        _pkg_root.joinpath('beam_elements/elements_src/track_yrotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/wedge_track.h'),
        _pkg_root.joinpath('beam_elements/elements_src/fringe_track.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_linear.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
        _pkg_root.joinpath('beam_elements/elements_src/dipoleedge.h')]

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
        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            return

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


class LineSegmentMap(BeamElement):

    _xofields={
        'length': xo.Float64,

        'qx': xo.Float64,
        'qy': xo.Float64,

        'dqx': xo.Float64,
        'dqy': xo.Float64,
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
            det_xx=0.0, det_xy=0.0, det_yy=0.0, det_yx=0.0,
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
            Horizontal damping rate on the particles motion defined such that
            emit_x = emit_x(n=0) * exp(-damping_rate_x * n) where n is the turn
            number. Optional, default is ``0``.
        damping_rate_y : float
            Vertical damping rate on the particles motion defined such that
            emit_y = emit_y(n=0) * exp(-damping_rate_y * n) where n is the turn
            number. Optional, default is ``0``.
        damping_rate_s : float
            Longitudinal damping rate on the particles motion defined such that
            emit_s = emit_s(n=0) * exp(-damping_rate_s * n) where n is the turn
            number. Optional, default is ``0``.
        equ_emit_x : float
            Horizontal equilibrium emittance (geometric). Optional.
        equ_emit_y : float
            Vertical equilibrium emittance (geometric). Optional.
        equ_emit_s : float
            Longitudinal equilibrium emittance (geometric). Optional.
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

        if '_xobject' in nargs.keys() and nargs['_xobject'] is not None:
            self._xobject = nargs['_xobject']
            return

        assert longitudinal_mode in [
            'linear_fixed_qs', 'nonlinear', 'linear_fixed_rf', 'frozen', None]

        nargs['qx'] = qx
        nargs['qy'] = qy
        nargs['dqx'] = dqx
        nargs['dqy'] = dqy
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
    """First order Taylor map.

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
    """

    isthick = True

    _xofields = {
        'radiation_flag': xo.Int64,
        'length': xo.Float64,
        'm0': xo.Field(xo.Float64[6], default=np.zeros(6, dtype=np.float64)),
        'm1': xo.Field(xo.Float64[6, 6], default=np.eye(6, dtype=np.float64)),
    }

    _depends_on = [RandomUniform, RandomExponential]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/firstordertaylormap.h')]

    _internal_record_class = SynchrotronRadiationRecord # not functional,
    # included for compatibility with Multipole


class LinearTransferMatrix:
    def __init__(self, **kwargs):
        raise NotImplementedError(
            '`LinearTransferMatrix` is deprecated. Use `LineSegmentMap` instead.'
        )


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


def _get_order(array):
    nonzero_indices = np.where(array)
    if not np.any(nonzero_indices):
        return 0
    return np.max(nonzero_indices)


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
        _pkg_root.joinpath('beam_elements/elements_src/second_order_taylor_map.h')]

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

class ThinSliceNotNeededError(Exception):
    pass