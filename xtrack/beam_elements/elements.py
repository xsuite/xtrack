# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.special import factorial

import xobjects as xo
import xpart as xp

from ..base_element import BeamElement
from ..general import _pkg_root
from ..internal_record import RecordIndex, RecordIdentifier

class ReferenceEnergyIncrease(BeamElement):

    '''Beam element modeling a change of reference energy (acceleration, deceleration). Parameters:

             - Delta_p0c [eV]: Change in reference energy. Default is ``0``.
    '''

    _xofields = {
        'Delta_p0c': xo.Float64}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/referenceenergyincrease.h')]

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(Delta_p0c=-self.Delta_p0c,
                              _context=_context, _buffer=_buffer, _offset=_offset)

class Marker(BeamElement):
    """A marker beam element with no effect on the particles.

    Parameters:
        - name (str): Name of the element
    """

    _xofields = {
	'_dummy': xo.Int64}

    _extra_c_sources = [
        "/*gpufun*/\n"
        "void Marker_track_local_particle(MarkerData el, LocalParticle* part0){}"
    ]

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(_context=_context, _buffer=_buffer, _offset=_offset)

class Drift(BeamElement):
    '''Beam element modeling a drift section. Parameters:

             - length [m]: Length of the drift section. Default is ``0``.
    '''

    _xofields = {
        'length': xo.Float64}
    isthick=True
    behaves_like_drift=True

    _extra_c_sources = [_pkg_root.joinpath('beam_elements/elements_src/drift.h')]

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(length=-self.length,
                              _context=_context, _buffer=_buffer, _offset=_offset)


class Cavity(BeamElement):
    '''Beam element modeling an RF cavity. Parameters:

             - voltage [V]: Voltage of the RF cavity. Default is ``0``.
             - frequency [Hz]: Frequency of the RF cavity. Default is ``0``.
             - lag [deg]: Phase seen by the reference particle. Default is ``0``.
    '''

    _xofields = {
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/cavity.h')]

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              voltage=-self.voltage,
                              frequency=self.frequency,
                              lag=self.lag,
                              _context=_context, _buffer=_buffer, _offset=_offset)


class XYShift(BeamElement):
    '''Beam element modeling an transverse shift of the reference system. Parameters:

             - dx [m]: Horizontal shift. Default is ``0``.
             - dy [m]: Vertical shift. Default is ``0``.

    '''
    _xofields = {
        'dx': xo.Float64,
        'dy': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xyshift.h')]

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              dx=-self.dx, dy=-self.dy,
                              _context=_context, _buffer=_buffer, _offset=_offset)


class Elens(BeamElement):
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


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              current=self.current,
                              inner_radius=self.inner_radius,
                              outer_radius=self.outer_radius,
                              elens_length=-self.elens_length,
                              voltage=self.voltage,
                              coefficients_polynomial = self.coefficients_polynomial,
                              polynomial_order = self.polynomial_order,
                              _context=_context, _buffer=_buffer, _offset=_offset)


class Wire(BeamElement):

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


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        raise NotImplementedError


class SRotation(BeamElement):
    '''Beam element modeling an rotation of the reference system around the s axis. Parameters:

                - angle [deg]: Rotation angle. Default is ``0``.

    '''

    _xofields = {
        'cos_z': xo.Float64,
        'sin_z': xo.Float64,
        }

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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(angle=-self.angle,
                              _context=_context, _buffer=_buffer, _offset=_offset)


class XRotation(BeamElement):
    '''Beam element modeling an rotation of the reference system around the x axis. Parameters:

                - angle [deg]: Rotation angle. Default is ``0``.

    '''

    _xofields={
        'sin_angle': xo.Float64,
        'cos_angle': xo.Float64,
        'tan_angle': xo.Float64,
        }

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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(angle=-self.angle,
                              _context=_context, _buffer=_buffer, _offset=_offset)

class YRotation(BeamElement):
    '''Beam element modeling an rotation of the reference system around the y axis. Parameters:

                - angle [deg]: Rotation angle. Default is ``0``.

    '''

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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(angle=-self.angle,
                              _context=_context, _buffer=_buffer, _offset=_offset)



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
    '''Beam element modeling a thin magnetic multipole. Parameters:

            - order [int]: Horizontal shift. Default is ``0``.
            - knl [m^-n, array]: Normalized integrated strength of the normal components.
            - ksl [m^-n, array]: Normalized integrated strength of the skew components.
            - hxl [rad]: Rotation angle of the reference trajectory in the horizontal plane.
            - hyl [rad]: Rotation angle of the reference trajectory in the vertical plane.
            - length [m]: Length of the originating thick multipole.

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

    _extra_c_sources = [
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/multipole.h')]

    _internal_record_class = SynchrotronRadiationRecord

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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(
                              order=self.order,
                              length=-self.length,
                              hxl=-self.hxl,
                              hyl=-self.hyl,
                              radiation_flag=0, #TODO, I force radiation off for now
                              knl=-ctx2np(self.knl), # TODO: maybe it can be made more efficient
                              ksl=-ctx2np(self.ksl), # TODO: maybe it can be made more efficient
                              _context=_context, _buffer=_buffer, _offset=_offset)


class SimpleThinQuadrupole(BeamElement):
    """An optimised version of a quadrupole with zero knl[0], ksl, hxl, hyl, and length.
    Parameters:

            - knl [m^-n, array]: Normalized integrated strength of the normal components.

    """

    _xofields={
        'knl': xo.Float64[2],
    }

    _extra_c_sources = [
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(knl=-ctx2np(self.knl), _context=_context,
                              _buffer=_buffer, _offset=_offset)


class SimpleThinBend(BeamElement):
    """An optimised version of a dipole with zero ksl and hyl. Parameters:

            - knl [m^-n, array]: Normalized integrated strength of the normal components.
            - hxl [rad]: Rotation angle of the reference trajectory in the horizontal plane.
            - length [m]: Length of the originating thick multipole.

    """

    _xofields={
        'knl': xo.Float64[1],
        'hxl': xo.Float64,
        'length': xo.Float64,
    }

    _extra_c_sources = [
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(knl=-ctx2np(self.knl),
                              hxl=-self.hxl,
                              length=-self.length,
                              _context=_context, _buffer=_buffer, _offset=_offset)


class RFMultipole(BeamElement):
    '''Beam element modeling a thin modulated multipole, with strengths dependent on the z coordinate:

            kn(z) = k_n cos(2pi w tau + pn/180*pi)

            ks[n](z) = k_n cos(2pi w tau + pn/180*pi)

        Its parameters are:

            - order [int]: Horizontal shift. Default is ``0``.
            - frequency [Hz]: Frequency of the RF cavity. Default is ``0``.
            - knl [m^-n, array]: Normalized integrated strength of the normal components.
            - ksl [m^-n, array]: Normalized integrated strength of the skew components.
            - pn [deg, array]: Phase of the normal components.
            - ps [deg, array]: Phase of the skew components.
            - voltage [V]: Longitudinal voltage. Default is ``0``.
            - lag [deg]: Longitudinal phase seen by the reference particle. Default is ``0``.

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


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._context.nparray_from_context_array
        return self.__class__(
                              order=self.order,
                              voltage=-self.voltage,
                              frequency=self.frequency,
                              lag=self.lag,
                              knl=-ctx2np(self.knl),
                              ksl=-ctx2np(self.ksl),
                              pn = ctx2np(self.pn),
                              ps = ctx2np(self.ps),
                              _context=_context, _buffer=_buffer, _offset=_offset)


class DipoleEdge(BeamElement):
    '''Beam element modeling a dipole edge. Parameters:

            - h [1/m]: Curvature.
            - e1 [rad]: Face angle.
            - hgap [m]: Equivalent gap.
            - fint []: Fringe integral.

    '''

    _xofields = {
            'r21': xo.Float64,
            'r43': xo.Float64,
            'hgap': xo.Float64,
            'h': xo.Float64,
            'e1': xo.Float64,
            'fint': xo.Float64,
            }

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/dipoleedge.h')]

    _store_in_to_dict = ['h', 'e1', 'hgap', 'fint']
    _skip_in_to_dict = ['r21', 'r43']

    def __init__(
        self,
        r21=None,
        r43=None,
        h=None,
        e1=None,
        hgap=None,
        fint=None,
        **kwargs
    ):

        if r21 is not None or r43 is not None:
            raise NotImplementedError(
                "Please initialize using `h`, `e1`, `hgap` and `fint`")

        if hgap is None:
            hgap = 0.
        if h is None:
            h = 0.
        if e1 is None:
            e1 = 0.
        if fint is None:
            fint = 0.

        # Check that the argument e1 is not too close to ( 2k + 1 ) * pi/2
        # so that the cos in the denominator of the r43 calculation and
        # the tan in the r21 calculations blow up
        assert not np.isclose(np.absolute(np.cos(e1)), 0)

        corr = np.float64(2.0) * h * hgap * fint
        r21 = h * np.tan(e1)
        temp = corr / np.cos(e1) * (np.float64(1) + np.sin(e1) * np.sin(e1))

        # again, the argument to the tan calculation should be limited
        assert not np.isclose(np.absolute(np.cos(e1 - temp)), 0)
        r43 = -h * np.tan(e1 - temp)

        super().__init__(h=h, hgap=hgap, e1=e1, fint=fint, r21=r21, r43=r43,
                         **kwargs)


    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              h=self.h,
                              hgap=self.hgap,
                              e1=-self.e1,
                              fint=-self.fint,
                              _context=_context, _buffer=_buffer, _offset=_offset)


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

    _extra_c_sources = [
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
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



class FirstOrderTaylorMap(BeamElement):
    isthick = True

    _xofields={
        'radiation_flag': xo.Int64,
        'length': xo.Float64,
        'm0': xo.Float64[6],
        'm1': xo.Float64[6,6]}

    _extra_c_sources = [
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
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
