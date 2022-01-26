from pathlib import Path

import numpy as np
from scipy.special import factorial

import xobjects as xo
import xpart as xp

from ..base_element import BeamElement
from ..general import _pkg_root

class ReferenceEnergyIncrease(BeamElement):

    '''Beam element modeling a change of reference energy (acceleration, deceleration). Parameters:

             - Delta_p0c [eV]: Change in reference energy. Default is ``0``.
    '''

    _xofields = {
        'Delta_p0c': xo.Float64}

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(Delta_p0c=-self.Delta_p0c,
                              _context=_context, _buffer=_buffer, _offset=_offset)

ReferenceEnergyIncrease.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/referenceenergyincrease.h')]


class Drift(BeamElement):
    '''Beam element modeling a drift section. Parameters:

             - length [m]: Length of the drift section. Default is ``0``.
    '''

    _xofields = {
        'length': xo.Float64}
    isthick=True

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(length=-self.length,
                              _context=_context, _buffer=_buffer, _offset=_offset)

Drift.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h')]

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

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              voltage=-self.voltage,
                              frequency=self.frequency,
                              lag=self.lag,
                              _context=_context, _buffer=_buffer, _offset=_offset)

Cavity.XoStruct.extra_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/cavity.h')]


class XYShift(BeamElement):
    '''Beam element modeling an transverse shift of the reference system. Parameters:

             - dx [m]: Horizontal shift. Default is ``0``.
             - dy [m]: Vertical shift. Default is ``0``.

    '''
    _xofields = {
        'dx': xo.Float64,
        'dy': xo.Float64,
        }

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              dx=-self.dx, dy=-self.dy,
                              _context=_context, _buffer=_buffer, _offset=_offset)

XYShift.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xyshift.h')]


## ELECTRON LENS

class Elens(BeamElement):
# if array is needed we do it like this
#    _xofields={'inner_radius': xo.Float64[:]}
    _xofields={
               'current':      xo.Float64,
               'inner_radius': xo.Float64,
               'outer_radius': xo.Float64,
               'elens_length': xo.Float64,
               'voltage':      xo.Float64,
               'residual_kick_x': xo.Float64,
               'residual_kick_y': xo.Float64
              }

    def __init__(self,  inner_radius  = None,
                        outer_radius  = None,
                        current       = None,
                        elens_length  = None,
                        voltage       = None,
                        residual_kick_x = 0,
                        residual_kick_y = 0,
                        **kwargs):
        super().__init__(**kwargs)
        self.inner_radius    = inner_radius
        self.outer_radius    = outer_radius
        self.current         = current
        self.elens_length    = elens_length
        self.voltage         = voltage
        self.residual_kick_x   = residual_kick_x
        self.residual_kick_y   = residual_kick_y

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              current=self.current,
                              inner_radius=self.inner_radius,
                              outer_radius=self.outer_radius,
                              elens_length=-self.elens_length,
                              voltage=self.voltage,
                              _context=_context, _buffer=_buffer, _offset=_offset)

Elens.XoStruct.extra_sources = [
    _pkg_root.joinpath('beam_elements/elements_src/elens.h')]




class SRotation(BeamElement):
    '''Beam element modeling an rotation of the reference system around the s axis. Parameters:

                - angle [deg]: Rotation angle. Default is ``0``.

    '''

    _xofields={
        'cos_z': xo.Float64,
        'sin_z': xo.Float64,
        }

    def to_dict(self):
        dct = super().to_dict()
        dct['angle'] = self.angle
        return dct

    def __init__(self, angle=0, **nargs):
        anglerad = angle / 180 * np.pi
        nargs['cos_z']=np.cos(anglerad)
        nargs['sin_z']=np.sin(anglerad)
        super().__init__(**nargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_z, self.cos_z) * (180.0 / np.pi)

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              angle=-self.angle,
                              _context=_context, _buffer=_buffer, _offset=_offset)

SRotation.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/srotation.h')]


def _update_bal_from_knl_ksl(knl, ksl, bal, context=None):
    assert len(bal) == 2*len(knl) == 2*len(ksl)
    idx = np.array([ii for ii in range(0, len(knl))])
    inv_factorial = 1.0 / factorial(idx, exact=True)
    if context is not None:
        inv_factorial = context.nparray_to_context_array(inv_factorial)
    bal[0::2] = knl * inv_factorial
    bal[1::2] = ksl * inv_factorial

class Multipole(BeamElement):
    '''Beam element modeling a thin magnetic multipole. Parameters:

            - order [int]: Horizontal shift. Default is ``0``.
            - knl [m^-n, array]: Normalized integrated strength of the normal components.
            - ksl [m^-n, array]: Normalized integrated strength of the skew components.
            - hxl [rad]: Rotation angle of the reference trajectoryin the horizzontal plane.
            - hyl [rad]: Rotation angle of the reference trajectory in the vertical plane.
            - length [m]: Length of the originating thick multipole.

    '''

    _xofields={
        'order': xo.Int64,
        'length': xo.Float64,
        'hxl': xo.Float64,
        'hyl': xo.Float64,
        'radiation_flag': xo.Int64,
        'bal': xo.Float64[:],
        }

    def to_dict(self):
        dct = super().to_dict()
        dct['knl'] = self.knl
        dct['ksl'] = self.ksl
        return dct


    def __init__(self, order=None, knl=None, ksl=None, bal=None, **kwargs):

        if bal is None and (
            knl is not None or ksl is not None or order is not None
        ):
            if knl is None:
                knl = []
            if ksl is None:
                ksl = []
            if order is None:
                order = 0

            n = max((order + 1), max(len(knl), len(ksl)))
            assert n > 0

            _knl = np.array(knl)
            nknl = np.zeros(n, dtype=_knl.dtype)
            nknl[: len(knl)] = knl
            knl = nknl
            del _knl
            assert len(knl) == n

            _ksl = np.array(ksl)
            nksl = np.zeros(n, dtype=_ksl.dtype)
            nksl[: len(ksl)] = ksl
            ksl = nksl
            del _ksl
            assert len(ksl) == n

            order = n - 1
            bal = np.zeros(2 * order + 2)

            _update_bal_from_knl_ksl(knl, ksl, bal)

            kwargs["bal"] = bal
            kwargs["order"] = order

        elif bal is not None and len(bal) > 0:
            kwargs["bal"] = bal
            kwargs["order"] = (len(bal) - 2) // 2

        self.xoinitialize(**kwargs)

    @property
    def knl(self):
        bal_length = len(self.bal)
        idxes = np.array([ii for ii in range(0, bal_length, 2)])
        _bal = self._buffer.context.nparray_from_context_array(self.bal)
        _knl = self._buffer.context.nparray_to_context_array(np.array(
            [_bal[idx] * factorial(idx // 2, exact=True) for idx in idxes]))
        return self._buffer.context.linked_array_type.from_array(
                                        _knl,
                                        mode='setitem_from_container',
                                        container=self,
                                        container_setitem_name='_knl_setitem')

    @knl.setter
    def knl(self, value):
        self.knl[:] = value

    def _knl_setitem(self, indx, val):
        _knl = self.knl.copy()
        _knl[indx] = val
        _update_bal_from_knl_ksl(_knl, self.ksl, self.bal,
                                 context=self._buffer.context)

    @property
    def ksl(self):
        bal_length = len(self.bal)
        idxes = np.array([ii for ii in range(0, bal_length, 2)])
        _bal = self._buffer.context.nparray_from_context_array(self.bal)
        _ksl = self._buffer.context.nparray_to_context_array(np.array(
            [_bal[idx + 1] * factorial(idx // 2, exact=True) for idx in idxes]))
        return self._buffer.context.linked_array_type.from_array(
                                        _ksl,
                                        mode='setitem_from_container',
                                        container=self,
                                        container_setitem_name='_ksl_setitem')

    @ksl.setter
    def ksl(self, value):
        self.ksl[:] = value

    def _ksl_setitem(self, indx, val):
        _ksl = self.ksl.copy()
        _ksl[indx] = val
        _update_bal_from_knl_ksl(self.knl, _ksl, self.bal,
                                 context=self._buffer.context)

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              order=self.order,
                              length=-self.length,
                              hxl=-self.hxl,
                              hyl=-self.hyl,
                              radiation_flag=0, #TODO, I force radiation off for now
                              bal=-self.bal, # TODO: maybe it can be made more efficient
                              _context=_context, _buffer=_buffer, _offset=_offset)

Multipole.XoStruct.extra_sources = [
    xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
    _pkg_root.joinpath('headers/constants.h'),
    _pkg_root.joinpath('headers/synrad_spectrum.h'),
    _pkg_root.joinpath('beam_elements/elements_src/multipole.h')]


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
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'bal': xo.Float64[:],
        'phase': xo.Float64[:],
    }

    def to_dict(self):
        dct = super().to_dict()
        dct['knl'] = self.knl
        dct['ksl'] = self.ksl
        dct['pn'] = self.pn
        dct['ps'] = self.ps
        return dct

    def __init__(
        self,
        order=None,
        knl=None,
        ksl=None,
        pn=None,
        ps=None,
        bal=None,
        phase=None,
        **kwargs
    ):

        assert 'p' not in kwargs, "`p` in RF Multipole is not supported anymore"

        if bal is None and (
            knl is not None
            or ksl is not None
            or pn is not None
            or ps is not None
            or order is not None
        ):
            if knl is None:
                knl = []
            if ksl is None:
                ksl = []
            if pn is None:
                pn = []
            if ps is None:
                ps = []
            if order is None:
                order = 0

            n = max((order + 1), max(len(knl), len(ksl), len(pn), len(ps)))
            assert n > 0

            _knl = np.array(knl)
            nknl = np.zeros(n, dtype=_knl.dtype)
            nknl[: len(knl)] = knl
            knl = nknl
            del _knl
            assert len(knl) == n

            _ksl = np.array(ksl)
            nksl = np.zeros(n, dtype=_ksl.dtype)
            nksl[: len(ksl)] = ksl
            ksl = nksl
            del _ksl
            assert len(ksl) == n

            _pn = np.array(pn)
            npn = np.zeros(n, dtype=_pn.dtype)
            npn[: len(pn)] = pn
            pn = npn
            del _pn
            assert len(pn) == n

            _ps = np.array(ps)
            nps = np.zeros(n, dtype=_ps.dtype)
            nps[: len(ps)] = ps
            ps = nps
            del _ps
            assert len(ps) == n

            order = n - 1
            bal = np.zeros(2 * order + 2)
            phase = np.zeros(2 * order + 2)

            idx = np.array([ii for ii in range(0, len(knl))])
            inv_factorial = 1.0 / factorial(idx, exact=True)
            bal[0::2] = knl * inv_factorial
            bal[1::2] = ksl * inv_factorial

            phase[0::2] = pn
            phase[1::2] = ps

            kwargs["bal"] = bal
            kwargs["phase"] = phase
            kwargs["order"] = order

        elif (
            bal is not None
            #and bal
            and len(bal) >= 2
            and ((len(bal) % 2) == 0)
            and phase is not None
            #and phase
            and len(phase) >= 2
            and ((len(phase) % 2) == 0)
        ):
            kwargs["bal"] = bal
            kwargs["phase"] = phase
            kwargs["order"] = (len(bal) - 2) / 2
        elif '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            pass
        else:
            raise ValueError('RF Multipole Invalid input!')


        if '_xobject' in kwargs.keys() and kwargs['_xobject'] is not None:
            super().__init__(**kwargs)
        else:
            temp_bal = kwargs["bal"]
            temp_phase = kwargs["phase"]

            kwargs["bal"] = len(temp_bal)
            kwargs["phase"] = len(temp_phase)

            super().__init__(**kwargs)

            ctx = self._buffer.context
            self.bal[:] = ctx.nparray_to_context_array(temp_bal)
            self.phase[:] = ctx.nparray_to_context_array(temp_phase)

    @property
    def pn(self):
        _pn = self._buffer.context.nparray_to_context_array(np.array(
            [self._xobject.phase[ii] for ii in range(0, len(self.phase), 2)]))
        return self._buffer.context.linked_array_type.from_array(
                                        _pn, mode='readonly')

    @property
    def ps(self):
        _ps = self._buffer.context.nparray_to_context_array(np.array(
            [self._xobject.phase[ii+1] for ii in range(0, len(self.phase), 2)]))
        return self._buffer.context.linked_array_type.from_array(
                                        _ps, mode='readonly')

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              order=self.order,
                              voltage=-self.voltage,
                              frequency=self.frequency,
                              lag=self.lag,
                              bal=[-bb for bb in self.bal], # TODO: maybe it can be made more efficient
                              p = [pp for pp in self.phase],
                              _context=_context, _buffer=_buffer, _offset=_offset)

RFMultipole.knl = Multipole.knl
RFMultipole.ksl = Multipole.ksl
RFMultipole._knl_setitem = Multipole._knl_setitem
RFMultipole._ksl_setitem = Multipole._ksl_setitem

RFMultipole.XoStruct.extra_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/rfmultipole.h')]


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
            }

    def to_dict(self):
        dct = super().to_dict()
        dct['h'] = self.h
        dct['e1'] = self.e1
        dct['hgap'] = self.hgap
        dct['fint'] = self.fint
        return dct

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
        if r21 is None and r43 is None:
            ZERO = np.float64(0.0)
            if hgap is None:
                hgap = ZERO
            if h is None:
                h = ZERO
            if e1 is None:
                e1 = ZERO
            if fint is None:
                fint = ZERO

            # Check that the argument e1 is not too close to ( 2k + 1 ) * pi/2
            # so that the cos in the denominator of the r43 calculation and
            # the tan in the r21 calculations blow up
            assert not np.isclose(np.absolute(np.cos(e1)), ZERO)

            corr = np.float64(2.0) * h * hgap * fint
            r21 = h * np.tan(e1)
            temp = corr / np.cos(e1) * (np.float64(1) + np.sin(e1) * np.sin(e1))

            # again, the argument to the tan calculation should be limited
            assert not np.isclose(np.absolute(np.cos(e1 - temp)), ZERO)
            r43 = -h * np.tan(e1 - temp)

            self.h = h
            self.e1 = e1
            self.hgap = hgap
            self.fint = fint

        if r21 is not None and r43 is not None:
            kwargs['r21'] = r21
            kwargs['r43'] = r43
            super().__init__(**kwargs)
        else:
            raise ValueError(
                "DipoleEdge needs either coefficiants r21 and r43"
                " or suitable values for h, e1, hgap, and fint provided"
            )

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return self.__class__(
                              r21=-self.r21,
                              r43=-self.r43,
                              _context=_context, _buffer=_buffer, _offset=_offset)


DipoleEdge.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/dipoleedge.h')]


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
        }

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

LinearTransferMatrix.XoStruct.extra_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/lineartransfermatrix.h')]

class EnergyChange(BeamElement):
    _xofields={
        'energy_ref_increment': xo.Float64,
        'energy_increment': xo.Float64
        }

    def __init__(self, energy_increment=0.0,energy_ref_increment=0.0, **nargs):
        nargs['energy_ref_increment']=energy_ref_increment # acceleration with change of reference momentum (e.g. ramp)
        nargs['energy_increment']=energy_increment # acceleration without change of reference momentum (e.g. compensation of energy loss)
        super().__init__(**nargs)

EnergyChange.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/energychange.h')]


