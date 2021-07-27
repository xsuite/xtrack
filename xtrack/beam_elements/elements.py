from pathlib import Path

import numpy as np
import xobjects as xo
from scipy.special import factorial

from ..base_element import BeamElement
from ..particles import ParticlesData
from ..general import _pkg_root

class ReferenceEnergyIncrease(BeamElement):

    '''Beam element modeling a change of reference energy (acceleration, deceleration). Parameters:

             - Delta_p0c [eV]: Change in reference energy. Default is ``0``.
    '''

    _xofields = {
        'Delta_p0c': xo.Float64}

ReferenceEnergyIncrease.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/referenceenergyincrease.h')]


class Drift(BeamElement):
    '''Beam element modeling a drift section. Parameters:

             - length [m]: Length of the drift section. Default is ``0``.
    '''

    _xofields = {
        'length': xo.Float64}

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

XYShift.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xyshift.h')]


class SRotation(BeamElement):
    '''Beam element modeling an rotation of the reference system around the s axis. Parameters:

                - angle [deg]: Rotation angle. Default is ``0``.

    '''

    _xofields={
        'cos_z': xo.Float64,
        'sin_z': xo.Float64,
        }

    def __init__(self, angle=0, **nargs):
        anglerad = angle / 180 * np.pi
        nargs['cos_z']=np.cos(anglerad)
        nargs['sin_z']=np.sin(anglerad)
        super().__init__(**nargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_z, self.cos_z) * (180.0 / np.pi)

SRotation.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/srotation.h')]


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
        'bal': xo.Float64[:],
        }

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

            idx = np.array([ii for ii in range(0, len(knl))])
            inv_factorial = 1.0 / factorial(idx, exact=True)
            bal[0::2] = knl * inv_factorial
            bal[1::2] = ksl * inv_factorial

            kwargs["bal"] = bal
            kwargs["order"] = order

        elif bal is not None and len(bal) > 0:
            kwargs["bal"] = bal
            kwargs["order"] = (len(bal) - 2) // 2


        # TODO: Remove when xobjects is fixed
        kwargs["bal"] = list(kwargs['bal'])
        self._temp_bal_length = len(kwargs['bal'])

        self.xoinitialize(**kwargs)

    @property
    def knl(self):
        idxes = np.array([ii for ii in range(0, self._temp_bal_length, 2)])
        return [self.bal[idx] * factorial(idx // 2, exact=True) for idx in idxes]

    @property
    def ksl(self):
        idxes = np.array([ii for ii in range(0, self._temp_bal_length, 2)])
        return [self.bal[idx + 1] * factorial(idx // 2, exact=True) for idx in idxes]
        #idx = np.array([ii for ii in range(0, len(self.bal), 2)])
        #return self.bal[idx + 1] * factorial(idx // 2, exact=True)

Multipole.XoStruct.extra_sources = [
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

    def __init__(
        self,
        order=None,
        knl=None,
        ksl=None,
        pn=None,
        ps=None,
        bal=None,
        p=None,
        **kwargs
    ):
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
            p = np.zeros(2 * order + 2)

            idx = np.array([ii for ii in range(0, len(knl))])
            inv_factorial = 1.0 / factorial(idx, exact=True)
            bal[0::2] = knl * inv_factorial
            bal[1::2] = ksl * inv_factorial

            p[0::2] = pn
            p[1::2] = ps

            kwargs["bal"] = bal
            kwargs["phase"] = p
            kwargs["order"] = order

        elif (
            bal is not None
            and bal
            and len(bal) > 2
            and ((len(bal) % 2) == 0)
            and p is not None
            and p
            and len(p) > 2
            and ((len(p) % 2) == 0)
        ):
            kwargs["bal"] = bal
            kwargs["phase"] = p
            kwargs["order"] = (len(bal) - 2) / 2


        temp_bal = kwargs["bal"]
        temp_phase = kwargs["phase"]

        kwargs["bal"] = len(temp_bal)
        kwargs["phase"] = len(temp_phase)

        super().__init__(**kwargs)

        ctx = self._buffer.context
        self.bal[:] = ctx.nparray_to_context_array(temp_bal)
        self.phase[:] = ctx.nparray_to_context_array(temp_phase)

    @property
    def knl(self):
        idx = np.array([ii for ii in range(0, len(self.bal), 2)])
        return self.bal[idx] * factorial(idx // 2, exact=True)

    @property
    def ksl(self):
        idx = np.array([ii for ii in range(0, len(self.bal), 2)])
        return self.bal[idx + 1] * factorial(idx // 2, exact=True)

    def set_knl(self, value, order):
        assert order <= self.order
        self.bal[order * 2] = value / factorial(order, exact=True)

    def set_ksl(self, value, order):
        assert order <= self.order
        self.bal[order * 2 + 1] = value / factorial(order, exact=True)

    @property
    def pn(self):
        idx = np.array([ii for ii in range(0, len(self.p), 2)])
        return self.phase[idx]

    @property
    def ps(self):
        idx = np.array([ii for ii in range(0, len(self.p), 2)])
        return self.phase[idx + 1]

    def set_pn(self, value, order):
        assert order <= self.order
        self.phase[order * 2] = value

    def set_ps(self, value, order):
        assert order <= self.order
        self.phase[order * 2 + 1] = value

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

        if r21 is not None and r43 is not None:
            kwargs['r21'] = r21
            kwargs['r43'] = r43
            super().__init__(**kwargs)
        else:
            raise ValueError(
                "DipoleEdge needs either coefficiants r21 and r43"
                " or suitable values for h, e1, hgap, and fint provided"
            )


DipoleEdge.XoStruct.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/dipoleedge.h')]
