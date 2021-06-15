from pathlib import Path

import numpy as np
import xobjects as xo
from scipy.special import factorial

from ..dress_element import dress_element
from ..particles import ParticlesData
from ..general import _pkg_root


class DriftData(xo.Struct):
    length = xo.Float64
DriftData.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h')]

class Drift(dress_element(DriftData)):
    '''The drift...'''
    pass

class CavityData(xo.Struct):
    voltage = xo.Float64
    frequency = xo.Float64
    lag = xo.Float64
CavityData.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/cavity.h')]

class Cavity(dress_element(CavityData)):
    pass

class XYShiftData(xo.Struct):
    dx = xo.Float64
    dy = xo.Float64
XYShiftData.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/xyshift.h')]

class XYShift(dress_element(XYShiftData)):
    pass

class SRotationData(xo.Struct):
    cos_z = xo.Float64
    sin_z = xo.Float64
SRotationData.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/srotation.h')]


class SRotation(dress_element(SRotationData)):

    def __init__(self, angle=0, **nargs):
        anglerad = angle / 180 * np.pi
        nargs['cos_z']=np.cos(anglerad)
        nargs['sin_z']=np.sin(anglerad)
        super().__init__(**nargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_z, self.cos_z) * (180.0 / np.pi)

class MultipoleData(xo.Struct):
    order = xo.Int64
    length = xo.Float64
    hxl = xo.Float64
    hyl = xo.Float64
    bal = xo.Float64[:]
MultipoleData.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/multipole.h')]

class Multipole(dress_element(MultipoleData)):

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

class RFMultipoleData(xo.Struct):
    order = xo.Int64
    voltage = xo.Float64
    frequency = xo.Float64
    lag = xo.Float64
    bal = xo.Float64[:]
    phase = xo.Float64[:]
RFMultipoleData.extra_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/rfmultipole.h')]

class RFMultipole(dress_element(RFMultipoleData)):

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

