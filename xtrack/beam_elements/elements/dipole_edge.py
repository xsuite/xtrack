# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import numpy as np
import xobjects as xo

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
        '#include "xtrack/beam_elements/elements_src/dipoleedge.h"',
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
