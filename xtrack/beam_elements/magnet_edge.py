# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import numpy as np
import xobjects as xo
from ..random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    _HasKnlKsl,
    _NOEXPR_FIELDS,
)
from ._edge import (
    EDGE_ACTIVE,
    EDGE_LINEAR,
    MAGNET_EDGE_MODELS,
    edge_bits_to_model,
    edge_model_to_bits,
)


class MagnetEdge(_HasKnlKsl, BeamElement):
    """Beam element modeling a magnet edge. Mostly used for testing purposes.

    Parameters
    ----------
    model : str
        "linear", "full", "dipole-only", "soft-only" or "suppressed".
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
    qint1, qint2 : float
        SAD soft quadrupole fringe integrals f1 (m), f2 (m^2), used by the
        models "full" and "soft-only".
    """
    isthick = True
    has_backtrack = True

    _xofields = {
        # Fringe mask of the face (edge_flags.h)
        '_flags': xo.Field(xo.Int64, default=EDGE_ACTIVE | EDGE_LINEAR),
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
        'qint1': xo.Float64,
        'qint2': xo.Float64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/magnet_edge.h"',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _repr_fields = [
        'model', 'is_exit', 'kn', 'ks', 'k_order', 'knl', 'ksl', 'kl_order',
        'length', 'half_gap', 'face_angle', 'face_angle_feed_down',
        'fringe_integral', 'qint1', 'qint2', 'delta_taper',
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
        if len(kwargs.get('kn', [])) > 4:
            raise ValueError(
                'MagnetEdge supports kn and ks up to k_order=3, like the magnets'
            )

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
        return edge_bits_to_model(MAGNET_EDGE_MODELS, int(self._flags))

    @model.setter
    def model(self, value):
        self._flags = EDGE_ACTIVE | edge_model_to_bits(
            MAGNET_EDGE_MODELS, value, 'MagnetEdge'
        )

    def to_dict(self, copy_to_cpu=True):
        # _HasKnlKsl.to_dict assumes a single `order`; this element has two.
        out = BeamElement.to_dict(self, copy_to_cpu=copy_to_cpu)

        out.pop('_flags', None)
        out['model'] = getattr(self, 'model')

        # See the comment in Multipole.to_dict about knl/ksl/order dumping
        for field in ['knl', 'ksl', 'kn', 'ks']:
            if field in out and np.allclose(out[field], 0, atol=1e-16):
                out.pop(field, None)

        if self.kl_order != -1 and 'knl' not in out and 'ksl' not in out:
            out['kl_order'] = self.kl_order

        if self.k_order != -1 and 'kn' not in out and 'ks' not in out:
            out['k_order'] = self.k_order

        out['is_exit'] = bool(self.is_exit)

        return out
