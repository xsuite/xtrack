# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import re
import warnings
from pathlib import Path

import xobjects as xo

from ..general import DEPRECATION_INFO_PREP_1_0

# Bits of the per-face edge mask, read from edge_flags.h (the one place they
# are defined); a bit missing there fails here, at import
_EDGE_FLAGS_H = Path(__file__).parent / 'elements_src' / 'edge_flags.h'
_EDGE_BITS = {
    name: int(value)
    for name, value in re.findall(
        r'^#define (EDGE_[A-Z_]+)\s+(\d+)\s', _EDGE_FLAGS_H.read_text(), re.MULTILINE
    )
}
EDGE_ACTIVE = _EDGE_BITS['EDGE_ACTIVE']
EDGE_LINEAR = _EDGE_BITS['EDGE_LINEAR']
EDGE_BEND = _EDGE_BITS['EDGE_BEND']
EDGE_MULT = _EDGE_BITS['EDGE_MULT']
EDGE_QSAD = _EDGE_BITS['EDGE_QSAD']

FACES = ('entry', 'exit')

# Parameters of the edges of each kind of magnet
# The knl[0] of a straight magnet gets the hard-edge dipole fringe only.
# Open question: should straight magnets also have fint, hgap (the soft dipole
# fringe, quadratic in k0)? Then they would use BEND_EDGE_PARAMS' fint, hgap and
# load with EDGE_LOAD_BEND (edge_load.h).
STRAIGHT_EDGE_PARAMS = ('qint1', 'qint2')
BEND_EDGE_PARAMS = ('angle', 'angle_fdown', 'fint', 'hgap', *STRAIGHT_EDGE_PARAMS)

# Deprecated: index stored in the `edge_*_model` field (`model` for
# MagnetEdge) of earlier versions; still read from old dicts and accepted by
# the setters, to be removed (names only)
_EDGE_MODEL_INDEX_TO_NAME = {
    -1: 'suppressed',
    0: 'linear',
    1: 'full',
    2: 'no-mult',
}

# Deprecated model names -> current name
_DEPRECATED_EDGE_MODELS = {
    'dipole-only': 'no-mult',
}

# Edge models of the magnets: name -> mask bits (without EDGE_ACTIVE, which
# is set by `edge_*_active`). Each fringe map acts only if its field is there:
# the dipole fringe if k0 != 0, the multipole fringe if a multipole of order
# >= 1 is nonzero, the soft quadrupole fringe if qint1 or qint2 != 0.
#   full: all the fringe maps
#   no-mult: all but the hard-edge multipole fringe
BEND_EDGE_MODELS = {
    'linear': EDGE_LINEAR,
    'full': EDGE_BEND | EDGE_MULT | EDGE_QSAD,
    'no-mult': EDGE_BEND | EDGE_QSAD,
    'suppressed': 0,
}
# Straight magnets (no face angle, fint, hgap): no linear model
STRAIGHT_EDGE_MODELS = {
    name: bits for name, bits in BEND_EDGE_MODELS.items() if name != 'linear'
}
# The standalone MagnetEdge has the models of the bends
MAGNET_EDGE_MODELS = BEND_EDGE_MODELS


def edge_model_to_bits(models, model, owner):
    """Mask bits of a model name (or index of earlier versions) in the
    table ``models``."""
    if not isinstance(model, str):
        model = _EDGE_MODEL_INDEX_TO_NAME.get(int(model), model)
    elif model in _DEPRECATED_EDGE_MODELS:
        warnings.warn(
            f'Edge model "{model}" is deprecated, use '
            f'"{_DEPRECATED_EDGE_MODELS[model]}".'
            + DEPRECATION_INFO_PREP_1_0,
            FutureWarning,
        )
        model = _DEPRECATED_EDGE_MODELS[model]
    try:
        return models[model]
    except KeyError:
        raise ValueError(
            f'Invalid edge model for {owner}: {model}. Available: '
            + ', '.join(f'"{mm}"' for mm in models)
        ) from None


def edge_bits_to_model(models, bits):
    """Name of the model with mask bits ``bits`` (EDGE_ACTIVE ignored) in
    the table ``models``; the bits if it is no named model."""
    bits &= ~EDGE_ACTIVE
    for name, model_bits in models.items():
        if model_bits == bits:
            return name
    return bits


def edge_xofields(default_flags, params=()):
    """Flat xofields of the edges of both faces.

    The edge model and the active switch of a face are stored in one mask,
    ``_edge_<face>_flags``; the parameters in ``edge_<face>_<param>``.
    """
    out = {}
    for face in FACES:
        out[f'_edge_{face}_flags'] = xo.Field(xo.Int64, default=default_flags)
    for face in FACES:
        for pp in params:
            out[f'edge_{face}_{pp}'] = xo.Float64
    return out


class _HasEdges:
    """Mixin for elements with an edge at each face.

    ``_edge_models`` maps the model names of the element to mask bits; None
    if the element only has ``edge_*_active``. The default mask is the one of
    the ``_edge_<face>_flags`` xofields.

    Each face has its own settings.

    Wanted, not implemented: an exit setting given as None would follow the
    entry one (live) until set again. It was removed as too large a change
    for now; see the notes at EDGE_ACTIVE in edge_flags.h.
    """

    _edge_models = None

    @classmethod
    def get_available_edge_models(cls):
        """List of the edge models of this element."""
        if cls._edge_models is None:
            return []
        return list(cls._edge_models)

    @classmethod
    def _edge_model_to_bits(cls, model):
        if not isinstance(model, str) and cls._edge_models is not BEND_EDGE_MODELS:
            raise ValueError(f'Invalid edge model for {cls.__name__}: {model}')
        return edge_model_to_bits(cls._edge_models, model, cls.__name__)

    @classmethod
    def _check_has_edge_model(cls, face):
        if cls._edge_models is None:
            raise AttributeError(f'`{cls.__name__}` has no `edge_{face}_model`')

    def _edge_flags(self, face):
        return int(getattr(self, f'_edge_{face}_flags'))

    def _get_edge_model(self, face):
        self._check_has_edge_model(face)
        return edge_bits_to_model(self._edge_models, self._edge_flags(face))

    @classmethod
    def _flags_with_model(cls, flags, model):
        return (int(flags) & EDGE_ACTIVE) | cls._edge_model_to_bits(model)

    @staticmethod
    def _flags_with_active(flags, value):
        flags = int(flags)
        return flags | EDGE_ACTIVE if value else flags & ~EDGE_ACTIVE

    def _set_edge_model(self, face, model):
        self._check_has_edge_model(face)
        name = f'_edge_{face}_flags'
        setattr(self, name, self._flags_with_model(getattr(self, name), model))

    def _get_edge_active(self, face):
        return int(bool(self._edge_flags(face) & EDGE_ACTIVE))

    def _set_edge_active(self, face, value):
        name = f'_edge_{face}_flags'
        setattr(self, name, self._flags_with_active(getattr(self, name), value))

    @classmethod
    def _edge_kwargs_to_flags(cls, kwargs):
        """Replace ``edge_<face>_model`` / ``edge_<face>_active`` in kwargs
        by the masks ``_edge_<face>_flags``."""
        kwargs = kwargs.copy()
        for face in FACES:
            name = f'_edge_{face}_flags'
            flags = int(kwargs.get(name, cls._xofields[name].default))
            if f'edge_{face}_model' in kwargs:
                cls._check_has_edge_model(face)
                flags = cls._flags_with_model(flags, kwargs.pop(f'edge_{face}_model'))
            if f'edge_{face}_active' in kwargs:
                flags = cls._flags_with_active(flags, kwargs.pop(f'edge_{face}_active'))
            kwargs[name] = flags
        return kwargs

    def _edges_to_dict(self, out):
        # Called by to_dict: the masks are written as `edge_*_model` and
        # `edge_*_active` (the keys of earlier versions), only when not at
        # the default
        for face in FACES:
            out.pop(f'_edge_{face}_flags', None)
            flags = int(getattr(self, f'_edge_{face}_flags'))
            default = self._xofields[f'_edge_{face}_flags'].default
            if (flags & EDGE_ACTIVE) != (default & EDGE_ACTIVE):
                out[f'edge_{face}_active'] = self._get_edge_active(face)
            if (
                self._edge_models is not None
                and flags & ~EDGE_ACTIVE != default & ~EDGE_ACTIVE
            ):
                out[f'edge_{face}_model'] = self._get_edge_model(face)


def _make_edge_properties(face):
    def get_model(self):
        return self._get_edge_model(face)

    def set_model(self, value):
        self._set_edge_model(face, value)

    def get_active(self):
        return self._get_edge_active(face)

    def set_active(self, value):
        self._set_edge_active(face, value)

    return property(get_model, set_model), property(get_active, set_active)


for _face in FACES:
    _model, _active = _make_edge_properties(_face)
    setattr(_HasEdges, f'edge_{_face}_model', _model)
    setattr(_HasEdges, f'edge_{_face}_active', _active)
