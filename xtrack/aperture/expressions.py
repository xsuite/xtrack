"""Named, storage-independent references for aperture expressions."""

import numpy as np
import xobjects as xo
from xdeps.refs import BaseRef

from xtrack.view import View


def expression_value(env, value):
    if isinstance(value, str):
        value = env.new_expr(value)
    return (value._value, value) if isinstance(value, BaseRef) else (value, None)


class ApertureView(View):
    """Expression-aware view that resolves its object after every repacking."""

    def __init__(self, reference, evaluator):
        object.__setattr__(self, '_ref', reference)
        object.__setattr__(self, '_eval', evaluator)

    @property
    def _obj(self):
        return self._ref._value

    def _get_viewed_object(self):
        return self._obj._raw

    def __getattr__(self, key):
        value = getattr(self._obj, key)
        if isinstance(value, ApertureAccessor):
            return type(self)(getattr(self._ref, key), self._eval)
        return value

    def __getitem__(self, key):
        value = self._obj[key]
        if isinstance(value, ApertureAccessor):
            return type(self)(self._ref[key], self._eval)
        return value

    def __setattr__(self, key, value):
        # SVG descriptions are strings, not variable expressions.
        if key == 'svg_data':
            setattr(self._ref, key, value)
        else:
            super().__setattr__(key, value)


class ApertureAccessor:
    """Numeric side of an Xdeps reference; setters also invalidate geometry.

    References target this accessor, not a public View, to avoid recursively
    registering expressions while Xdeps evaluates a task.
    """

    def __init__(self, owner, kind, name, path=()):
        self.__dict__.update(_owner=owner, _kind=kind, _name=name, _path=path)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def _raw(self):
        raw = getattr(self._owner, f'_resolve_{self._kind}')(self._name)
        for is_item, key in self._path:
            raw = raw[key] if is_item else getattr(raw, key)
        return raw

    @property
    def __class__(self):
        return self._raw.__class__

    def _child(self, value, is_item, key):
        if isinstance(value, xo.Struct) or hasattr(value, '__setitem__'):
            return ApertureAccessor(self._owner, self._kind, self._name,
                                    self._path + ((is_item, key),))
        return value

    def __getattr__(self, key):
        return self._child(getattr(self._raw, key), False, key)

    def __getitem__(self, key):
        return self._child(self._raw[key], True, key)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return
        raw = self._raw
        old = getattr(raw, key)
        changed = not np.array_equal(old, value)
        if isinstance(value, View):
            value = value._get_viewed_object()
        setattr(raw, key, value)
        if changed:
            self._owner._touch(self._kind, self._name, key, self._path)

    def __setitem__(self, key, value):
        raw = self._raw
        changed = not np.array_equal(raw[key], value)
        raw[key] = value
        if changed:
            self._owner._touch(self._kind, self._name, None, self._path)

    def __len__(self):
        return len(self._raw)

    def __iter__(self):
        return iter(self._raw)

    def __array__(self, dtype=None, copy=None):
        return np.array(self._raw, dtype=dtype, copy=True)

    def __repr__(self):
        return repr(self._raw)


class ApertureRefMap:
    def __init__(self, owner, kind):
        self.owner = owner
        self.kind = kind

    def __getitem__(self, name):
        return ApertureAccessor(self.owner, self.kind, name)


class ApertureRefRoot:
    def __init__(self, owner):
        self.profiles = ApertureRefMap(owner, 'profile')
        self.pipes = ApertureRefMap(owner, 'pipe')
