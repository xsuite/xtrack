from inspect import getattr_static
from warnings import warn

import numpy as np

from .general import _print, DEPRECATION_INFO_PREP_1_0
from .table import Table

class View:
    def __init__(self, obj, ref, evaluator):
        if issubclass(type(obj), View):
            obj = obj._get_viewed_object()
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_ref", ref)
        object.__setattr__(self, "_eval", evaluator)

    def _get_viewed_object(self):
        return object.__getattribute__(self, "_obj")

    @property
    def __class__(self):
        # return type("View",(self._obj.__class__,),{})
        return self._obj.__class__

    def _get_value(self, key=None):
        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_value not supported for this object")

        if key is None:
            if hasattr(self._obj, "_xofields"):
                return {kk: self._get_value(kk) for kk in self._obj._xofields}
            else:
                return {kk: self._get_value(kk) for kk in dir(self._obj)}

        if hasattr(self._obj, "__iter__"):
            return self._obj[key]
        else:
            return getattr(self._obj, key)

    def get_value(self, key=None):
        warn('`View.get_value` is deprecated and will be removed in a future version. '
             'Please inspect the reference using `env.ref[\'myobj\'].xdeps.value` instead.'
             + DEPRECATION_INFO_PREP_1_0, FutureWarning, stacklevel=2)
        return self._get_value(key)

    def _get_expr(self, key=None, index=None):
        if index is not None:
            if key is None:
                raise ValueError('`key` must be provided when `index` is provided.')
            return getattr(self._ref, key)[index]._expr

        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            # is an array
            return self._ref[key]._expr

        if key is None:
            if hasattr(self._obj, "_xofields"):
                return {kk: self._get_expr(kk) for kk in self._obj._xofields}
            else:
                return {kk: self._get_expr(kk) for kk in dir(self._obj)}

        if hasattr(self._obj, "__iter__"):
            return self._ref[key]._expr
        else:
            return getattr(self._ref, key)._expr

    def get_expr(self, key=None, index=None):
        warn('`View.get_expr` is deprecated and will be removed in a future version. '
             'Please inspect the reference using `env.ref[\'myobj\'].xdeps.expr` instead.'
             + DEPRECATION_INFO_PREP_1_0, FutureWarning, stacklevel=2)
        return self._get_expr(key, index)

    def _get_info(self, key=None):
        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_info not supported for this object")

        if key is None:
            _print("Element of type: ", self._obj.__class__.__name__)
            self.get_table().show(header=True, max_col_width=100)
        else:
            if hasattr(self._obj, "__iter__"):
                return self._ref[key]._info()
            else:
                return getattr(self._ref, key)._info()

    def get_info(self, key=None):
        warn('`View.get_info` is deprecated and will be removed in a future version. '
             'Please inspect the reference using `env.ref[\'myobj\'].xdeps.info()` instead.'
             + DEPRECATION_INFO_PREP_1_0, FutureWarning, stacklevel=2)
        return self._get_info(key)

    def get_table(self):

        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_table not supported for this object")

        out_expr = self._get_expr()
        out_value = self._get_value()

        value = [out_value[kk] for kk in out_expr.keys()]
        for ii, vv in enumerate(value):
            if not(np.isscalar(vv)):
                value[ii] = str(vv)

        data = {
            "name": np.array(list(out_expr.keys()), dtype=object),
            "value": np.array(value, dtype=object),
            "expr": np.array(
                [str(out_expr[kk]) for kk in out_expr.keys()], dtype=object
            ),
        }
        return Table(data)

    def __getattr__(self, key):
        val = getattr(self._obj, key)
        if hasattr(val, "__setitem__") and key != "extra": # extra is handled with no view
            return View(val, getattr(self._ref, key), self._eval)
        else:
            return val

    def __getitem__(self, key):
        val = self._obj[key]
        if hasattr(val, "__setitem__"):
            return View(val, self._ref[key], self._eval)
        else:
            return val

    def __setattr__(self, key, value):
        if isinstance(value, str):
            if hasattr(self._obj, "_noexpr_fields") and key in self._obj._noexpr_fields:
                value = value
            else:
                value = self._eval(value)
        setattr(self._ref, key, value)

    def __setitem__(self, key, value):
        if isinstance(value, str):
            value = self._eval(value)
        self._ref[key] = value

    def __repr__(self):
        return f"View of {self._obj!r}"

    def __dir__(self):
        return dir(self._obj)

    def __len__(self):
        return len(self._obj)

    def __iter__(self):
        return iter(self._obj)

    def __contains__(self, item):
        return item in self._obj

    def __add__(self, other):
        return self._obj + other

    def __radd__(self, other):
        return other + self._obj

    def __sub__(self, other):
        return self._obj - other

    def __rsub__(self, other):
        return other - self._obj

    def __mul__(self, other):
        return self._obj * other

    def __rmul__(self, other):
        return other * self._obj

    def __truediv__(self, other):
        return self._obj / other

    def __rtruediv__(self, other):
        return other / self._obj

    def __floordiv__(self, other):
        return self._obj // other

    def __rfloordiv__(self, other):
        return other // self._obj

    def __mod__(self, other):
        return self._obj % other

    def __rmod__(self, other):
        return other % self._obj

    def __pow__(self, other):
        return self._obj**other

    def __rpow__(self, other):
        return other**self._obj

    def __eq__(self, value: object) -> bool:
        return self._obj == value

    def __ne__(self, value: object) -> bool:
        return self._obj != value

    def __lt__(self, value: object) -> bool:
        return self._obj < value

    def __le__(self, value: object) -> bool:
        return self._obj <= value

    def __gt__(self, value: object) -> bool:
        return self._obj > value

    def __ge__(self, value: object) -> bool:
        return self._obj >= value


class LineElementView(View):
    """Element view carrying the line context, independently of expressions.

    Beam element classes can opt in by declaring ``_line_view_class`` to be
    this class or a subclass. The constructor receives the same object,
    reference and evaluator as View, plus the line and the element name.
    A missing reference means direct numerical access (no expressions).
    The line owns the element; the element never owns this view.
    """

    def __init__(self, obj, ref, evaluator, *, line, name):
        super().__init__(obj, ref, evaluator)
        object.__setattr__(self, "_view_line", line)
        object.__setattr__(self, "_view_name", name)

    def __getattr__(self, key):
        if self._ref is None:
            return getattr(self._obj, key)
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if key in ("_view_line", "_view_name"):
            raise AttributeError("Line view context is read-only.")
        # Honour properties/descriptors defined on a specialised view,
        # without invoking the descriptor at class-level lookup.
        descriptor = getattr_static(type(self), key, None)
        if hasattr(type(descriptor), "__set__"):
            object.__setattr__(self, key, value)
        elif self._ref is None:
            setattr(self._obj, key, value)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key):
        if self._ref is None:
            return self._obj[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if self._ref is None:
            self._obj[key] = value
        else:
            super().__setitem__(key, value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | {
            key for key in dir(type(self)) if not key.startswith("_")
        })

    def _get_expr(self, key=None, index=None):
        if self._ref is not None:
            return super()._get_expr(key, index)
        if key is None:
            if index is not None:
                raise ValueError("`key` must be provided when `index` is provided.")
            return {kk: None for kk in self._obj._xofields}
        return None

    def _get_info(self, key=None):
        if self._ref is not None:
            return super()._get_info(key)
        if key is None:
            return self.get_table().show(header=True, max_col_width=100)
        raise ValueError("No Xdeps reference manager is available.")
