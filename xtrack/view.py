import numpy as np
from .table import Table

class View:
    def __init__(self, obj, ref, evaluator):
        if type(obj) is View:
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

    def get_value(self, key=None):
        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_value not supported for this object")

        if key is None:
            if hasattr(self._obj, "_xofields"):
                return {kk: self.get_value(kk) for kk in self._obj._xofields}
            else:
                return {kk: self.get_value(kk) for kk in dir(self._obj)}

        if hasattr(self._obj, "__iter__"):
            return self._obj[key]
        else:
            return getattr(self._obj, key)

    def get_expr(self, key=None, index=None):
        if index is not None:
            if key is None:
                raise ValueError('`key` must be provided when `index` is provided.')
            return getattr(self._ref, key)[index]._expr

        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            # is an array
            return self._ref[key]._expr

        if key is None:
            if hasattr(self._obj, "_xofields"):
                return {kk: self.get_expr(kk) for kk in self._obj._xofields}
            else:
                return {kk: self.get_expr(kk) for kk in dir(self._obj)}

        if hasattr(self._obj, "__iter__"):
            return self._ref[key]._expr
        else:
            return getattr(self._ref, key)._expr


    def get_info(self, key=None):
        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_info not supported for this object")

        if key is None:
            print("Element of type: ", self._obj.__class__.__name__)
            self.get_table().show(header=True, max_col_width=100)
        else:
            if hasattr(self._obj, "__iter__"):
                return self._ref[key]._info()
            else:
                return getattr(self._ref, key)._info()

    def get_table(self):

        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_table not supported for this object")

        out_expr = self.get_expr()
        out_value = self.get_value()

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