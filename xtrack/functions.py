import math
import numpy as np

import xdeps as xd

def frac(x):
    return x % 1

def sinc(x):
    return np.sinc(x / np.pi)

class Functions:

    _mathfunctions = dict(
        sqrt = math.sqrt,
        log = math.log,
        log10 = math.log10,
        exp = math.exp,
        sin = math.sin,
        cos = math.cos,
        tan = math.tan,
        asin = math.asin,
        acos = math.acos,
        atan = math.atan,
        atan2 = math.atan2,
        sinh = math.sinh,
        cosh = math.cosh,
        tanh = math.tanh,
        sinc = sinc,
        abs = math.fabs,
        erf = math.erf,
        erfc = math.erfc,
        floor = math.floor,
        ceil = math.ceil,
        round = np.round,
        frac = frac,
    )

    def __init__(self):
        object.__setattr__(self, '_funcs', {})

    def __setitem__(self, name, value):
        self._funcs[name] = value

    def __getitem__(self, name):
        if name in self._funcs:
            return self._funcs[name]
        elif name in self._mathfunctions:
            return self._mathfunctions[name]
        else:
            raise KeyError(f'Unknown function {name}')

    def __getattr__(self, name):
        if name == '_funcs':
            return object.__getattribute__(self, '_funcs')
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'Unknown function {name}')

    def update(self, other):
        self._funcs.update(other._funcs)

    def to_dict(self):
        fdict = {}
        for kk, ff in self._funcs.items():
            fdict[kk] = ff.to_dict()
            fdict[kk]['__class__'] = ff.__class__.__name__
        out = {'_funcs': fdict}
        return out

    @classmethod
    def from_dict(cls, dct):
        _funcs = {}
        for kk, ff in dct['_funcs'].items():
            ffcls = getattr(xd, ff.pop('__class__'))
            _funcs[kk] = ffcls.from_dict(ff)
        out = cls()
        out._funcs.update(_funcs)
        return out
