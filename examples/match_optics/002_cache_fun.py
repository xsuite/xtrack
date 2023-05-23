import time

import numpy as np

import xtrack as xt
import xdeps as xd

# xt._print.suppress = True

# Load the line
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

manager = collider._var_sharing.manager

f = manager.gen_fun('on_x1', on_x1=collider.vars['on_x1'])

fstr = manager.mk_fun('on_x1', on_x1=collider.vars['on_x1'])

class VarSetter:
    def __init__(self, multiline, varname):
        self.multiline = multiline
        self.varname = varname
        self.fstr = manager.mk_fun(varname, **{varname: multiline.vars[varname]})
        self.gbl = {k: r._owner for k, r in manager.containers.items()}
        self._build_fun()

    def _build_fun(self):
        lcl = {}
        exec(self.fstr, self.gbl.copy(), lcl)
        self.fun = lcl[self.varname]

    def __call__(self, value):
        self.fun(**{self.varname: value})

    def __getstate__(self):
        out = self.__dict__.copy()
        out.pop('fun')
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_fun()

vs = VarSetter(collider, 'on_x1')
collider._var_sharing = None

import pickle
vs2 = pickle.loads(pickle.dumps(vs))
collider2 = vs2.collider

vs2(123)
assert np.isclose(collider2['lhcb1'].twiss(method='4d')['px', 'ip1'], 123e-6, atol=1e-9, rtol=0)



