import numpy as np

import xtrack as xt
import xpart as xp
import xdeps as xd

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_00_from_mad.json')
line.build_tracker()
line.twiss_default['method'] = '4d'

line.vars['k2bi1bsw1l11'] = 0
line.vars['k2bi1bsw1l12'] = 0
line.vars['k2bi1bsw1l13'] = 0
line.vars['k2bi1bsw1l14'] = 0
line.element_refs['bi1.bsw1l1.1'].k2 = line.vars['k2bi1bsw1l11']
line.element_refs['bi1.bsw1l1.2'].k2 = line.vars['k2bi1bsw1l12']
line.element_refs['bi1.bsw1l1.3'].k2 = line.vars['k2bi1bsw1l13']
line.element_refs['bi1.bsw1l1.4'].k2 = line.vars['k2bi1bsw1l14']

line.vars['bsw_k2l_ref'] = -9.7429e-02
line.vars['on_chicane_k2'] = 0

line.vars['k2bi1bsw1l11'] = (line.vars['on_chicane_k2'] * line.vars['bsw_k2l_ref']
                                / line['bi1.bsw1l1.1'].length)
line.vars['k2bi1bsw1l12'] = (-line.vars['on_chicane_k2'] * line.vars['bsw_k2l_ref']
                                / line['bi1.bsw1l1.2'].length)
line.vars['k2bi1bsw1l13'] = (-line.vars['on_chicane_k2'] * line.vars['bsw_k2l_ref']
                                / line['bi1.bsw1l1.3'].length)
line.vars['k2bi1bsw1l14'] = (line.vars['on_chicane_k2'] * line.vars['bsw_k2l_ref']
                                / line['bi1.bsw1l1.4'].length)

tw0 = line.twiss()

class FunctionPieceWiseLinear:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def __call__(self, x):
        return np.interp(x, self.x, self.y, left=self.y[0], right=self.y[-1])

    @classmethod
    def from_dict(cls, dct):
        return cls(dct['x'], dct['y'])

    def to_dict(self):
        return {'x': self.x, 'y': self.y}





k2_function = FunctionPieceWiseLinear(x=[1, 2, 9, 10], y=[0, 1, 1, 0])

x_test = np.linspace(0, 15, 1000)
y_test = k2_function(x_test)

functions = {'k2_vs_ms': k2_function}


mgr = line._xdeps_manager
mref = mgr.ref(functions, 'm')

line.vars['t_ms'] = 0.0

line.vars['on_chicane_k2'] = mref['k2_vs_ms'](line.vars['t_ms'])



plt.close('all')
plt.plot(x_test, y_test)

plt.show()
