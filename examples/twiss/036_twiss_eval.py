import xtrack as xt
import numpy as np

# Add eval method to Table (to be moved to xtrack)
_TO_ADD = {}
_TO_ADD.update(xt.functions.Functions._mathfunctions)
for nn in xt.functions.Functions._mathfunctions:
    if hasattr(np, nn):
        _TO_ADD[nn] = getattr(np, nn)

def _eval(self, string, self_name = 'tw'):
    ddd = self._data.copy()
    ddd[self_name] = self
    ddd.update(_TO_ADD)
    ddd['np'] = np
    return eval(string, locals=ddd)
xt.Table._eval = _eval

# Load a line and twiss
env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
mytwiss = env.lhcb1.twiss4d()

#############################################
#### Examples of usage of the eval method ###
#############################################

# Normalized chromaticity
mytwiss._eval('dqx/qx') # 0.03

# Fractional tune
mytwiss._eval('qx % 1') # 0.31

# Beta in one point
mytwiss._eval('tw["betx", "ip1"]') # 0.15

# Phase difference between two points
mytwiss._eval('tw["mux", "ip5"] - tw["mux", "ip1"]') # 30.9305

# Normalized dispersion (all elements)
mytwiss._eval('dx/sqrt(betx)') # Full array

# Normalized dispersion (one point)
mytwiss._eval('tw["dx", "ip3"]/sqrt(tw["betx", "ip3"])') # -0.0463
