import numpy as np

import numpy as np
import pandas as pd

import xtrack as xt
import xpart as xp
import xdeps as xd

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_01_with_chicane.json')
line.build_tracker()
line.twiss_default['method'] = '4d'

df = pd.read_csv('chicane_collapse.csv', delimiter=',', skipinitialspace=True)

line.functions['fun_bsw_k0l'] = xd.FunctionPieceWiseLinear(
    x=df['time'].values, y=df['bsw_k0l'].values)
line.functions['fun_bsw_k2l'] = xd.FunctionPieceWiseLinear(
    x=df['time'].values, y=df['bsw_k2l'].values)

# Control knob with function
line.vars['on_chicane_k0'] = 1
line.vars['on_chicane_k2'] = 1
line.vars['bsw_k0l'] = (line.functions.fun_bsw_k0l(line.vars['t_turn_s'])
                        * line.vars['on_chicane_k0'])
line.vars['bsw_k2l'] = (line.functions.fun_bsw_k2l(line.vars['t_turn_s'])
                        * line.vars['on_chicane_k2'])

