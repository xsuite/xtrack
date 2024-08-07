import numpy as np

import numpy as np
import pandas as pd

import xtrack as xt
import xdeps as xd

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_01_with_chicane.json')
line.build_tracker()
line.twiss_default['method'] = '4d'

df = pd.read_csv('../../test_data/psb_chicane/chicane_collapse.csv',
                 delimiter=',', skipinitialspace=True)

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

line.to_json('psb_02_with_chicane_time_functions.json')

t_test = np.linspace(0, 6e-3, 100)

k0_bsw1 = []
k2l_bsw1 = []
k0_bsw2 = []
k2l_bsw2 = []
qx = []
qy = []
bety_at_qde3 = []
for ii, tt in enumerate(t_test):
    print(f'Twiss at t = {tt*1e3:.2f} ms   ', end='\r', flush=True)
    line.vars['t_turn_s'] = tt

    tw = line.twiss()

    qx.append(tw.qx)
    qy.append(tw.qy)
    bety_at_qde3.append(tw['bety', 'br.qde3'])
    k0_bsw1.append(line['bi1.bsw1l1.1'].k0)
    k2l_bsw1.append(line['bi1.bsw1l1.1'].knl[2])
    k0_bsw2.append(line['bi1.bsw1l1.2'].k0)
    k2l_bsw2.append(line['bi1.bsw1l1.2'].knl[2])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(2,1,1)
plt.plot(t_test*1e3, k0_bsw1, label='k0 bsw1')
plt.plot(t_test*1e3, k0_bsw2, label='k0 bsw2')
plt.legend()
plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, k2l_bsw1, label='k2l bsw1')
plt.plot(t_test*1e3, k2l_bsw2, label='k2l bsw2')
plt.legend()
plt.xlabel('time [ms]')

plt.figure(2)
sp1 = plt.subplot(2,1,1, sharex=sp1)
plt.plot(t_test*1e3, qy, label='qy')
plt.legend()
sp2 = plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, bety_at_qde3, label='bety at qde3')
plt.legend()
plt.xlabel('time [ms]')

plt.show()