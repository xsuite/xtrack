import numpy as np

import numpy as np
import pandas as pd

import xtrack as xt
import xpart as xp
import xdeps as xd

from xtrack.slicing import Teapot, Strategy

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_03_with_chicane_corrected.json')
line.build_tracker()

line.vars['on_chicane_k0'] = 1
line.vars['on_chicane_k2'] = 1
line.vars['on_chicane_beta_corr'] = 0
line.vars['on_chicane_tune_corr'] = 0

line_thick = line.copy()
line_thick.build_tracker()

line.discard_tracker()

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default
    Strategy(slicing=Teapot(2), element_type=xt.TrueBend),
    Strategy(slicing=Teapot(8), element_type=xt.CombinedFunctionMagnet),
]

print("Slicing thick elements...")
line.slice_in_place(slicing_strategies)
line.build_tracker()

tw_thin = line.twiss()
tw_thick = line_thick.twiss()

print('\n')
print(f'Qx: thick {tw_thin.qx:.4f} thin {tw_thick.qx:.4f}, diff {tw_thin.qx-tw_thick.qx:.4e}')
print(f'Qy: thick {tw_thin.qy:.4f} thin {tw_thick.qy:.4f}, diff {tw_thin.qy-tw_thick.qy:.4e}')
print(f"Q'x: thick {tw_thin.dqx:.4f} thin {tw_thick.dqx:.4f}, diff {tw_thin.dqx-tw_thick.dqx:.4f}")
print(f"Q'y: thick {tw_thin.dqy:.4f} thin {tw_thick.dqy:.4f}, diff {tw_thin.dqy-tw_thick.dqy:.4f}")

bety_interp = np.interp(tw_thick.s, tw_thin.s, tw_thin.bety)
print(f"Max beta beat: {np.max(np.abs(tw_thick.bety/bety_interp - 1)):.4e}")

plt.close('all')

delta_values = np.linspace(-0.001, 0.001, 100)
qy_thick_list = 0 * delta_values
qx_thick_list = 0 * delta_values
qy_thin_list = 0 * delta_values
qx_thin_list = 0 * delta_values

for ii, dd in enumerate(delta_values):
    tt_thin = line.twiss(delta0=dd)
    tt_thick = line_thick.twiss(delta0=dd)

    qy_thick_list[ii] = tt_thick.qy
    qx_thick_list[ii] = tt_thick.qx
    qy_thin_list[ii] = tt_thin.qy
    qx_thin_list[ii] = tt_thin.qx

plt.close('all')
plt.figure(1)

plt.plot(delta_values, qx_thin_list, label='qy thin')
plt.plot(delta_values, qx_thick_list, label='qy thick')
plt.plot(delta_values, qy_thin_list, label='qx thin')
plt.plot(delta_values, qy_thick_list, label='qx thick')

plt.xlabel('delta')
plt.ylabel('tune')
plt.show()

# plot difference
plt.figure(2)
plt.plot(delta_values, qx_thick_list - qx_thin_list, label='qx thick - thin')
plt.plot(delta_values, qy_thick_list - qy_thin_list, label='qy thick - thin')
plt.xlabel('delta')
plt.ylabel('tune')
plt.legend()
plt.show()
