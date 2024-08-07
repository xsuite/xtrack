import numpy as np
from scipy.constants import c as clight

import xtrack as xt
import xpart as xp
import xobjects as xo
from functools import partial

line = xt.Line.from_json(
    # '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
# line.cycle('ip1')
line.build_tracker()
line.vv['vrf400'] = 16

for vv in line.vars.get_table().rows[
    'on_x.*|on_sep.*|on_crab.*|on_alice|on_lhcb|corr_.*'].name:
    line.vars[vv] = 0

tw = line.twiss()

df_hz = -50
h_rf = 35640
f_rev = 1/tw.T_rev0
df_rev = df_hz / h_rf
eta = tw.slip_factor

f_rf0 = 1/tw.T_rev0 * h_rf

f_rf = f_rf0 + df_hz
line.vars['f_rf'] = f_rf
tt = line.get_table()
for nn in tt.rows[tt.element_type=='Cavity'].name:
    line.element_refs[nn].absolute_time = 1 # Need property
    line.element_refs[nn].frequency = line.vars['f_rf']

tw1 = line.twiss(search_for_t_rev=True)

f_rev_expected = f_rf / h_rf

assert np.isclose(f_rev_expected, 1/tw1.T_rev, atol=1e-5, rtol=0)
assert np.allclose(tw1.delta, tw1.delta[0], atol=1e-5, rtol=0) # Check that it is flat
delta_expected = -df_rev / f_rev / eta
assert np.allclose(tw1.delta, delta_expected, atol=2e-6, rtol=0)
tw_off_mom = line.twiss(method='4d', delta0=tw1.delta[0])
assert np.allclose(tw1.x, tw_off_mom.x, atol=1e-5, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw1.s, tw1.x*1e3, label='x')
plt.ylabel('x [mm]')
plt.grid()
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw1.s, tw1.delta*1e3, label='delta')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$ [$10^{-3}$]')
plt.ylim(-0.5, 0.5)
plt.grid()

plt.suptitle(r'$\Delta f_{\mathrm{rf}}$ = ' +f'{df_hz} Hz, '
             f'expexted $\delta$ = {delta_expected*1e3:.2f}e-3')

plt.show()