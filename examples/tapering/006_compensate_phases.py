"""Compensate radiation loss with fixed magnets and active RF frequencies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt


# Same lattice and RF configuration as 004_taper_keep_mean.py.
line = xt.Line.from_json(
    Path(__file__).parent / '../../test_data/clic_dr/line_for_taper.json')
line.build_tracker()
line.particle_ref.p0c = 4e9
line['rf3'].voltage = 0
line['rf'].voltage = 0
for name in ('rf1', 'rf2a', 'rf2b'):
    line[name].voltage *= 2
line['rf1'].harmonic = line['rf1'].frequency * line.get_length() / clight
line['rf1'].frequency = 0

tw_no_rad = line.twiss(method='4d')
line.configure_radiation(model='mean')

# Only phase_taper is adjusted. Existing magnet settings are preserved.
# taper_magnets can also be True or a list of magnets to taper.
line.compensate_radiation_energy_loss(
    taper_magnets=False, delta0='zero_mean')
# To prescribe the starting momentum instead, use e.g. delta0=0.

tw = line.twiss5d(zeta0=0)
delta_mean = np.trapezoid(tw.delta, tw.s) / line.get_length()
xo.assert_allclose(delta_mean, 0, rtol=0, atol=1e-9)
for coord in ('x', 'px', 'y', 'py', 'delta'):
    values = getattr(tw, coord)
    xo.assert_allclose(values[-1], values[0], rtol=0, atol=2e-10)
xo.assert_allclose(line.attr['delta_taper'], 0, rtol=0, atol=0)
print(f'Mean delta: {delta_mean:.6g}')
print(f'Zeta slip per turn: {tw.zeta[-1] - tw.zeta[0]:.6g} m')

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
axes[0].plot(tw.s, tw.betx, label=r'$\beta_x$')
axes[0].plot(tw.s, tw.bety, label=r'$\beta_y$')
axes[0].plot(tw_no_rad.s, tw_no_rad.betx, '--', label=r'$\beta_x$, no radiation')
axes[0].plot(tw_no_rad.s, tw_no_rad.bety, '--', label=r'$\beta_y$, no radiation')
axes[0].set_ylabel(r'$\beta$ [m]')
axes[0].legend()
axes[1].plot(tw.s, tw.x, label='x')
axes[1].plot(tw.s, tw.y, label='y')
axes[1].set_ylabel('Orbit [m]')
axes[1].legend()
axes[2].plot(tw.s, tw.delta)
axes[2].set_ylabel(r'$\delta$')
axes[2].set_xlabel('s [m]')
fig.suptitle('Radiation compensated with fixed magnets')
fig.tight_layout()
plt.show()
