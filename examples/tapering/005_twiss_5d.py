"""Find energy-balanced orbits while allowing a nonzero zeta slip per turn."""

from pathlib import Path

import numpy as np
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt


# Same lattice and RF configuration as 004_taper_keep_mean.py.
line = xt.Line.from_json(
    Path(__file__).parent / '../../test_data/clic_dr/line_for_taper.json')
line.particle_ref.p0c = 4e9
line.build_tracker()
line['rf3'].voltage = 0
line['rf'].voltage = 0
for name in ('rf1', 'rf2a', 'rf2b'):
    line[name].voltage *= 2
line['rf1'].harmonic = line['rf1'].frequency * line.get_length() / clight
line['rf1'].frequency = 0
line.configure_radiation(model='mean')

# Use the existing compensation routine to obtain initial cavity phases,
# then remove the magnet taper. The phases remain fixed during 5D Twiss.
line.compensate_radiation_energy_loss(delta0='zero_mean')
taper_setter = line.attr._cache['delta_taper'].multisetter
taper_setter.set_values(np.zeros_like(taper_setter.get_values()))

# Fix starting zeta (default: zero) and solve for starting delta.
tw_zeta0 = line.twiss(method='5d', zeta0=0)

# Alternatively, fix starting delta and solve for starting zeta.
tw_delta0 = line.twiss(method='5d', delta0=0)

for label, tw in [('zeta0=0', tw_zeta0), ('delta0=0', tw_delta0)]:
    for coord in ('x', 'px', 'y', 'py', 'delta'):
        values = getattr(tw, coord)
        xo.assert_allclose(values[-1], values[0], rtol=0, atol=2e-10)
    zeta_slip = tw.zeta[-1] - tw.zeta[0]
    t_rev = (tw.s[-1] - tw.s[0] - zeta_slip) / (tw.particle_on_co.beta0[0] * clight)
    print(f'{label}: delta_start={tw.delta[0]:.8g}, '
          f'zeta_start={tw.zeta[0]:.8g} m, '
          f'zeta_slip={zeta_slip:.8g} m, t_rev={t_rev:.12g} s')

# The optics use the full 6D map with the reference slip subtracted each turn.
# Ordinary multi-turn tracking does not apply that subtraction automatically.
