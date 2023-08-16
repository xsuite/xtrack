import numpy as np


from cpymad.madx import Madx
import xtrack as xt
from xtrack.slicing import Teapot, Strategy
import xpart as xp

# Import a thick sequence
mad = Madx()
mad.call('../../test_data/clic_dr/sequence.madx')
mad.use('ring')
twm = mad.twiss()

mad.sequence.ring.beam.radiate = True
mad.emit()

emitsumm = mad.table.emitsumm
# Get the emittances
emitsumm.ex, emitsumm.ey

line = xt.Line.from_madx_sequence(mad.sequence['RING'], allow_thick=True)
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV, q0=-1, gamma0=mad.sequence.ring.beam.gamma)
line.build_tracker()
tw_thick = line.twiss()
line.discard_tracker()

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(2), element_type=xt.Bend),
    Strategy(slicing=Teapot(5), element_type=xt.CombinedFunctionMagnet),
    Strategy(slicing=Teapot(10), element_type=xt.Quadrupole),
    Strategy(slicing=Teapot(1), name=r'^wig\..*'),
]

line.slice_thick_elements(slicing_strategies=slicing_strategies)

line.build_tracker()
tw = line.twiss()

# Tapering!!!
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)

ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=30)
line.track(p, num_turns=2000, time=True, turn_by_turn_monitor=True)
print(f'Tracking time: {line.time_last_track}')

# twe = tw.rows[:-1]
# cur_H_x = twe.gamx * twe.dx**2 + 2 * twe.alfx * twe.dx * twe.dpx + twe.betx * twe.dpx**2
# I5_x  = np.sum(cur_H_x * hh**3 * dl)
# I2_x = np.sum(hh**2 * dl)
# I4_x = np.sum(twe.dx * hh**3 * dl) # to be generalized for combined function magnets

# cur_H_y = twe.gamy * twe.dy**2 + 2 * twe.alfy * twe.dy * twe.dpy + twe.bety * twe.dpy**2
# I5_y  = np.sum(cur_H_y * hh**3 * dl)
# I2_y = np.sum(hh**2 * dl)
# I4_y = np.sum(twe.dy * hh**3 * dl) # to be generalized for combined function magnets

# lam_comp = 2.436e-12 # [m]
# ex_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_x / (I2_x - I4_x)
# ey_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_y / (I2_y - I4_y)

mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0))
spx.axhline(np.sqrt(ex * tw.betx[0] + ey * tw.betx2[0] + (np.std(p.delta) * tw.dx[0])**2), color='red')
# spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0))
spy.axhline(np.sqrt(ex * tw.bety1[0] + ey * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='red')
# spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw.betz0), color='red')

plt.show()