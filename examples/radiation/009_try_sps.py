import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')

# mad.input('beam, particle=proton, pc=26;')
# mad.input('beam, particle=electron, pc=20;')

# # realistic
# mad.input('beam, particle=electron, pc=20;')
# v_mv = 25
# num_turns = 8000

# # higher energy
mad.input('beam, particle=electron, pc=50;')
v_mv = 250
num_turns = 500

mad.call('../../test_data/sps_thick/lhc_q20.str')

mad.use(sequence='sps')
mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10   # To stay in the linear region
mad.sequence.sps.elements['actcse.31632'].freq = 350 / 10  # having the same qs
mad.sequence.sps.elements['actcse.31632'].lag = 0.5

# # Some vertical orbit
# mad.sequence.sps.elements['mdv.10107'].kick = 100e-6

mad.input('twiss, table=tw6d;')
twm6d = mad.table.tw6d

mad.sequence.sps.beam.radiate = True
mad.emit()

line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True)
line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV,
                                    q0=-1, gamma0=mad.sequence.sps.beam.gamma)
line.build_tracker()
tw_thick = line.twiss()

Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot

line.discard_tracker()
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default
    Strategy(slicing=Teapot(2), element_type=xt.Bend),
    Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
]

line.slice_thick_elements(slicing_strategies)
line.build_tracker()

tw = line.twiss()

line.configure_radiation(model='mean')

# Switch off radiation in quadrupoles
tt = line.get_table()
tt_mult = tt.rows[tt.element_type=='Multipole']
for nn in tt_mult.name:
    if line[nn].order > 0:
        line[nn].radiation_flag = False
# Tapering!!!
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True, method='6d',
                    use_full_inverse=False)
tw_rad2 = line.twiss(eneloss_and_damping=True, method='6d',
                     radiation_method='full')

assert tw_rad.eq_gemitt_x is not None
assert tw_rad.eq_gemitt_y is not None
assert tw_rad.eq_gemitt_zeta is not None

assert tw_rad2.eq_gemitt_x is None
assert tw_rad2.eq_gemitt_y is None
assert tw_rad2.eq_gemitt_zeta is None

ex = tw_rad.eq_nemitt_x / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.eq_nemitt_y / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.eq_nemitt_zeta / (tw_rad.gamma0 * tw_rad.beta0)

line.configure_radiation(model='quantum')
for nn in tt_mult.name:
    if line[nn].order > 0:
        line[nn].radiation_flag = False
p = line.build_particles(num_particles=1000)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
line.track(p, num_turns=num_turns, time=True, turn_by_turn_monitor=True)
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
