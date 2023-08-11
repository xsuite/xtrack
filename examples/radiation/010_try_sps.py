import numpy as np

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')
# mad.input('beam, particle=proton, pc=26;')
# mad.input('beam, particle=electron, pc=20;')

# mad.input('beam, particle=electron, pc=20;')
# v_mv = 25

mad.input('beam, particle=electron, pc=50;')
v_mv = 250

mad.call('../../test_data/sps_thick/lhc_q20.str')
mad.use(sequence='sps')
mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10   # To stay in the linear region
mad.sequence.sps.elements['actcse.31632'].freq = 350 / 10  # having the same qs
mad.sequence.sps.elements['actcse.31632'].lag = 0.5

# Some vertical orbit
mad.sequence.sps.elements['mdv.10107'].kick = 10e-6

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

# Tapering!!!
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True, method='6d',
                    use_full_inverse=False)

alpha_damp = tw_rad.damping_constants_turns[0]

tt = line.get_table()
rho_inv = np.zeros(shape=(len(tt['s']),), dtype=np.float64)
dl = np.zeros(shape=(len(tt['s']),), dtype=np.float64)
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xt.Multipole) and ee.knl[0] != 0:
        assert ee.length > 0
        rho_inv[ii] = ee.knl[0] / ee.length
        dl[ii] = ee.length

integ = np.sum(dl
    * np.abs(rho_inv)**3 * np.squeeze(tw.W_matrix[:-1, 4, 0]**2 + tw.W_matrix[:-1, 4, 1]**2))

mass0 = line.particle_ref.mass0
q0 = line.particle_ref.q0
gamma0 = line.particle_ref.gamma0[0]

from scipy.constants import c as clight
q_elect = 1.602176634e-19
emass = 0.51099895000
hbar = 6.582119569e-25; #/* GeV*s */

arad = 1e-10 * q0 * q0 * q_elect * clight * clight / emass # 1e-10 is guessed
clg = ((55.* hbar * clight) / (96 * np.sqrt(3))) * ((arad * gamma0**5) / emass)
ex = clg * integ / alpha_damp

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=100)
line.track(p, num_turns=500, time=True, turn_by_turn_monitor=True)

mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(np.std(mon.x, axis=0))
plt.axhline(np.sqrt(ex * tw.betx[0]))
plt.axhline(np.mean(np.std(mon.x, axis=0)[-100:]))

plt.figure(2)
plt.plot(np.std(mon.y, axis=0))
plt.axhline(np.mean(np.std(mon.y, axis=0)[-100:]))

plt.show()
