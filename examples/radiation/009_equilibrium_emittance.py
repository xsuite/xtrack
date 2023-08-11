import numpy as np
from scipy.constants import c as clight

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

prrrr

line.configure_radiation(model='mean')
tw_rad = line.twiss(eneloss_and_damping=True, method='6d',
                    use_full_inverse=False)

alpha_damp = tw_rad.damping_constants_turns[0]

tt = line.get_table()

rho_inv = np.zeros(shape=(len(tt['s']),), dtype=np.float64)
dl = np.zeros(shape=(len(tt['s']),), dtype=np.float64)
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xt.Multipole):
        assert ee.length > 0
        rho_inv[ii] = ee.knl[0] / ee.length
        dl[ii] = ee.length

# Get plank constant hbar in eV*s
hbar = 6.582119569e-16

r0 = line.particle_ref.get_classical_particle_radius0()
mass0 = line.particle_ref.mass0
q0 = line.particle_ref.q0
gamma0 = line.particle_ref.gamma0[0]
C_L = 55.0 / 48.0 * np.sqrt(3.0) * r0 * hbar / mass0

integ = np.sum(dl
    * np.abs(rho_inv)**3 * np.squeeze(tw.W_matrix[:-1, 4, 0]**2 + tw.W_matrix[:-1, 4, 1]**2))

eq_emitt = C_L / alpha_damp * gamma0**5 / clight * integ

q_elect = 1.602176634e-19
emass = 0.51099895000
hbar = 6.582119569e-25; #/* GeV*s */

arad = 1e-16 * q0 * q0 * q_elect * clight * clight / emass
clg = ((55.* hbar * clight) / (96 * np.sqrt(3))) * ((arad * gamma0**5) / emass)
ex = clg * integ / alpha_damp