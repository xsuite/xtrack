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
mad.emit()

emitsumm = mad.table.emitsumm
# Get the emittances
emitsumm.ex, emitsumm.ey

line = xt.Line.from_madx_sequence(mad.sequence['RING'], allow_thick=True)
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV, q0=-1, gamma0=mad.sequence.ring.beam.gamma)
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(4), element_type=xt.Bend),
    Strategy(slicing=Teapot(4), element_type=xt.Quadrupole),
    Strategy(slicing=Teapot(21), name=r'^wig\..*'),
]

line.slice_thick_elements(slicing_strategies=slicing_strategies)

line.build_tracker()
line.twiss_default['method'] = '4d'
tw = line.twiss()

tt = line.get_table()

rho_inv = np.zeros(shape=(len(tt['s']),), dtype=np.float64)
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xt.Multipole):
        assert ee.length > 0 
        rho_inv[ii] = ee.knl[0] / ee.length