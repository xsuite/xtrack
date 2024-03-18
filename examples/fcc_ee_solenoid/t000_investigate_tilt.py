import numpy as np
import xtrack as xt
import xobjects as xo
from scipy.constants import c as clight
from scipy.constants import e as qe

from cpymad.madx import Madx

fname = 'fccee_z'; pc_gev = 45.6
# fname = 'fccee_t'; pc_gev = 182.5

mad = Madx()
mad.call('../../test_data/fcc_ee/' + fname + '.seq')
mad.beam(particle='positron', pc=pc_gev)
mad.use('fccee_p_ring')

line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                 gamma0=mad.sequence.fccee_p_ring.beam.gamma)
line.cycle('ip.4', inplace=True)
line.append_element(element=xt.Marker(), name='ip.4.l')

Strategy = xt.Strategy
Teapot = xt.Teapot
slicing_strategies = [
    Strategy(slicing=None),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(3), element_type=xt.Bend),
    Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
    # Strategy(slicing=Teapot(50), element_type=xt.Quadrupole), # Starting point
    Strategy(slicing=Teapot(5), name=r'^qf.*'),
    Strategy(slicing=Teapot(5), name=r'^qd.*'),
    Strategy(slicing=Teapot(5), name=r'^qfg.*'),
    Strategy(slicing=Teapot(5), name=r'^qdg.*'),
    Strategy(slicing=Teapot(5), name=r'^ql.*'),
    Strategy(slicing=Teapot(5), name=r'^qs.*'),
    Strategy(slicing=Teapot(10), name=r'^qb.*'),
    Strategy(slicing=Teapot(10), name=r'^qg.*'),
    Strategy(slicing=Teapot(10), name=r'^qh.*'),
    Strategy(slicing=Teapot(10), name=r'^qi.*'),
    Strategy(slicing=Teapot(10), name=r'^qr.*'),
    Strategy(slicing=Teapot(10), name=r'^qu.*'),
    Strategy(slicing=Teapot(10), name=r'^qy.*'),
    Strategy(slicing=Teapot(50), name=r'^qa.*'),
    Strategy(slicing=Teapot(50), name=r'^qc.*'),
    Strategy(slicing=Teapot(20), name=r'^sy\..*'),
    Strategy(slicing=Teapot(30), name=r'^mwi\..*'),
]
line.discard_tracker()
line.slice_thick_elements(slicing_strategies=slicing_strategies)
line.build_tracker()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw = line.twiss(eneloss_and_damping=True)

print(tw.partition_numbers)