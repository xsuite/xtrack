import numpy as np
import xtrack as xt
import xdeps as xd

from cpymad.madx import Madx

fname = 'fccee_z'; pc_gev = 45.6

mad = Madx()
mad.call(fname + '.seq')
mad.beam(particle='positron', pc=pc_gev)
mad.use('fccee_p_ring')

line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                 gamma0=mad.sequence.fccee_p_ring.beam.gamma)
line.build_tracker()

line.to_json(fname + '_thick.json')

