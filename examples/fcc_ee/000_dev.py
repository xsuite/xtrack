import xtrack as xt

from cpymad.madx import Madx

mad = Madx()
mad.call('fccee_h.seq')
mad.beam(particle='positron', pc=120)
mad.use('fccee_p_ring')
twm = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                 gamma0=mad.sequence.fccee_p_ring.beam.gamma)
line.build_tracker()
tw_thick_no_rad = line.twiss(method='4d')