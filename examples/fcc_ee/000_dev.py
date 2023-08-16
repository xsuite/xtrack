import xtrack as xt

from cpymad.madx import Madx

mad = Madx()
mad.call('fccee_h.seq')
mad.beam(particle='positron', pc=120)
mad.use('fccee_p_ring')
twm = mad.twiss()

line_thick = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                  deferred_expressions=True)
line_thick.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                 gamma0=mad.sequence.fccee_p_ring.beam.gamma)
line_thick.build_tracker()
tw_thick_no_rad = line_thick.twiss()

line = line_thick.copy()

Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(3), element_type=xt.Bend),
    Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
    Strategy(slicing=Teapot(10), element_type=xt.Quadrupole),
    Strategy(slicing=Teapot(20), name=r'^qc\..*'),
    Strategy(slicing=Teapot(20), name=r'^sy\..*'),
    Strategy(slicing=Teapot(1), name=r'^mw\..*'),
]

line.slice_thick_elements(slicing_strategies=slicing_strategies)
line.build_tracker()