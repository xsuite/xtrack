from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")

line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'], deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)
line.build_tracker()

line.to_json('test.json')
