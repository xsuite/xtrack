from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')
mad.input('beam, particle=proton, pc=26;')
mad.call('../../test_data/sps_thick/lhc_q20.str')
mad.use(sequence='sps')
twmad = mad.twiss()

mad_sbend = Madx()
mad_sbend.call('../../test_data/sps_thick/sps_sbends.seq')
mad_sbend.input('beam, particle=proton, pc=26;')
mad_sbend.call('../../test_data/sps_thick/lhc_q20.str')
mad_sbend.use(sequence='sps')
twmad_sbend = mad_sbend.twiss()

deferred_expressions = True

line = xt.Line.from_madx_sequence(
    sequence=mad.sequence.sps,
    deferred_expressions=deferred_expressions,
    allow_thick=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                 gamma0=mad.sequence.sps.beam.gamma)
line.build_tracker()
line.twiss_default['method'] = '4d'

tw = line.twiss()


line_sbend = xt.Line.from_madx_sequence(
    sequence=mad_sbend.sequence.sps,
    deferred_expressions=deferred_expressions,
    allow_thick=True)
line_sbend.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                 gamma0=mad.sequence.sps.beam.gamma)
line_sbend.build_tracker()
line_sbend.twiss_default['method'] = '4d'

tw_sbend = line_sbend.twiss()


