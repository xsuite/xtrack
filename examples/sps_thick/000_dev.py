from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')
mad.input('beam, particle=proton, pc=26;')
mad.call('../../test_data/sps_thick/lhc_q20.str')
mad.use(sequence='sps')
twmad = mad.twiss()

deferred_expressions = True

line = xt.Line.from_madx_sequence(
    sequence=mad.sequence.sps,
    deferred_expressions=deferred_expressions,
    allow_thick=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                 gamma0=mad.sequence.sps.beam.gamma)
line.build_tracker()
line.twiss_default['method'] = '4d'

# Check a bend
assert line.element_names[52] == 'mbb.10150_entry'
assert line.element_names[53] == 'mbb.10150_den'
assert line.element_names[54] == 'mbb.10150'
assert line.element_names[55] == 'mbb.10150_dex'
assert line.element_names[56] == 'mbb.10150_exit'



tw = line.twiss()


