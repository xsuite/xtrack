import numpy as np

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

assert isinstance(line['mbb.10150_entry'], xt.Marker)
assert isinstance(line['mbb.10150_den'], xt.DipoleEdge)
assert isinstance(line['mbb.10150'], xt.Bend)
assert isinstance(line['mbb.10150_den'], xt.DipoleEdge)
assert isinstance(line['mbb.10150_exit'], xt.Marker)

assert line['mbb.10150_den'].model == 'linear'
assert line['mbb.10150_den'].side == 'entry'
assert line['mbb.10150_dex'].model == 'linear'
assert line['mbb.10150_dex'].side == 'exit'
assert line['mbb.10150_den']._linear_mode == 0
assert line['mbb.10150_dex']._linear_mode == 0
assert line['mbb.10150'].model == 'expanded'

ang = line['mbb.10150'].k0 * line['mbb.10150'].length
assert np.isclose(line['mbb.10150_den'].e1, ang / 2, atol=1e-11, rtol=0)
assert np.isclose(line['mbb.10150_dex'].e1, ang / 2, atol=1e-11, rtol=0)

tw0 = line.twiss()
assert np.isclose(twmad.s[-1], tw0.s[-1], atol=1e-11, rtol=0)

line.configure_bend_model(edge='full', core='full')

assert line['mbb.10150_den'].model == 'full'
assert line['mbb.10150_den'].side == 'entry'
assert line['mbb.10150_dex'].model == 'full'
assert line['mbb.10150_dex'].side == 'exit'
assert line['mbb.10150_den']._linear_mode == 0
assert line['mbb.10150_dex']._linear_mode == 0
assert line['mbb.10150'].model == 'full'

tw1 = line.twiss()

