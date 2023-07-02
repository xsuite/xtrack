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

tw = line.twiss()
assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-11, rtol=0)
assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)

line.configure_bend_model(edge='full', core='full')

tw = line.twiss()

assert line['mbb.10150_den'].model == 'full'
assert line['mbb.10150_den'].side == 'entry'
assert line['mbb.10150_dex'].model == 'full'
assert line['mbb.10150_dex'].side == 'exit'
assert line['mbb.10150_den']._linear_mode == 0
assert line['mbb.10150_dex']._linear_mode == 0
assert line['mbb.10150'].model == 'full'

assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-11, rtol=0)
assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.01)
assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.01)

line.configure_bend_model(core='expanded')

tw = line.twiss()

assert line['mbb.10150_den'].model == 'full'
assert line['mbb.10150_den'].side == 'entry'
assert line['mbb.10150_dex'].model == 'full'
assert line['mbb.10150_dex'].side == 'exit'
assert line['mbb.10150_den']._linear_mode == 0
assert line['mbb.10150_dex']._linear_mode == 0
assert line['mbb.10150'].model == 'expanded'

assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-11, rtol=0)
assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)

line.configure_bend_model(edge='linear')

assert line['mbb.10150_den'].model == 'linear'
assert line['mbb.10150_den'].side == 'entry'
assert line['mbb.10150_dex'].model == 'linear'
assert line['mbb.10150_dex'].side == 'exit'
assert line['mbb.10150_den']._linear_mode == 0
assert line['mbb.10150_dex']._linear_mode == 0
assert line['mbb.10150'].model == 'expanded'

line.configure_bend_model(core='full')
line.configure_bend_model(edge='full')

assert line['mbb.10150_den'].model == 'full'
assert line['mbb.10150_den'].side == 'entry'
assert line['mbb.10150_dex'].model == 'full'
assert line['mbb.10150_dex'].side == 'exit'
assert line['mbb.10150_den']._linear_mode == 0
assert line['mbb.10150_dex']._linear_mode == 0
assert line['mbb.10150'].model == 'full'

