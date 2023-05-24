import numpy as np

from cpymad.madx import Madx
import xpart as xp
import xtrack as xt
import xobjects as xo

mad = Madx()
mad.call('../../test_data/psb_injection/psb_injection.seq')
mad.use('psb')
twmad = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence['psb'])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1.,
                                 gamma0=mad.sequence['psb'].beam.gamma)

line.build_tracker()
beta0 = line.particle_ref.beta0[0]

tw = line.twiss()

# With the approximation beta ~= beta0 we have delta ~= pzeta ~= 1/beta0 ptau
# ==> ptau ~= beta0 delta ==> dptau / ddelta~= beta0

assert np.isclose(tw.dqx, twmad.summary.dq1 * beta0, rtol=0, atol=1e-6)
assert np.isclose(tw.dqy, twmad.summary.dq2 * beta0, rtol=0, atol=1e-6)

dx_ref = np.interp(tw.s, twmad.s, twmad.dx * beta0)
betx_ref = np.interp(tw.s, twmad.s, twmad.betx)
bety_ref = np.interp(tw.s, twmad.s, twmad.bety)

assert np.allclose(tw.dx, dx_ref, rtol=0, atol=1e-3)
assert np.allclose(tw.betx, betx_ref, rtol=1e-5, atol=0)
assert np.allclose(tw.bety, bety_ref, rtol=1e-5, atol=0)

assert np.isclose(tw.momentum_compaction_factor, twmad.summary.alfa, atol=0, rtol=5e-3)

