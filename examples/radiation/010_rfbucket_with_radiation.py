import xtrack as xt
import xobjects as xo

env = xt.load('../../test_data/fcc_ee/fccee_h.seq')
pc_gev = 120.

line = env.fccee_p_ring
line.set_particle_ref('positron', p0c=pc_gev*1e9)


tw_no_rad = line.twiss6d()
rfb_no_rad = line._get_bucket()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss6d(eneloss_and_damping=True)
rfb_rad = line._get_bucket()

# Check that the effect of the radiation is visible on qs
assert tw_no_rad.qs > 0.045
assert tw_rad.qs < 0.035

# Check consistency of qs and bets0 between twiss and rfb
xo.assert_allclose(rfb_no_rad.Q_s, tw_no_rad.qs, rtol=0.01)
xo.assert_allclose(rfb_rad.Q_s, tw_rad.qs, rtol=0.01)
xo.assert_allclose(rfb_no_rad.beta_z, tw_no_rad.bets0, rtol=0.015)
xo.assert_allclose(rfb_rad.beta_z, tw_rad.bets0, rtol=0.015)